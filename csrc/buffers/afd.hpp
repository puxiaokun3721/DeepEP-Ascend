#pragma once
#include <array>
#include <climits>
#include <algorithm>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>
#include <deep_ep/layout/afd/token.hpp>
#include <deep_ep/layout/afd/active.hpp>
#include <deep_ep/layout/afd/return.hpp>
#include <deep_ep/layout/afd/scheduler.hpp>
#include "../utils/tensor.hpp"
#include "../kernels/comm/hccl.hpp"
#include "../kernels/afd/dispatch.hpp"
#include "../kernels/afd/remaining.hpp"

// ### PATCH START: AFD owner, reuses HCCLContext and EPBuffer allocation/JIT conventions (3b25377d)
namespace deep_ep {
class AFDBuffer {
    std::shared_ptr<HCCLContext> payload_context, control_context;
    afd::AFDContextLayout config{};
    int num_ub_bytes;
    int64_t timeout_cycles;
    bool fp8, closed=false;
    int hidden;
    // ### PATCH START: receiver resources belong to the existing buffer owner
    int recv_rank=-1,recv_q=0,recv_q_alloc=0,recv_alignment=0,recv_tile=0;
    bool recv_prepared=false;
    torch::Tensor owned_active,owned_execution,owned_recv_workspace,owned_recv_x,owned_recv_metadata;
    std::optional<torch::Tensor> owned_recv_sf;
    // F state persists across replay; A return scratch is independent of dispatchSend.
    bool ffn_prepared=false,combine_recv_prepared=false,scheduler_precheck=false;
    bool combine_recv_round_step=false,source_launched=false;
    torch::Tensor owned_combine_recv_progress;
    int combine_tile=32,combine_chunk=0;
    torch::Tensor owned_scheduler_workspace,owned_completion,owned_psum,owned_counts;
    torch::Tensor owned_combine_x,owned_combine_send_workspace,owned_combine_recv_workspace,owned_recv_status;
    struct SourceRegistration {
        std::shared_ptr<HCCLContext> context;
        torch::Tensor tensor;
        std::vector<HcommMemHandle> handles;
        ~SourceRegistration() {
            // Owner destruction requires all launches to have completed, like native EPBuffer.
            for(size_t i=0;i<handles.size();++i)
                HCOMM_CHECK(HcommMemUnreg(context->jetty_endpoints[i],handles[i]));
        }
    };
    std::shared_ptr<SourceRegistration> combine_registration;
    AFDBuffer():num_ub_bytes(0),timeout_cycles(0),fp8(false),hidden(0) {}
    // ### PATCH END: receiver resources

    uint64_t known_allocation_bytes=0, transport_reserve_bytes=0, budget_bytes=0;
    torch::Tensor owned_context, owned_control, owned_ledger, owned_workspace, owned_status, owned_payload;

    static void check_bytes(const at::Tensor& value,uint64_t bytes) {
        EP_HOST_ASSERT(value.device().type()==TORCH_DEVICE_NPU && value.scalar_type()==torch::kByte);
        EP_HOST_ASSERT(value.is_contiguous() && value.dim()==1 && uint64_t(value.numel())==bytes);
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(value.data_ptr())%32==0);
    }
    static void check_owned(const at::Tensor& value,const at::Tensor& owner) {
        EP_HOST_ASSERT(value.data_ptr()==owner.data_ptr() && value.storage().unsafeGetStorageImpl()==owner.storage().unsafeGetStorageImpl());
        EP_HOST_ASSERT(value.sizes()==owner.sizes() && value.scalar_type()==owner.scalar_type() && value.is_contiguous());
        EP_HOST_ASSERT(value.device()==owner.device());
    }
    static uint64_t tensor_span_bytes(const at::Tensor& value) {
        if(value.numel()==0)return 0;
        uint64_t last=0;
        for(int i=0;i<value.dim();++i) {
            EP_HOST_ASSERT(value.stride(i)>=0);
            const uint64_t count=uint64_t(value.size(i)-1),stride=uint64_t(value.stride(i));
            EP_HOST_ASSERT(count==0 || stride<=(UINT64_MAX-last)/count);
            last+=count*stride;
        }
        EP_HOST_ASSERT(last<UINT64_MAX/value.element_size());
        return (last+1)*value.element_size();
    }
    void check_input_alias(const at::Tensor& input) const {
        const uint64_t begin=reinterpret_cast<uintptr_t>(input.data_ptr()),bytes=tensor_span_bytes(input);
        EP_HOST_ASSERT(bytes<=UINT64_MAX-begin);
        for(const auto* owner:{&owned_context,&owned_control,&owned_ledger,&owned_workspace,&owned_status,&owned_payload,&owned_combine_recv_progress}) {
            if(!owner->defined())continue;
            const uint64_t other=reinterpret_cast<uintptr_t>(owner->data_ptr()),other_bytes=owner->nbytes();
            EP_HOST_ASSERT(other_bytes<=UINT64_MAX-other);
            EP_HOST_ASSERT(bytes==0 || other_bytes==0 || begin+bytes<=other || other+other_bytes<=begin);
        }
    }
    // Registered source slots accumulate independently; reject byte-size overflow before allocation.
    static uint64_t checked_allocation_bytes(std::initializer_list<uint64_t> sizes) {
        uint64_t total=0;
        for(uint64_t bytes:sizes) {
            EP_HOST_ASSERT(bytes<=uint64_t(INT64_MAX)-total);
            total+=bytes;
        }
        return total;
    }
    static torch::Tensor byte_tensor(uint64_t bytes) {
        EP_HOST_ASSERT(bytes<=uint64_t(std::numeric_limits<int64_t>::max()));
        return torch::empty({int64_t(bytes)},torch::TensorOptions().dtype(torch::kByte).device(TORCH_DEVICE_NPU));
    }
    static void zero_tensor(const torch::Tensor& tensor) {
        ACL_CHECK(aclrtMemset(tensor.data_ptr(),tensor.nbytes(),0,tensor.nbytes()));
    }
public:
    // lane_specs rows: source rank, physical slot, DP, member, phase, per-slot round capacity C_m.
    // num_token_capacity is this rank full-transaction maxBS (allocated input extent), not C_m.
    // DP membership synchronizes rounds; it imposes no fixed aggregate token limit.
    // All ranks share topology/lane/layer tables and create payload then control in this order.
    // num_token_capacity is rank-local; FFN ranks carry an unused positive value for this field.
    // The mandatory transport reserve is an explicit estimate, NOT measured driver allocation usage.
    AFDBuffer(int rank_idx,const std::string& group_name,
        const std::vector<int>& src_domain_ranks,const std::vector<int>& ffn_domain_ranks,
        const std::vector<std::array<uint32_t,6>>& lane_specs,const std::vector<uint32_t>& layer_ids,
        int hidden,int num_sf_packs,int num_topk,int num_experts,int num_token_capacity,
        bool use_fp8,int num_vec_cores,int num_cpu_timeout_secs,int num_gpu_timeout_secs,
        uint64_t memory_budget_bytes,uint64_t transport_budget_reserve_bytes):
        num_ub_bytes(0),timeout_cycles(0),fp8(use_fp8),hidden(hidden),
        transport_reserve_bytes(transport_budget_reserve_bytes),budget_bytes(memory_budget_bytes) {
        using namespace afd;
        EP_HOST_ASSERT(!src_domain_ranks.empty() && !ffn_domain_ranks.empty());
        const int domain_size=int(src_domain_ranks.size()+ffn_domain_ranks.size());
        EP_HOST_ASSERT(domain_size%2==0 && rank_idx>=0 && rank_idx<domain_size);
        EP_HOST_ASSERT(domain_size<=kNumMaxRanks && !lane_specs.empty() && !layer_ids.empty());
        EP_HOST_ASSERT(lane_specs.size()<=INT_MAX && layer_ids.size()<=INT_MAX);
        std::set<int> ranks;
        for(int r:src_domain_ranks)EP_HOST_ASSERT(r>=0 && r<domain_size && ranks.insert(r).second);
        for(int r:ffn_domain_ranks)EP_HOST_ASSERT(r>=0 && r<domain_size && ranks.insert(r).second);
        const int n=int(ffn_domain_ranks.size());
        EP_HOST_ASSERT(num_experts>0 && num_experts<=512 && num_experts%n==0);
        EP_HOST_ASSERT(num_topk>0 && num_topk<=32 && num_topk<=num_experts);
        EP_HOST_ASSERT(hidden>0 && hidden%128==0 && num_token_capacity>0);
        EP_HOST_ASSERT(num_sf_packs>=0 && (use_fp8 ? num_sf_packs>0 : num_sf_packs==0));
        EP_HOST_ASSERT(num_cpu_timeout_secs>0 && num_gpu_timeout_secs>0);
        EP_HOST_ASSERT(memory_budget_bytes>0 && transport_budget_reserve_bytes>0);
        EP_HOST_ASSERT(num_vec_cores>0 && num_vec_cores<=runtime->get_num_vec_cores());
        EP_HOST_ASSERT(num_vec_cores<=64);
        const auto available_ub=runtime->get_num_ubuf_bytes_per_vec_core();
        EP_HOST_ASSERT(available_ub>0 && available_ub<=INT_MAX);
        num_ub_bytes=int(available_ub);
        timeout_cycles=int64_t(num_gpu_timeout_secs)*comm::kNumSystemCyclesPerSecond;
        const TokenLayout token{uint64_t(hidden)*(use_fp8?1:2),uint64_t(num_sf_packs)*2,uint64_t(num_topk)};
        EP_HOST_ASSERT(token.num_hidden_bytes<=INT_MAX && token.metadata_stride()<=INT_MAX);
        const uint64_t m=token.metadata_stride(),d=token.record_stride();
        std::vector<AFDLane> lanes;
        std::set<std::pair<uint32_t,uint32_t>> slots;
        std::map<std::pair<uint32_t,uint32_t>,std::pair<uint32_t,uint32_t>> dp_members;
        uint32_t cmax=0,max_slots=0;
        for(const auto& v:lane_specs) {
            EP_HOST_ASSERT(v[0]<src_domain_ranks.size() && v[5]>0 && v[5]<=kMaxLaneTokens);
            EP_HOST_ASSERT(v[1]<lane_specs.size() && slots.emplace(v[0],v[1]).second);
            EP_HOST_ASSERT(uint64_t(v[5])*num_topk<UINT32_MAX);
            // A physical member retains its source and C_m across microbatch slots.
            // Different members of one DP may use different C_m.
            auto key=std::make_pair(v[2],v[3]);
            auto identity=std::make_pair(v[0],v[5]);
            auto found=dp_members.find(key);
            EP_HOST_ASSERT(found==dp_members.end() || found->second==identity);
            dp_members[key]=identity;
            lanes.push_back({v[0],v[1],v[2],v[3],v[4],v[5],{0,0}});
            cmax=std::max(cmax,v[5]);max_slots=std::max(max_slots,v[1]+1);
        }
        uint32_t tile=0;
        for(uint32_t t:{32u,64u,128u,256u}) {
            const uint64_t ub=2*align32(4*uint64_t(num_experts))+2*align32(4*uint64_t(n))+
                2*align32(4*uint64_t(t)*num_topk)+align32(48*uint64_t(n))+96*uint64_t(num_vec_cores)*uint64_t(n)+512+256+2*align32(4*m);
            if(ub<=available_ub)tile=t;
        }
        EP_HOST_ASSERT(tile!=0);
        const auto source_it=std::find(src_domain_ranks.begin(),src_domain_ranks.end(),rank_idx);
        const bool is_source=source_it!=src_domain_ranks.end();
        const auto receiver_it=std::find(ffn_domain_ranks.begin(),ffn_domain_ranks.end(),rank_idx);
        if(receiver_it!=ffn_domain_ranks.end())recv_rank=int(receiver_it-ffn_domain_ranks.begin());
        const uint32_t src_idx=is_source?uint32_t(source_it-src_domain_ranks.begin()):UINT32_MAX;
        uint64_t f_payload_bytes=0,a_payload_bytes=0;
        std::vector<uint64_t> source_payload_bytes(src_domain_ranks.size(),0);
        std::vector<AFDReturnLane> return_lanes;
        // BF16 result rows need their actual hidden bytes plus transport alignment.
        const uint64_t return_stride=align512(2*uint64_t(hidden));
        std::vector<AFDAddress> addresses;
        for(uint32_t l=0;l<lanes.size();++l) {
            const uint64_t c=lanes[l].capacity;
            AFDAddress a{};a.records_offset=f_payload_bytes;f_payload_bytes=checked_allocation_bytes({f_payload_bytes,align512(c*d)});
            a.route_reference_offset=f_payload_bytes;f_payload_bytes=checked_allocation_bytes({f_payload_bytes,align32(4*c*std::min(num_topk,num_experts/n))});
            a.count_offset=f_payload_bytes;f_payload_bytes=checked_allocation_bytes({f_payload_bytes,align32(4*uint64_t(num_experts/n))});
            a.prefix_offset=f_payload_bytes;f_payload_bytes=checked_allocation_bytes({f_payload_bytes,align32(4*uint64_t(num_experts/n))});
            a.inbox_offset=uint64_t(l)*128;a.input_ready_offset=a.inbox_offset+64;
            for(int f=0;f<n;++f) {a.consumed_offset=(uint64_t(l)*n+f)*64;addresses.push_back(a);}
            // Reserve one aligned BF16 result per route in each local round window.
            auto& source_bytes=source_payload_bytes[lanes[l].src_rank_idx];
            return_lanes.push_back({source_bytes,uint64_t(l)*n*64+32,uint64_t(l)*128+96,0});
            source_bytes=checked_allocation_bytes({source_bytes,align512(c*num_topk*return_stride)});
        }
        if(is_source)a_payload_bytes=source_payload_bytes[src_idx];
        const uint64_t payload_bytes=is_source?std::max(uint64_t(512),a_payload_bytes):f_payload_bytes;
        const uint64_t control_bytes=align32(is_source?uint64_t(lanes.size())*n*64:uint64_t(lanes.size())*128);
        config.abi_version=kAbiVersion;config.num_ranks=n;config.num_src_ranks=src_domain_ranks.size();
        config.num_source_lanes=lanes.size();config.num_experts=num_experts;config.num_topk=num_topk;
        config.num_hidden_bytes=token.num_hidden_bytes;config.num_sf_packs=num_sf_packs;
        config.num_vec_cores=num_vec_cores;config.num_token_capacity=num_token_capacity;
        // Keep native per-call capacity naming: AFD calls cover a window, not the full transaction.
        config.num_max_tokens_per_rank=cmax;config.num_tokens_per_tile=tile;
        config.num_layers=layer_ids.size();config.src_rank_idx=src_idx;config.num_slots=max_slots;
        config.payload_bytes=payload_bytes;config.control_bytes=control_bytes;
        config.lanes_offset=sizeof(config);config.peers_offset=config.lanes_offset+align32(lanes.size()*sizeof(AFDLane));
        config.addresses_offset=config.peers_offset+align32(n*sizeof(AFDPeer));
        config.layers_offset=config.addresses_offset+align32(addresses.size()*sizeof(AFDAddress));
        config.context_bytes=config.layers_offset+align32(layer_ids.size()*sizeof(uint32_t));
        config.reserved_address=config.context_bytes;
        AFDReturnDirectory return_directory{config.reserved_address+32,
            config.reserved_address+32+32*uint64_t(src_domain_ranks.size()),return_stride,1,0};
        config.context_bytes=return_directory.lanes_offset+32*uint64_t(lanes.size());
        config.workspace=make_workspace(cmax,tile,num_topk,num_experts,n,m,max_slots);
        const uint64_t ledger_bytes=128*uint64_t(max_slots);
        const uint64_t app_bytes=checked_allocation_bytes({payload_bytes,control_bytes,config.context_bytes,config.workspace.num_bytes,ledger_bytes,32});
        // Grouped AFD layout keeps one channel per (peer, AIV); world2 stays legacy.
        const uint64_t channel_capacity=domain_size>2?uint64_t(domain_size-1)*uint64_t(num_vec_cores):
            uint64_t(std::max(num_vec_cores,domain_size-1));
        const uint64_t tables=2*channel_capacity*sizeof(ChannelHandle);
        EP_HOST_ASSERT(app_bytes<=budget_bytes && tables<=budget_bytes-app_bytes);
        known_allocation_bytes=app_bytes+tables;
        EP_HOST_ASSERT(transport_reserve_bytes<=budget_bytes-known_allocation_bytes);
        // Registration is exclusively initialization-time; all ranks use identical context order.
        payload_context=std::make_shared<HCCLContext>(group_name,rank_idx,domain_size,payload_bytes,
            num_vec_cores,num_cpu_timeout_secs,false,nullptr,true,true);
        control_context=std::make_shared<HCCLContext>(group_name,rank_idx,domain_size,control_bytes,
            num_vec_cores,num_cpu_timeout_secs,false,nullptr,true,true);
        // ### PATCH START: branch B channel stride (AFD private)
        EP_HOST_ASSERT(payload_context->channel_stride==control_context->channel_stride);
        config.channel_stride=payload_context->channel_stride;
        const uint32_t channel_base_stride=payload_context->channel_stride;
        auto peer_table_base=[&](int r)->uint32_t {
            if(r==rank_idx) return UINT32_MAX;
            const uint32_t base=uint32_t(r<rank_idx?r:r-1);
            return channel_base_stride?base*channel_base_stride:base;
        };
        // ### PATCH END: branch B channel stride (AFD private)
        std::vector<AFDPeer> peers(n);
        for(int f=0;f<n;++f) {
            const int r=ffn_domain_ranks[f];
            peers[f]={uint32_t(r),peer_table_base(r),payload_context->peer_buffer_bytes[r],
                uint64_t(control_context->peer_ptrs[r]),control_context->peer_buffer_bytes[r]};
            EP_HOST_ASSERT(peers[f].payload_bytes>=f_payload_bytes && peers[f].control_bytes>=lanes.size()*128);
        }
        std::vector<AFDPeer> source_peers(src_domain_ranks.size());
        for(uint32_t s=0;s<source_peers.size();++s) {
            const int r=src_domain_ranks[s];
            source_peers[s]={uint32_t(r),peer_table_base(r),
                payload_context->peer_buffer_bytes[r],uint64_t(control_context->peer_ptrs[r]),control_context->peer_buffer_bytes[r]};
            EP_HOST_ASSERT(source_peers[s].payload_bytes>=source_payload_bytes[s]);
            EP_HOST_ASSERT(source_peers[s].control_bytes>=uint64_t(lanes.size())*n*64);
        }
        std::vector<uint8_t> packed(config.context_bytes,0);
        std::memcpy(packed.data(),&config,sizeof(config));
        std::memcpy(packed.data()+config.lanes_offset,lanes.data(),lanes.size()*sizeof(AFDLane));
        std::memcpy(packed.data()+config.peers_offset,peers.data(),peers.size()*sizeof(AFDPeer));
        std::memcpy(packed.data()+config.addresses_offset,addresses.data(),addresses.size()*sizeof(AFDAddress));
        std::memcpy(packed.data()+config.layers_offset,layer_ids.data(),layer_ids.size()*sizeof(uint32_t));
        std::memcpy(packed.data()+config.reserved_address,&return_directory,sizeof(return_directory));
        std::memcpy(packed.data()+return_directory.source_peers_offset,source_peers.data(),source_peers.size()*sizeof(AFDPeer));
        std::memcpy(packed.data()+return_directory.lanes_offset,return_lanes.data(),return_lanes.size()*sizeof(AFDReturnLane));
        owned_context=byte_tensor(config.context_bytes);
        ACL_CHECK(aclrtMemcpy(owned_context.data_ptr(),config.context_bytes,packed.data(),packed.size(),ACL_MEMCPY_HOST_TO_DEVICE));
        auto options=torch::TensorOptions().dtype(torch::kByte).device(TORCH_DEVICE_NPU);
        owned_payload=torch::from_blob(payload_context->buffer,{int64_t(payload_bytes)},[owner=payload_context](void*){},options);
        owned_control=torch::from_blob(control_context->buffer,{int64_t(control_bytes)},[owner=control_context](void*){},options);
        owned_workspace=byte_tensor(config.workspace.num_bytes);zero_tensor(owned_workspace);
        owned_ledger=byte_tensor(ledger_bytes);zero_tensor(owned_ledger);
        owned_status=torch::empty({8},options.dtype(torch::kUInt32));zero_tensor(owned_status);
        // Initialize only owned incoming signal blocks. UBMEM peers must not start before all owners finish.
        std::vector<uint8_t> initial_control(control_bytes,0);
        for(uint32_t l=0;l<lanes.size();++l) {
            if(is_source)for(int f=0;f<n;++f) {
                const uint64_t off=(uint64_t(l)*n+f)*64;
                uint32_t one=1;std::memcpy(initial_control.data()+off,&one,4);std::memcpy(initial_control.data()+off+32,&one,4);
            } else {uint32_t one=1;std::memcpy(initial_control.data()+uint64_t(l)*128+64,&one,4);}
        }
        ACL_CHECK(aclrtMemcpy(owned_control.data_ptr(),control_bytes,initial_control.data(),control_bytes,ACL_MEMCPY_HOST_TO_DEVICE));
    }
    // Borrowed tensors retain their underlying owner. Callers must not mutate context or reset protocol state.
    torch::Tensor context()const{return owned_context;}
    torch::Tensor control_buffer()const{return owned_control;}
    torch::Tensor round_ledger()const{return owned_ledger;}
    torch::Tensor workspace()const{return owned_workspace;}
    torch::Tensor send_status()const{return owned_status;}
    torch::Tensor payload()const{return owned_payload;}
    std::array<uint64_t,3> memory_usage()const{return {known_allocation_bytes,transport_reserve_bytes,budget_bytes};}
    // All ranks must first stop launches and finish their transactions, including final ACK.
    // Synchronize before dropping owner references; outstanding exported tensors retain allocations.
    void destroy() {
        EP_HOST_ASSERT(!closed);
        // Teardown stage markers (diagnostic): role/stage/monotonic time/rc only.
        const auto afd_destroy_t0 = std::chrono::steady_clock::now().time_since_epoch();
        auto afd_stage = [&](const char* name, int rc) {
            const auto t_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch() - afd_destroy_t0).count();
            std::fprintf(stderr, "AFD_DESTROY role=%c stage=%s t_us=%lld rc=%d\n",
                         config.src_rank_idx != UINT32_MAX ? 'A' : 'F', name,
                         static_cast<long long>(t_us), rc);
            std::fflush(stderr);
        };
        afd_stage("device_sync_begin", 0);
        const auto afd_sync_rc = aclrtSynchronizeDevice();
        afd_stage("device_sync_end", static_cast<int>(afd_sync_rc));
        ACL_CHECK(afd_sync_rc);
        closed=true;
        owned_payload=torch::Tensor();owned_control=torch::Tensor();owned_context=torch::Tensor();
        owned_ledger=torch::Tensor();owned_workspace=torch::Tensor();owned_status=torch::Tensor();
        owned_active=torch::Tensor();owned_execution=torch::Tensor();owned_recv_workspace=torch::Tensor();
        owned_recv_x=torch::Tensor();owned_recv_metadata=torch::Tensor();owned_recv_sf.reset();
        afd_stage("combine_registration_reset_begin", 0);
        combine_registration.reset();
        afd_stage("combine_registration_reset_end", 0);
        owned_scheduler_workspace=torch::Tensor();owned_completion=torch::Tensor();
        owned_psum=torch::Tensor();owned_counts=torch::Tensor();owned_combine_x=torch::Tensor();
        owned_combine_send_workspace=torch::Tensor();owned_combine_recv_workspace=torch::Tensor();owned_recv_status=torch::Tensor();
        owned_combine_recv_progress=torch::Tensor();
        afd_stage("payload_context_reset_begin", 0);
        payload_context.reset();
        afd_stage("payload_context_reset_end", 0);
        afd_stage("control_context_reset_begin", 0);
        control_context.reset();
        afd_stage("control_context_reset_end", 0);
    }
    // ### PATCH START: receiver initialization and out-style public operation
    // Local fixture owner: same packed Context and kernel, no fake HCCL or scheduler.
    static std::shared_ptr<AFDBuffer> create_local_receiver(int num_ranks,int rank_idx,
        const std::vector<std::array<uint32_t,6>>& lane_specs,const std::vector<uint32_t>& layer_ids,
        int hidden,int num_sf_packs,int num_topk,int num_experts,int num_token_capacity,
        bool use_fp8,int num_vec_cores,uint64_t memory_budget_bytes) {
        using namespace afd;
        EP_HOST_ASSERT(num_ranks>0 && num_ranks<=kNumMaxRanks && rank_idx>=0 && rank_idx<num_ranks);
        EP_HOST_ASSERT(num_experts>0 && num_experts<=512 && num_experts%num_ranks==0);
        EP_HOST_ASSERT(num_topk>0 && num_topk<=32 && num_topk<=num_experts);
        EP_HOST_ASSERT(hidden>0 && hidden%128==0 && num_token_capacity>0);
        EP_HOST_ASSERT(use_fp8?num_sf_packs>0:num_sf_packs==0);
        EP_HOST_ASSERT(!lane_specs.empty() && lane_specs.size()<=INT_MAX && !layer_ids.empty() && layer_ids.size()<=INT_MAX);
        EP_HOST_ASSERT(num_vec_cores>0 && num_vec_cores<=64 && num_vec_cores<=runtime->get_num_vec_cores());
        auto owner=std::shared_ptr<AFDBuffer>(new AFDBuffer());
        owner->recv_rank=rank_idx;owner->hidden=hidden;owner->fp8=use_fp8;owner->budget_bytes=memory_budget_bytes;
        const auto available_ub=runtime->get_num_ubuf_bytes_per_vec_core();
        EP_HOST_ASSERT(available_ub>0 && available_ub<=INT_MAX);owner->num_ub_bytes=int(available_ub);
        const TokenLayout token{uint64_t(hidden)*(use_fp8?1:2),uint64_t(num_sf_packs)*2,uint64_t(num_topk)};
        EP_HOST_ASSERT(token.num_hidden_bytes<=INT_MAX && token.record_stride()<=INT_MAX);
        std::vector<AFDLane> lanes;std::vector<AFDAddress> addresses;
        std::set<std::pair<uint32_t,uint32_t>> slots;std::map<std::pair<uint32_t,uint32_t>,std::pair<uint32_t,uint32_t>> identities;
        uint64_t payload_bytes=0;uint32_t cmax=0,max_slots=0,num_src=0;
        for(uint32_t l=0;l<lane_specs.size();++l) {
            const auto v=lane_specs[l];
            EP_HOST_ASSERT(v[0]<kNumMaxRanks && v[1]<lane_specs.size() && v[5]>0 && v[5]<=kMaxLaneTokens && slots.emplace(v[0],v[1]).second);
            EP_HOST_ASSERT(uint64_t(v[5])*num_topk<UINT32_MAX);
            const auto key=std::make_pair(v[2],v[3]),identity=std::make_pair(v[0],v[5]);
            auto found=identities.find(key);EP_HOST_ASSERT(found==identities.end() || found->second==identity);
            identities[key]=identity;
            lanes.push_back({v[0],v[1],v[2],v[3],v[4],v[5],{0,0}});
            AFDAddress address{};address.records_offset=payload_bytes;payload_bytes=checked_allocation_bytes({payload_bytes,align512(uint64_t(v[5])*token.record_stride())});
            address.route_reference_offset=payload_bytes;payload_bytes=checked_allocation_bytes({payload_bytes,align32(4*uint64_t(v[5])*std::min(num_topk,num_experts/num_ranks))});
            address.count_offset=payload_bytes;payload_bytes=checked_allocation_bytes({payload_bytes,align32(4*uint64_t(num_experts/num_ranks))});
            address.prefix_offset=payload_bytes;payload_bytes=checked_allocation_bytes({payload_bytes,align32(4*uint64_t(num_experts/num_ranks))});
            address.inbox_offset=uint64_t(l)*128;address.input_ready_offset=address.inbox_offset+64;
            for(int f=0;f<num_ranks;++f)addresses.push_back(address);
            cmax=std::max(cmax,v[5]);max_slots=std::max(max_slots,v[1]+1);num_src=std::max(num_src,v[0]+1);
        }
        auto& c=owner->config;c.abi_version=kAbiVersion;c.num_ranks=num_ranks;c.num_src_ranks=num_src;
        c.num_source_lanes=lanes.size();c.num_experts=num_experts;c.num_topk=num_topk;
        c.num_hidden_bytes=token.num_hidden_bytes;c.num_sf_packs=num_sf_packs;c.num_vec_cores=num_vec_cores;
        c.num_token_capacity=num_token_capacity;c.num_max_tokens_per_rank=cmax;c.num_layers=layer_ids.size();
        c.src_rank_idx=UINT32_MAX;c.num_slots=max_slots;c.payload_bytes=payload_bytes;c.control_bytes=128*uint64_t(lanes.size());
        c.lanes_offset=sizeof(c);c.peers_offset=c.lanes_offset+32*uint64_t(lanes.size());
        c.addresses_offset=c.peers_offset+32*uint64_t(num_ranks);c.layers_offset=c.addresses_offset+64*uint64_t(addresses.size());
        c.context_bytes=c.layers_offset+align32(4*uint64_t(layer_ids.size()));
        c.reserved_address=c.context_bytes;
        AFDReturnDirectory directory{c.reserved_address+32,c.reserved_address+32+32*uint64_t(num_src),
            align512(2*uint64_t(hidden)),1,0};
        c.context_bytes=directory.lanes_offset+32*uint64_t(lanes.size());
        const uint64_t bytes=checked_allocation_bytes({c.context_bytes,c.payload_bytes,c.control_bytes});
        EP_HOST_ASSERT(bytes<=memory_budget_bytes);owner->known_allocation_bytes=bytes;
        std::vector<uint8_t> packed(c.context_bytes,0);std::memcpy(packed.data(),&c,sizeof(c));
        std::memcpy(packed.data()+c.lanes_offset,lanes.data(),lanes.size()*sizeof(AFDLane));
        std::memcpy(packed.data()+c.addresses_offset,addresses.data(),addresses.size()*sizeof(AFDAddress));
        std::memcpy(packed.data()+c.layers_offset,layer_ids.data(),layer_ids.size()*sizeof(uint32_t));
        std::memcpy(packed.data()+c.reserved_address,&directory,sizeof(directory));
        std::vector<uint64_t> source_offsets(num_src,0);
        for(uint32_t l=0;l<lanes.size();++l) {
            auto& offset=source_offsets[lanes[l].src_rank_idx];
            const AFDReturnLane reverse{offset,uint64_t(l)*num_ranks*64+32,uint64_t(l)*128+96,0};
            offset=checked_allocation_bytes({offset,uint64_t(lanes[l].capacity)*num_topk*directory.return_stride});
            std::memcpy(packed.data()+directory.lanes_offset+32*uint64_t(l),&reverse,sizeof(reverse));
        }
        owner->owned_context=byte_tensor(c.context_bytes);owner->owned_payload=byte_tensor(c.payload_bytes);owner->owned_control=byte_tensor(c.control_bytes);
        ACL_CHECK(aclrtMemcpy(owner->owned_context.data_ptr(),c.context_bytes,packed.data(),packed.size(),ACL_MEMCPY_HOST_TO_DEVICE));
        zero_tensor(owner->owned_payload);zero_tensor(owner->owned_control);
        return owner;
    }
    // Called once, outside capture. SF may be contiguous [Qalloc,S] or packed columns.
    void prepare_recv(int q,int q_alloc,int expert_alignment,bool sf_column_major=false) {
        using namespace afd;
        EP_HOST_ASSERT(!closed && recv_rank>=0 && !recv_prepared);
        EP_HOST_ASSERT(q>0 && q_alloc>=q && expert_alignment>0 && q_alloc>=expert_alignment && (expert_alignment&(expert_alignment-1))==0);
        const TokenLayout token{config.num_hidden_bytes,uint64_t(config.num_sf_packs)*2,config.num_topk};
        EP_HOST_ASSERT(token.record_stride()<=INT_MAX);
        for(int t:{1,2,4,8,16}) {
            const uint64_t stage=uint64_t(t)*token.record_stride();
            const uint64_t ub=2*align32(stage)+2*align32(16*uint64_t(t))+2*align32(4*uint64_t(t))+64+512+256;
            if(ub<=uint64_t(num_ub_bytes) && (stage<=65536 || t==1))recv_tile=t;
        }
        EP_HOST_ASSERT(recv_tile>0);
        const uint64_t l=config.num_source_lanes,r=config.num_experts/config.num_ranks;
        EP_HOST_ASSERT(l<=INT_MAX && config.num_layers<=INT_MAX && l*r<=UINT32_MAX);
        const uint64_t recv_scratch=dispatch_recv_workspace_bytes(config.num_vec_cores,l,r,q);
        const uint64_t additional=checked_allocation_bytes({active_bytes(l,r),execution_bytes(l),recv_scratch,
            uint64_t(q_alloc)*(config.num_hidden_bytes+2*uint64_t(config.num_sf_packs)+16)});
        EP_HOST_ASSERT(known_allocation_bytes<=budget_bytes && additional<=budget_bytes-known_allocation_bytes);
        EP_HOST_ASSERT(transport_reserve_bytes<=budget_bytes-known_allocation_bytes-additional);
        owned_active=byte_tensor(active_bytes(l,r));owned_execution=byte_tensor(execution_bytes(l));
        owned_recv_workspace=byte_tensor(recv_scratch);
        zero_tensor(owned_active);zero_tensor(owned_execution);zero_tensor(owned_recv_workspace);
        AFDActive header{};header.entries_offset=active_entries_offset();header.members_offset=active_members_offset(l);header.segments_offset=active_segments_offset(l);
        ACL_CHECK(aclrtMemcpy(owned_active.data_ptr(),owned_active.nbytes(),&header,sizeof(header),ACL_MEMCPY_HOST_TO_DEVICE));
        auto options=torch::TensorOptions().device(TORCH_DEVICE_NPU);
        owned_recv_x=torch::empty({q_alloc,hidden},options.dtype(fp8?torch::kFloat8_e4m3fn:torch::kBFloat16));
        owned_recv_metadata=torch::empty({q_alloc,4},options.dtype(torch::kInt32));
        if(fp8)owned_recv_sf=torch::empty_strided({q_alloc,int64_t(config.num_sf_packs)},
            sf_column_major?std::vector<int64_t>{1,q_alloc}:std::vector<int64_t>{int64_t(config.num_sf_packs),1},options.dtype(torch::kInt16));
        recv_q=q;recv_q_alloc=q_alloc;recv_alignment=expert_alignment;recv_prepared=true;known_allocation_bytes+=additional;
    }
    torch::Tensor active()const{return owned_active;}
    torch::Tensor execution_state()const{return owned_execution;}
    torch::Tensor recv_workspace()const{return owned_recv_workspace;}
    torch::Tensor recv_x()const{return owned_recv_x;}
    std::optional<torch::Tensor> recv_sf()const{return owned_recv_sf;}
    torch::Tensor recv_src_metadata()const{return owned_recv_metadata;}
    // New AFD initialization: immutable capacities; only device state advances during replay.
    void prepare_ffn(bool precheck=false) {
        using namespace afd;
        EP_HOST_ASSERT(!closed && recv_prepared && recv_rank>=0 && !ffn_prepared);
        const uint32_t l=config.num_source_lanes,r=config.num_experts/config.num_ranks;
        // Exact array-aligned arenas of the device families, including source-lane maps.
        const uint64_t send_ub=align32(48*uint64_t(config.num_src_ranks))+
            96*uint64_t(config.num_vec_cores)*uint64_t(config.num_src_ranks)+512+
            32*uint64_t(combine_tile)+2*align32(4*uint64_t(l))+256;
        EP_HOST_ASSERT(send_ub<=uint64_t(num_ub_bytes) && 256+5*align32(4*uint64_t(r))<=uint64_t(num_ub_bytes));
        const uint64_t scratch=recv_workspace_bytes(config.num_vec_cores);
        const uint64_t extra=checked_allocation_bytes({scheduler_workspace_bytes(l,r),completion_bytes(l,r),
            8*uint64_t(r),uint64_t(recv_q_alloc)*hidden*2,scratch});
        EP_HOST_ASSERT(known_allocation_bytes<=budget_bytes && extra<=budget_bytes-known_allocation_bytes);
        EP_HOST_ASSERT(transport_reserve_bytes<=budget_bytes-known_allocation_bytes-extra);
        owned_scheduler_workspace=byte_tensor(scheduler_workspace_bytes(l,r));zero_tensor(owned_scheduler_workspace);
        owned_completion=byte_tensor(completion_bytes(l,r));zero_tensor(owned_completion);
        owned_combine_send_workspace=byte_tensor(scratch);zero_tensor(owned_combine_send_workspace);
        auto options=torch::TensorOptions().device(TORCH_DEVICE_NPU);
        owned_psum=torch::zeros({r},options.dtype(torch::kInt32));
        owned_counts=torch::zeros({r},options.dtype(torch::kInt32));
        owned_combine_x=torch::empty({recv_q_alloc,hidden},options.dtype(torch::kBFloat16));
        if(!payload_context) {
            // Local scheduler fixture starts with no publication; populate fixture Ready afterwards.
            std::vector<uint8_t> controls(config.control_bytes,0);
            for(uint32_t lane=0;lane<l;++lane) {const uint32_t one=1;std::memcpy(controls.data()+128*uint64_t(lane)+64,&one,4);}
            ACL_CHECK(aclrtMemcpy(owned_control.data_ptr(),config.control_bytes,controls.data(),controls.size(),ACL_MEMCPY_HOST_TO_DEVICE));
        }
        if(payload_context) {
            // Register the actual GMM2 output, never a second hidden send mirror.
            auto registration=std::make_shared<SourceRegistration>();
            registration->context=payload_context;registration->tensor=owned_combine_x;
            CommMem mem{};mem.type=COMM_MEM_TYPE_DEVICE;mem.addr=owned_combine_x.data_ptr();mem.size=owned_combine_x.nbytes();
            for(const auto endpoint:payload_context->jetty_endpoints) {
                HcommMemHandle handle{};
                HCOMM_CHECK(HcommMemReg(endpoint,"afd_combine_source",&mem,&handle));
                registration->handles.push_back(handle);
            }
            combine_registration=registration;
        }
        scheduler_precheck=precheck;ffn_prepared=true;known_allocation_bytes+=extra;
    }
    // Independent A-side scratch permits concurrent dispatchSend and combineRecv.
    void prepare_combine_recv(bool round_step=false,bool fair_round_robin=false) {
        using namespace afd;
        EP_HOST_ASSERT(!closed && config.src_rank_idx!=UINT32_MAX && !combine_recv_prepared && !source_launched);
        // Explicit framework obligation, not an automatic liveness proof: the caller must
        // sweep every active local slot fairly, including WAIT_ACK, on this one recv stream.
        EP_HOST_ASSERT(round_step==fair_round_robin);
        // Both polling kernels must be able to reside together; stream concurrency
        // and SyncAll resource isolation still require target-runtime validation.
        EP_HOST_ASSERT(2*uint64_t(config.num_vec_cores)<=uint64_t(runtime->get_num_vec_cores()));
        const uint64_t fixed=align32(4*uint64_t(config.num_topk))+align32(8*uint64_t(config.num_topk))+256;
        EP_HOST_ASSERT(uint64_t(num_ub_bytes)>fixed);
        combine_chunk=int(std::min(uint64_t(hidden),(uint64_t(num_ub_bytes)-fixed)/14)/128*128);
        EP_HOST_ASSERT(combine_chunk>=128);
        const uint64_t scratch=combine_recv_workspace_bytes(config.num_vec_cores);
        const uint64_t progress_bytes=round_step?sizeof(AFDReturnProgress)*uint64_t(config.num_slots):0;
        const uint64_t extra=checked_allocation_bytes({scratch,32,progress_bytes});
        EP_HOST_ASSERT(known_allocation_bytes<=budget_bytes && extra<=budget_bytes-known_allocation_bytes);
        EP_HOST_ASSERT(transport_reserve_bytes<=budget_bytes-known_allocation_bytes-extra);
        owned_combine_recv_workspace=byte_tensor(scratch);zero_tensor(owned_combine_recv_workspace);
                // Wiring fix: aclnnInplaceZero rejects DT_UINT32; zero via int32 then view keeps identical bits/ABI.
        owned_recv_status=torch::zeros({8},torch::TensorOptions().device(TORCH_DEVICE_NPU).dtype(torch::kInt32)).view(torch::kUInt32);
        if(round_step) {
            owned_combine_recv_progress=byte_tensor(progress_bytes);zero_tensor(owned_combine_recv_progress);
        }
        // Initialization-time only. The former reserved Context word fixes both send and
        // recv admission; mode-1 contexts must not be passed to an older binary.
        config.return_progress_mode=round_step?RETURN_ROUND_STEP:RETURN_TRANSACTION;
        ACL_CHECK(aclrtMemcpy(static_cast<uint8_t*>(owned_context.data_ptr())+offsetof(AFDContextLayout,return_progress_mode),
            sizeof(config.return_progress_mode),&config.return_progress_mode,sizeof(config.return_progress_mode),ACL_MEMCPY_HOST_TO_DEVICE));
        combine_recv_round_step=round_step;
        known_allocation_bytes+=extra;combine_recv_prepared=true;
    }
    torch::Tensor scheduler_workspace()const{return owned_scheduler_workspace;}
    torch::Tensor completion()const{return owned_completion;}
    torch::Tensor psum_num_recv_tokens_per_expert()const{return owned_psum;}
    torch::Tensor num_unaligned_recv_tokens_per_expert()const{return owned_counts;}
    torch::Tensor combine_x()const{return owned_combine_x;}
    torch::Tensor combine_send_workspace()const{return owned_combine_send_workspace;}
    torch::Tensor combine_recv_workspace()const{return owned_combine_recv_workspace;}
    torch::Tensor recv_status()const{return owned_recv_status;}
    torch::Tensor combine_recv_progress()const{return owned_combine_recv_progress;}
    // Signature follows the scheduler design; buffer/rank and static family belong to this owner.
    void ffnScheduler(const at::Tensor& context,const at::Tensor& control_buffer,at::Tensor& workspace,
        at::Tensor& active,at::Tensor& execution_state,at::Tensor& completion,
        at::Tensor& psum_num_recv_tokens_per_expert,at::Tensor& num_unaligned_recv_tokens_per_expert) {
        EP_HOST_ASSERT(!closed && ffn_prepared);
        check_owned(context,owned_context);check_owned(control_buffer,owned_control);
        check_owned(workspace,owned_scheduler_workspace);check_owned(active,owned_active);
        check_owned(execution_state,owned_execution);check_owned(completion,owned_completion);
        check_owned(psum_num_recv_tokens_per_expert,owned_psum);check_owned(num_unaligned_recv_tokens_per_expert,owned_counts);
        afd::launch_ffn_scheduler(owned_payload.data_ptr(),context.data_ptr(),control_buffer.data_ptr(),workspace.data_ptr(),
            active.data_ptr(),execution_state.data_ptr(),completion.data_ptr(),owned_psum.data_ptr(),owned_counts.data_ptr(),
            config,recv_rank,recv_q,recv_q_alloc,recv_alignment,scheduler_precheck,num_ub_bytes);
    }
    // Enqueue after the device-gated GMM2 on the same stream. Does not execute or validate GMM.
    void computeReady() {
        EP_HOST_ASSERT(!closed && ffn_prepared);
        afd::launch_compute_ready(owned_active.data_ptr(),owned_execution.data_ptr(),num_ub_bytes);
    }
    // ### PATCH START: queue-metadata read-only probe (review candidate)
    void probeQueueMeta(at::Tensor& out) {
        EP_HOST_ASSERT(!closed and payload_context and control_context);
        EP_HOST_ASSERT(out.scalar_type() == at::kInt and out.numel() >= 32);
        afd::launch_queue_meta_probe(payload_context->jetty_device_ptrs,
                                     control_context->jetty_device_ptrs,
                                     out.data_ptr(), num_ub_bytes);
    }
    std::string compileQueueMetaProbeOnly() {
        return afd::compile_queue_meta_probe_only();
    }
    // ### PATCH START: combine compile-only entries (review candidate)
    std::string compileCombineSendOnly() {
        EP_HOST_ASSERT(!closed and payload_context);
        return afd::compile_combine_send_only(config,recv_rank,hidden,recv_q_alloc,
                                              combine_tile,num_ub_bytes,timeout_cycles);
    }
    std::string compileCombineRecvOnly() {
        EP_HOST_ASSERT(!closed);
        return afd::compile_combine_recv_only(config,hidden,combine_chunk,
                                              num_ub_bytes,timeout_cycles,combine_recv_round_step);
    }
    // ### PATCH START: pipeline prewarm compile-only entries
    std::string compileFFNSchedulerOnly() {
        EP_HOST_ASSERT(!closed && ffn_prepared);
        return afd::compile_ffn_scheduler_only(config,recv_q,recv_q_alloc,recv_alignment,scheduler_precheck);
    }
    std::string compileComputeReadyOnly() {
        EP_HOST_ASSERT(!closed && ffn_prepared);
        return afd::compile_compute_ready_only();
    }
    std::string compileDispatchSendOnly() {
        EP_HOST_ASSERT(!closed && config.src_rank_idx!=UINT32_MAX);
        return afd::compile_dispatch_send_only(config,num_ub_bytes,timeout_cycles);
    }
    std::string compileDispatchRecvOnly() {
        EP_HOST_ASSERT(!closed && recv_prepared);
        return afd::compile_dispatch_recv_only(config,recv_q,recv_q_alloc,recv_alignment,recv_tile,num_ub_bytes);
    }
    // ### PATCH END: pipeline prewarm compile-only entries
    // ### PATCH END: combine compile-only entries
    // ### PATCH END: queue-metadata read-only probe
    void combineSend(const at::Tensor& x,const at::Tensor& src_metadata,const at::Tensor& context,
        const at::Tensor& active,at::Tensor& control_buffer,at::Tensor& execution_state,
        at::Tensor& workspace,at::Tensor& completion) {
        EP_HOST_ASSERT(!closed && ffn_prepared && payload_context && combine_registration);
        check_owned(x,owned_combine_x);check_owned(src_metadata,owned_recv_metadata);check_owned(context,owned_context);
        check_owned(active,owned_active);check_owned(control_buffer,owned_control);check_owned(execution_state,owned_execution);
        check_owned(workspace,owned_combine_send_workspace);check_owned(completion,owned_completion);
        afd::launch_combine_send(x.data_ptr(),src_metadata.data_ptr(),context.data_ptr(),active.data_ptr(),control_buffer.data_ptr(),
            execution_state.data_ptr(),workspace.data_ptr(),payload_context->jetty_device_ptrs,completion.data_ptr(),
            config,recv_rank,hidden,recv_q_alloc,combine_tile,num_ub_bytes,timeout_cycles);
    }
    // AFD addition to the weighted epilogue owner: start admits one transaction
    // only in round-step mode. Later start=false calls share scratch but retain
    // each slot's cursor, so an unready transaction cannot block other slots.
    void combineRecv(const at::Tensor& combined_topk_idx,const at::Tensor& topk_weights,
        const std::optional<at::Tensor>& token_mask,const at::Tensor& task_info,const at::Tensor& context,
        at::Tensor& buffer,at::Tensor& control_buffer,at::Tensor& round_ledger,at::Tensor& workspace,
        at::Tensor& combined_x,at::Tensor& recv_status,bool start=false) {
        EP_HOST_ASSERT(!closed && combine_recv_prepared);
        EP_HOST_ASSERT(combine_recv_round_step || !start);
        check_owned(context,owned_context);check_owned(buffer,owned_payload);check_owned(control_buffer,owned_control);
        check_owned(round_ledger,owned_ledger);check_owned(workspace,owned_combine_recv_workspace);check_owned(recv_status,owned_recv_status);
        check_bytes(task_info,sizeof(afd::AFDTaskInfo));EP_HOST_ASSERT(task_info.device()==context.device());
        EP_HOST_ASSERT(combined_topk_idx.device()==context.device() && combined_topk_idx.scalar_type()==torch::kInt64 &&
            combined_topk_idx.is_contiguous() && combined_topk_idx.dim()==2 &&
            combined_topk_idx.size(0)==config.num_token_capacity && combined_topk_idx.size(1)==config.num_topk);
        EP_HOST_ASSERT(topk_weights.device()==context.device() && topk_weights.scalar_type()==torch::kFloat32 &&
            topk_weights.is_contiguous() && topk_weights.sizes()==combined_topk_idx.sizes());
        if(token_mask)EP_HOST_ASSERT(token_mask->device()==context.device() && token_mask->scalar_type()==torch::kBool &&
            token_mask->is_contiguous() && token_mask->dim()==1 && token_mask->size(0)==config.num_token_capacity);
        EP_HOST_ASSERT(combined_x.device()==context.device() && combined_x.scalar_type()==torch::kBFloat16 &&
            combined_x.is_contiguous() && combined_x.dim()==2 && combined_x.size(0)==config.num_token_capacity && combined_x.size(1)==hidden);
        std::vector<const at::Tensor*> regions{&combined_topk_idx,&topk_weights,&task_info,&context,&buffer,
            &control_buffer,&round_ledger,&workspace,&combined_x,&recv_status};
        if(combine_recv_round_step)regions.push_back(&owned_combine_recv_progress);
        if(token_mask)regions.push_back(&*token_mask);
        for(size_t i=0;i<regions.size();++i)for(size_t j=0;j<i;++j) {
            const auto a=reinterpret_cast<uintptr_t>(regions[i]->data_ptr()),b=reinterpret_cast<uintptr_t>(regions[j]->data_ptr());
            const auto an=tensor_span_bytes(*regions[i]),bn=tensor_span_bytes(*regions[j]);
            EP_HOST_ASSERT(an<=UINT64_MAX-a && bn<=UINT64_MAX-b && (a+an<=b || b+bn<=a));
        }
        check_input_alias(combined_x);
        check_input_alias(combined_topk_idx);check_input_alias(topk_weights);check_input_alias(task_info);
        if(token_mask)check_input_alias(*token_mask);
        afd::launch_combine_recv(combined_x.data_ptr(),combined_topk_idx.data_ptr(),topk_weights.data_ptr(),
            token_mask?token_mask->data_ptr():nullptr,buffer.data_ptr(),task_info.data_ptr(),context.data_ptr(),
            control_buffer.data_ptr(),round_ledger.data_ptr(),workspace.data_ptr(),recv_status.data_ptr(),
            config,hidden,combine_chunk,num_ub_bytes,timeout_cycles,
            combine_recv_round_step?owned_combine_recv_progress.data_ptr():nullptr,combine_recv_round_step,start);
    }
    // Public signature matches dispatchRecv/design.md. The owner supplies local Inbox and F rank.
    void dispatchRecv(const at::Tensor& buffer,const at::Tensor& context,const at::Tensor& active,
        at::Tensor& execution_state,at::Tensor& workspace,at::Tensor& recv_x,
        std::optional<at::Tensor>& recv_sf,at::Tensor& recv_src_metadata) {
        EP_HOST_ASSERT(!closed && recv_prepared && recv_rank>=0);
        check_owned(buffer,owned_payload);check_owned(context,owned_context);check_owned(active,owned_active);
        check_owned(execution_state,owned_execution);check_owned(workspace,owned_recv_workspace);
        check_bytes(buffer,config.payload_bytes);check_bytes(context,config.context_bytes);
        check_bytes(active,afd::active_bytes(config.num_source_lanes,config.num_experts/config.num_ranks));
        check_bytes(execution_state,afd::execution_bytes(config.num_source_lanes));
        check_bytes(workspace,afd::dispatch_recv_workspace_bytes(config.num_vec_cores,
            config.num_source_lanes,config.num_experts/config.num_ranks,recv_q));
        check_bytes(owned_control,config.control_bytes);
        // Outputs may be independently preallocated, but must match the static family and not overlap.
        EP_HOST_ASSERT(recv_x.device()==context.device() && recv_x.is_contiguous() && recv_x.dim()==2 &&
            recv_x.size(0)==recv_q_alloc && recv_x.size(1)==hidden && recv_x.scalar_type()==(fp8?torch::kFloat8_e4m3fn:torch::kBFloat16));
        EP_HOST_ASSERT(recv_src_metadata.device()==context.device() && recv_src_metadata.is_contiguous() &&
            recv_src_metadata.scalar_type()==torch::kInt32 && recv_src_metadata.dim()==2 &&
            recv_src_metadata.size(0)==recv_q_alloc && recv_src_metadata.size(1)==4);
        int s0=0,s1=0;
        if(fp8) {
            EP_HOST_ASSERT(recv_sf && recv_sf->device()==context.device() && recv_sf->scalar_type()==torch::kInt16 &&
                recv_sf->dim()==2 && recv_sf->size(0)==recv_q_alloc && recv_sf->size(1)==config.num_sf_packs);
            const auto a=recv_sf->stride(0),b=recv_sf->stride(1);
            EP_HOST_ASSERT(a>0 && b>0 && a<=INT_MAX && b<=INT_MAX);
            EP_HOST_ASSERT((b==1 && (recv_q_alloc==1 || a>=config.num_sf_packs)) ||
                (a==1 && (config.num_sf_packs==1 || b>=recv_q_alloc)));
            s0=int(a);s1=int(b);
        } else EP_HOST_ASSERT(!recv_sf);
        std::vector<const at::Tensor*> regions{&buffer,&context,&active,&execution_state,&workspace,&owned_control,&recv_x,&recv_src_metadata};
        if(recv_sf)regions.push_back(&*recv_sf);
        // Also protect owner allocations used by dispatchSend, when present.
        for(const auto* value:{&owned_ledger,&owned_workspace,&owned_status,&owned_scheduler_workspace,&owned_completion,
            &owned_psum,&owned_counts,&owned_combine_x,&owned_combine_send_workspace})if(value->defined())regions.push_back(value);
        for(size_t i=0;i<regions.size();++i) {
            const uint64_t a=reinterpret_cast<uintptr_t>(regions[i]->data_ptr()),an=tensor_span_bytes(*regions[i]);
            EP_HOST_ASSERT(an<=UINT64_MAX-a);
            for(size_t j=0;j<i;++j) {
                const uint64_t b=reinterpret_cast<uintptr_t>(regions[j]->data_ptr()),bn=tensor_span_bytes(*regions[j]);
                EP_HOST_ASSERT(bn<=UINT64_MAX-b && (an==0 || bn==0 || a+an<=b || b+bn<=a));
            }
        }
        afd::launch_dispatch_recv(buffer.data_ptr(),context.data_ptr(),active.data_ptr(),execution_state.data_ptr(),
            workspace.data_ptr(),recv_x.data_ptr(),recv_sf?recv_sf->data_ptr():nullptr,recv_src_metadata.data_ptr(),owned_control.data_ptr(),
            config,recv_rank,recv_q,recv_q_alloc,recv_alignment,recv_tile,num_ub_bytes,s0,s1);
    }
    // ### PATCH END: receiver initialization and out-style public operation
    // New AFD method: original Tensor order is unchanged, JIT/resource configuration belongs to this owner.
    void dispatchSend(const at::Tensor& x,const std::optional<at::Tensor>& sf,
        const at::Tensor& topk_idx,const at::Tensor& topk_weights,const std::optional<at::Tensor>& token_mask,
        const at::Tensor& task_info,const at::Tensor& context,at::Tensor& control_buffer,
        at::Tensor& round_ledger,at::Tensor& workspace,at::Tensor& send_status) {
        EP_HOST_ASSERT(!closed && config.src_rank_idx!=UINT32_MAX);
        check_owned(context,owned_context);check_owned(control_buffer,owned_control);
        check_owned(round_ledger,owned_ledger);check_owned(workspace,owned_workspace);check_owned(send_status,owned_status);
        EP_HOST_ASSERT(x.device()==context.device() && x.is_contiguous() && x.dim()==2);
        EP_HOST_ASSERT(x.size(0)==config.num_token_capacity && x.size(1)==hidden);
        EP_HOST_ASSERT(x.scalar_type()==(fp8?torch::kFloat8_e4m3fn:torch::kBFloat16));
        EP_HOST_ASSERT(topk_idx.device()==x.device() && topk_idx.scalar_type()==torch::kInt64 && topk_idx.is_contiguous());
        EP_HOST_ASSERT(topk_idx.dim()==2 && topk_idx.size(0)==x.size(0) && topk_idx.size(1)==config.num_topk);
        EP_HOST_ASSERT(topk_weights.device()==x.device() && topk_weights.scalar_type()==torch::kFloat32 && topk_weights.is_contiguous());
        EP_HOST_ASSERT(topk_weights.sizes()==topk_idx.sizes());
        check_bytes(task_info,sizeof(afd::AFDTaskInfo));EP_HOST_ASSERT(task_info.device()==x.device());
        if(token_mask)EP_HOST_ASSERT(token_mask->device()==x.device() && token_mask->scalar_type()==torch::kBool &&
            token_mask->dim()==1 && token_mask->size(0)==x.size(0) && token_mask->is_contiguous());
        int s0=0,s1=0;
        if(fp8) {
            EP_HOST_ASSERT(sf.has_value() && sf->device()==x.device() && sf->scalar_type()==torch::kInt16);
            EP_HOST_ASSERT(sf->dim()==2 && sf->size(0)==x.size(0) && sf->size(1)==config.num_sf_packs);
            EP_HOST_ASSERT(sf->stride(0)>0 && sf->stride(1)>0 && (sf->stride(0)==1 || sf->stride(1)==1));
            EP_HOST_ASSERT(sf->stride(0)<=INT_MAX && sf->stride(1)<=INT_MAX);
            s0=int(sf->stride(0));s1=int(sf->stride(1));
        } else EP_HOST_ASSERT(!sf.has_value());
        check_input_alias(x);check_input_alias(topk_idx);check_input_alias(topk_weights);check_input_alias(task_info);
        if(sf)check_input_alias(*sf);
        if(token_mask)check_input_alias(*token_mask);
        source_launched=true; // Context progress mode is immutable once any source launch starts.
        afd::launch_dispatch_send(x.data_ptr(),sf?sf->data_ptr():nullptr,topk_idx.data_ptr(),topk_weights.data_ptr(),
            token_mask?token_mask->data_ptr():nullptr,task_info.data_ptr(),context.data_ptr(),control_buffer.data_ptr(),
            round_ledger.data_ptr(),workspace.data_ptr(),payload_context->jetty_device_ptrs,send_status.data_ptr(),
            config,num_ub_bytes,timeout_cycles,s0,s1);
    }
};
} // namespace deep_ep
// ### PATCH END: AFD owner
