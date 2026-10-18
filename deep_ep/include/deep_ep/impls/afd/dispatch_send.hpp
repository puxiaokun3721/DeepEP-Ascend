#pragma once
#include <climits>
#include <kernel_operator.h>
#include <simt_api/device_atomic_functions.h>
#include <simt_api/device_sync_functions.h>
#include <simt_api/device_warp_functions.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>
#include <deep_ep/comm/handle.hpp>
#include <deep_ep/common/simt.hpp>
#include <deep_ep/layout/afd/token.hpp>

// ### PATCH START: bounded AFD sender, adapted from impls/ep/dispatch.hpp at 3b25377d
namespace deep_ep::afd {
constexpr uint32_t kNumMaxPendingWrites=32;
template<int E,int N,int K,int T,int M>
struct alignas(32) DispatchSendUB {
    alignas(32) int expert_histogram[E], expert_cursor[E];
    alignas(32) int rank_histogram[N], rank_psum[N];
    alignas(32) int dst_rank_idx[T*K], dst_slot_idx[T*K];
    alignas(32) handle::HcommPeerInfo peer_info[N];
    alignas(32) uint8_t jetty_info[96];
    alignas(32) uint8_t ub_sqe[512];
    alignas(32) uint8_t control_stage[256];
    alignas(32) uint8_t metadata[2][4*M];
};

// Original warp grouping, deduplicate and atomic histogram logic; no peer-rank self-copy.
// Added fixed tiles, full int64 validation, optional masks, and a second route-emission pass.
template<int E,int N,int K,int T,int M,bool Emit>
__simt_vf__ __aicore__ __launch_bounds__(512) void route_tile(
    __ubuf__ DispatchSendUB<E,N,K,T,M>* ub,
    __gm__ int64_t* topk_idx, __gm__ uint8_t* token_mask,
    __gm__ uint32_t* counts, __gm__ uint32_t* tile_psum,
    __gm__ uint32_t* expert_prefix, __gm__ int32_t* dst_slots,
    __gm__ uint32_t* route_reference, __gm__ uint32_t* error,
    uint32_t token_begin, uint32_t tile_begin, uint32_t num_tokens) {
    const int tid=threadIdx.x, lane=tid%32;
    constexpr int G=K==1 ? 1 : 1<<(32-__builtin_clz(unsigned(K-1)));
    constexpr int groups=512/G;
    const int g=tid/G, gl=tid%G;
    const uint32_t mask=(0xffffffffu>>(32-G))<<(lane/G*G);
    if(tid<E) ub->expert_histogram[tid]=0;
    if(tid<N) ub->rank_histogram[tid]=0;
    asc_syncthreads();
    for(int i=g;i<((num_tokens+groups-1)/groups)*groups;i+=groups) {
        const bool live=i<num_tokens && (token_mask==nullptr || token_mask[uint64_t(token_begin)+tile_begin+i]);
        int expert=-1;
        if(live && gl<K) {
            const int64_t value=topk_idx[(uint64_t(token_begin)+tile_begin+i)*K+gl];
            if(value<0 || value>=E) asc_atomic_max(error,uint32_t(BAD_ROUTE));
            else expert=int(value);
        }
        // Check distinct experts without truncating invalid int64 values into legal IDs.
        for(int k=0;k<K;++k) {
            const int other=asc_shfl(expert,lane/G*G+k);
            if(live && gl<K && k<gl && expert>=0 && other==expert)
                asc_atomic_max(error,uint32_t(BAD_ROUTE));
        }
        const int rank=expert<0 ? -1 : expert/(E/N);
        const bool representative=simt::deduplicate<N>(rank,lane,mask);
        int slot=-1;
        if(rank>=0 && representative) slot=asc_atomic_add(ub->rank_histogram+rank,1);
        const int ordinal=expert>=0 ? asc_atomic_add(ub->expert_histogram+expert,1) : -1;
        if constexpr(Emit) {
            if(slot>=0) slot+=asc_ldcg(tile_psum+rank);
            int record=-1;
            for(int k=0;k<K;++k) {
                const int peer=asc_shfl(representative?rank:-1,lane/G*G+k);
                const int value=asc_shfl(slot,lane/G*G+k);
                if(rank>=0 && peer==rank) record=value;
            }
            if(i<num_tokens && gl<K) {
                ub->dst_rank_idx[i*K+gl]=representative?rank:-1;
                ub->dst_slot_idx[i*K+gl]=record;
                asc_stcg(dst_slots+uint64_t(tile_begin+i)*K+gl,record);
                if(expert>=0) {
                    const uint64_t row=uint64_t(asc_ldcg(expert_prefix+expert))+asc_ldcg(tile_psum+N+expert)+ordinal;
                    asc_stcg(route_reference+row,uint32_t(record*K+gl));
                }
            }
        }
    }
    asc_syncthreads();
    if constexpr(!Emit) {
        if(tid<N) asc_stcg(counts+tid,uint32_t(ub->rank_histogram[tid]));
        if(tid<E) asc_stcg(counts+N+tid,uint32_t(ub->expert_histogram[tid]));
    }
    asc_threadfence();
}

__aicore__ inline void local_join() {
    asc_sync_notify(PIPE_MTE3,PIPE_V,EVENT_ID0);
    asc_sync_wait(PIPE_MTE3,PIPE_V,EVENT_ID0);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    AscendC::SyncAll<true>();
}
__aicore__ inline void load_control(__ubuf__ uint8_t* ub,__gm__ void* src) {
    asc_copy_gm2ub_align(ub,static_cast<__gm__ uint8_t*>(src),1,32,0,0,false,
        static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV),uint64_t(32),32);
    asc_sync_notify(PIPE_MTE2,PIPE_S,EVENT_ID1);
    asc_sync_wait(PIPE_MTE2,PIPE_S,EVENT_ID1);
}
__aicore__ inline void store_control(__gm__ void* dst,__ubuf__ uint8_t* ub,uint32_t bytes=32) {
    asc_sync_notify(PIPE_S,PIPE_MTE3,EVENT_ID1);
    asc_sync_wait(PIPE_S,PIPE_MTE3,EVENT_ID1);
    asc_copy_ub2gm_align(static_cast<__gm__ uint8_t*>(dst),ub,1,bytes,
        asc_store_l2_cache_mode::NOTALLOC_CLEAN,uint64_t(bytes),bytes);
    asc_sync_notify(PIPE_MTE3,PIPE_S,EVENT_ID1);
    asc_sync_wait(PIPE_MTE3,PIPE_S,EVENT_ID1);
}
template<int64_t Timeout>
__aicore__ inline bool wait_control(__ubuf__ uint8_t* ub,__gm__ void* src,uint32_t wanted) {
    const uint64_t start=asc_get_system_cycle();
    do {
        load_control(ub,src);
        if(*reinterpret_cast<__ubuf__ uint32_t*>(ub)==wanted) return true;
    } while(uint64_t(asc_get_system_cycle())-start<uint64_t(Timeout));
    return false;
}
// Integer-only offset token: build_write_sqe never dereferences the synthetic destination.
// Reuses original bounds checks, strong ordering/fence/CQE, SQ wrap and completion accounting.
// Commit a bounded group before drain: drain must never wait for unpublished SQEs.
template<int64_t Timeout>
__aicore__ inline void flush_writes(handle::HcommJetty& jetty,uint32_t& inflight) {
    if(inflight==0) return;
    jetty.ring_doorbell();
    jetty.template drain<Timeout>();
    inflight=0;
}
template<int64_t Timeout>
__aicore__ inline void put_at_offset(handle::HcommJetty& jetty,const handle::HcommPeer& peer,
    uint64_t offset,__gm__ void* src,uint64_t bytes,uint32_t& inflight) {
    EP_DEVICE_ASSERT(bytes>0 && bytes<=UINT32_MAX);
    EP_DEVICE_ASSERT(offset<=peer.peer_info->rmt_buffer_size && bytes<=peer.peer_info->rmt_buffer_size-offset);
    jetty.template put<Timeout,false>(peer,nullptr,reinterpret_cast<__gm__ void*>(offset),src,bytes);
    if(++inflight==kNumMaxPendingWrites) flush_writes<Timeout>(jetty,inflight);
}

// Native dispatch uses an SGE pair (hidden, metadata) for each token. Reuse
// HcommJetty's header encoder and wrap-aware SQ copy, adding that second SGE.
// Keep one record per SQE: AFD's align512 record gaps cannot be concatenated
// like native dispatch's packed six-pair SQEs without extra padding SGEs.
template<int64_t Timeout>
__aicore__ inline void put_record_at_offset(handle::HcommJetty& jetty,const handle::HcommPeer& peer,
    uint64_t offset,__gm__ void* x,__gm__ void* metadata_send_buffer,
    uint32_t num_hidden_bytes,uint32_t num_metadata_bytes,uint32_t& inflight) {
    constexpr uint32_t num_sges=2;
    constexpr uint32_t num_wqebbs=(sizeof(AscendC::HcommUrmaSqeCtx)+
        num_sges*sizeof(AscendC::HcommUrmaSgeCtx)+handle::kNumWQEBBBytes-1)/handle::kNumWQEBBBytes;
    static_assert(num_wqebbs*handle::kNumWQEBBBytes<=512);
    static_assert(kNumMaxPendingWrites*num_wqebbs<handle::kSQDepth);
    EP_DEVICE_ASSERT(offset<=peer.peer_info->rmt_buffer_size &&
        uint64_t(num_hidden_bytes)+num_metadata_bytes<=peer.peer_info->rmt_buffer_size-offset);
    const auto current_head=static_cast<uint32_t>(jetty.jetty_info->packed_head);
    jetty.template poll_cq_when_sq_overflow<Timeout>();
    jetty.template build_write_sqe<false>(peer,nullptr,reinterpret_cast<__gm__ void*>(offset),
        x,num_hidden_bytes,current_head);
    auto sqe=reinterpret_cast<__ubuf__ AscendC::HcommUrmaSqeCtx*>(jetty.ub_sqe);
    sqe->sgeNum=num_sges;
    auto sge=reinterpret_cast<__ubuf__ AscendC::HcommUrmaSgeCtx*>(
        jetty.ub_sqe+sizeof(AscendC::HcommUrmaSqeCtx))+1;
    sge->len=num_metadata_bytes;
    sge->tokenId=0;
    sge->va=reinterpret_cast<uint64_t>(metadata_send_buffer);
    // copy_sqe_to_sq transfers complete WQEBBs, including the tail after SGEs.
    constexpr uint32_t num_sqe_bytes=sizeof(AscendC::HcommUrmaSqeCtx)+
        num_sges*sizeof(AscendC::HcommUrmaSgeCtx);
    static_assert(num_sqe_bytes%sizeof(uint64_t)==0);
    for(uint32_t i=num_sqe_bytes/sizeof(uint64_t);
        i<num_wqebbs*handle::kNumWQEBBBytes/sizeof(uint64_t);++i)
        reinterpret_cast<__ubuf__ uint64_t*>(jetty.ub_sqe)[i]=0;
    jetty.copy_sqe_to_sq(current_head,num_wqebbs);
    jetty.advance_sq(num_wqebbs);
    if(++inflight==kNumMaxPendingWrites) flush_writes<Timeout>(jetty,inflight);
}

template<int N,int E,int K,int NC,int X,int S,int C,int T,int U,int64_t Timeout>
__global__ __vector__ void dispatch_send_impl(
    __gm__ void* x,__gm__ sf_pack_t* sf,__gm__ int64_t* topk_idx,__gm__ float* topk_weights,
    __gm__ uint8_t* token_mask,__gm__ const AFDTaskInfo* task_info,
    __gm__ const AFDContextLayout* context,__gm__ void* control_buffer,
    __gm__ AFDRoundLedger* round_ledger,__gm__ void* workspace,__gm__ void* jetty_ptrs,
    __gm__ uint32_t* send_status,const int sf_token_stride,const int sf_hidden_stride) {
    AscendC::InitSocState();
    constexpr TokenLayout token_layout{X,S*2,K};
    constexpr int M=token_layout.metadata_stride();
    using UB=DispatchSendUB<E,N,K,T,M>;
    static_assert(sizeof(UB)<=U && handle::HcommJetty::kNumMaxSQEBytes<=512);
    const int vec_core_idx=AscendC::GetBlockIdx();
    const auto ub=reinterpret_cast<__ubuf__ UB*>(0);
    const auto stage=ub->control_stage;
    const auto words=reinterpret_cast<__ubuf__ uint32_t*>(stage);
    const auto base=reinterpret_cast<__gm__ uint8_t*>(context);
    const auto lanes=reinterpret_cast<__gm__ const AFDLane*>(base+context->lanes_offset);
    const auto peers=reinterpret_cast<__gm__ const AFDPeer*>(base+context->peers_offset);
    const auto addresses=reinterpret_cast<__gm__ const AFDAddress*>(base+context->addresses_offset);
    const auto layers=reinterpret_cast<__gm__ const uint32_t*>(base+context->layers_offset);
    const auto ws=static_cast<__gm__ uint8_t*>(workspace);
    const auto controls=static_cast<__gm__ uint8_t*>(control_buffer);
    const auto w=context->workspace;
    const auto error=reinterpret_cast<__gm__ uint32_t*>(ws+w.error_scratch);
    const auto counts=reinterpret_cast<__gm__ uint32_t*>(ws+w.tile_counts);
    const auto psum=reinterpret_cast<__gm__ uint32_t*>(ws+w.tile_psum);
    const auto ecounts=reinterpret_cast<__gm__ uint32_t*>(ws+w.expert_counts);
    const auto eprefix=reinterpret_cast<__gm__ uint32_t*>(ws+w.expert_prefix);
    const auto rtotals=reinterpret_cast<__gm__ uint32_t*>(ws+w.rank_totals);
    const auto dst_slots=reinterpret_cast<__gm__ int32_t*>(ws+w.dst_buffer_slot_idx);
    const auto refs=reinterpret_cast<__gm__ uint32_t*>(ws+w.route_reference_send);
    uint32_t lane_idx=context->num_source_lanes;
    for(uint32_t l=0;l<context->num_source_lanes;++l)
        if(lanes[l].src_rank_idx==task_info->src_rank_idx && lanes[l].payload_slot_idx==task_info->payload_slot_idx) lane_idx=l;
    if(vec_core_idx==0) {
        uint32_t code=OK;
        if(context->abi_version!=kAbiVersion || task_info->abi_version!=kAbiVersion) code=BAD_ABI;
        else if(lane_idx==context->num_source_lanes || task_info->src_rank_idx!=context->src_rank_idx ||
            task_info->num_tokens>context->num_token_capacity || task_info->num_comm_rounds==0) code=BAD_TASK;
        else {
            const uint32_t c=lanes[lane_idx].capacity;
            if(task_info->num_comm_rounds<(uint64_t(task_info->num_tokens)+c-1)/c) code=BAD_TASK;
            bool valid_layer=false;
            for(uint32_t i=0;i<context->num_layers;++i) valid_layer|=layers[i]==task_info->layer_idx;
            if(!valid_layer) code=BAD_LAYER;
        }
        AscendC::WriteGmByPassDCache(error,code);
        for(int i=0;i<8;++i) words[i]=0;
        words[0]=1;store_control(send_status,stage);
    }
    local_join();
    if(AscendC::ReadGmByPassDCache(error)!=OK) {
        if(vec_core_idx==0) { for(int i=0;i<8;++i)words[i]=0;words[0]=3;words[1]=AscendC::ReadGmByPassDCache(error);store_control(send_status,stage); }
        return; // all launched AIVs take this branch after the same join
    }
    const auto lane=lanes[lane_idx];
    const auto state=reinterpret_cast<__gm__ AFDSendState*>(ws+w.send_state)+lane.payload_slot_idx;
    handle::HcommJetty jetty(reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(ub->jetty_info),ub->ub_sqe,jetty_ptrs,vec_core_idx);
    // Original Hcomm SQ encoder assumes kSQDepth=32768 WQEBBs. Host validates that contract.
    // Native poll_cq_when_sq_overflow starts reclaim 10 completions before wrap;
    // keep an unpublished batch strictly below that threshold (minimum CQ=64).
    constexpr uint32_t kNumPollCQThreshold=10;
    EP_DEVICE_ASSERT(jetty.jetty_info->cq_depth>kNumMaxPendingWrites+kNumPollCQThreshold &&
        (jetty.jetty_info->cq_depth & (jetty.jetty_info->cq_depth-1))==0 &&
        jetty.jetty_info->num_wqebb_bytes==handle::kNumWQEBBBytes);
    // Like native dispatch's peer_info preload, decode each peer once per AIV.
    // HcommPeer is a view of peer_info; rebinding the view below does not reload GM.
    handle::HcommPeer peer(ub->peer_info,jetty_ptrs,peers[0].peer_idx);
    for(int f=1;f<N;++f) {
        handle::HcommPeer preload_peer(ub->peer_info+f,jetty_ptrs,peers[f].peer_idx);
    }
    uint32_t current_round=0;
    for(uint32_t round=0;round<task_info->num_comm_rounds;++round) {
        current_round=round;
        const auto ledger=round_ledger+uint64_t(lane.payload_slot_idx)*2+(round%2);
        if(vec_core_idx==0) {
            load_control(stage,state);const auto previous=*reinterpret_cast<__ubuf__ AFDSendState*>(stage);
            bool ready=true;
            if(previous.has_published) for(int f=0;f<N;++f)
                ready=wait_control<Timeout>(stage,controls+addresses[lane_idx*N+f].consumed_offset,previous.last_input_bit)&&ready;
            ready=wait_control<Timeout>(stage,ledger,0)&&ready;
            if(!ready) AscendC::WriteGmByPassDCache(error,uint32_t(TIMEOUT));
        }
        local_join();
        if(AscendC::ReadGmByPassDCache(error)!=OK) break;
        const uint32_t begin=uint32_t(math::min(uint64_t(task_info->num_tokens),uint64_t(round)*lane.capacity));
        const uint32_t length=math::min(lane.capacity,task_info->num_tokens-begin);
        const uint32_t tiles=(length+T-1)/T;
        for(uint32_t tile=vec_core_idx;tile<tiles;tile+=NC) {
            AscendC::Simt::VF_CALL<route_tile<E,N,K,T,M,false>>(AscendC::Simt::Dim3(512,1,1),
                ub,topk_idx,token_mask,counts+tile*(N+E),psum+tile*(N+E),eprefix,dst_slots,refs,error,
                begin,tile*T,math::min(uint32_t(T),length-tile*T));
            asc_sync_notify(PIPE_V,PIPE_S,EVENT_ID0);asc_sync_wait(PIPE_V,PIPE_S,EVENT_ID0);
        }
        local_join();
        if(AscendC::ReadGmByPassDCache(error)!=OK) break;
        // Independent prefix columns need no extra workspace. Only the final
        // expert-major exclusive prefix is ordered after all column totals exist.
        for(int column=vec_core_idx;column<N+E;column+=NC) {
            uint32_t total=0;
            for(uint32_t tile=0;tile<tiles;++tile) {
                AscendC::WriteGmByPassDCache(psum+tile*(N+E)+column,total);
                total+=AscendC::ReadGmByPassDCache(counts+tile*(N+E)+column);
            }
            if(column<N) AscendC::WriteGmByPassDCache(rtotals+column,total);
            else AscendC::WriteGmByPassDCache(ecounts+column-N,total);
        }
        local_join();
        if(vec_core_idx==0) {
            uint32_t expert_base=0;
            for(int expert=0;expert<E;++expert) {
                AscendC::WriteGmByPassDCache(eprefix+expert,expert_base);
                expert_base+=AscendC::ReadGmByPassDCache(ecounts+expert);
            }
        }
        local_join();
        uint32_t inflight=0;
        for(uint32_t tile=vec_core_idx;tile<tiles;tile+=NC) {
            const uint32_t size=math::min(uint32_t(T),length-tile*T);
            AscendC::Simt::VF_CALL<route_tile<E,N,K,T,M,true>>(AscendC::Simt::Dim3(512,1,1),
                ub,topk_idx,token_mask,counts+tile*(N+E),psum+tile*(N+E),eprefix,dst_slots,refs,error,
                begin,tile*T,size);
            asc_sync_notify(PIPE_V,PIPE_S,EVENT_ID0);asc_sync_wait(PIPE_V,PIPE_S,EVENT_ID0);
            // Native preload_metadata / batch_idx / stage_idx progression, restricted
            // to this fixed routing tile. Metadata remains in existing GM scratch
            // until all WRITEs drain; no hidden mirror or additional workspace.
            constexpr int kNumTokensPerMetadataMTE=4;
            constexpr int kNumMetadataMTEStages=2;
            const int num_metadata_batches=(size+kNumTokensPerMetadataMTE-1)/kNumTokensPerMetadataMTE;
            const int num_active_stages=math::min(num_metadata_batches,kNumMetadataMTEStages);
            for(int stage_idx=0;stage_idx<num_active_stages;++stage_idx) {
                const auto ub_metadata=ub->metadata[stage_idx];
                for(int b=0;b<kNumTokensPerMetadataMTE*M;++b) ub_metadata[b]=0;
                const auto event=static_cast<event_t>(stage_idx+2);
                asc_sync_notify(PIPE_S,PIPE_MTE2,event);
                asc_sync_wait(PIPE_S,PIPE_MTE2,event);
                asc_sync_notify(PIPE_MTE3,PIPE_MTE2,event);
            }
            auto preload_metadata = [&](const int token_idx,const int stage_idx) __aicore__ {
                const auto ub_metadata=ub->metadata[stage_idx];
                const auto event=static_cast<event_t>(stage_idx+2);
                const int num_tokens_in_stage=math::min(kNumTokensPerMetadataMTE,int(size)-token_idx);
                const uint64_t global=uint64_t(begin)+tile*T+token_idx;
                asc_sync_wait(PIPE_MTE3,PIPE_MTE2,event);
                if constexpr(S>0) {
                    const auto gm_sf=sf+global*sf_token_stride;
                    if(sf_hidden_stride==1) {
                        asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ sf_pack_t*>(ub_metadata),gm_sf,
                            static_cast<uint16_t>(num_tokens_in_stage),S*sizeof(sf_pack_t),0,0,false,
                            static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                            uint64_t(sf_token_stride)*sizeof(sf_pack_t),M);
                    } else {
                        const nddma_desc sf_desc(
                            nddma_desc::loop_desc(num_tokens_in_stage,M/sizeof(sf_pack_t),uint32_t(sf_token_stride)),
                            nddma_desc::loop_desc(S,1,uint32_t(sf_hidden_stride)));
                        nddma_out_to_ub(reinterpret_cast<__ubuf__ sf_pack_t*>(ub_metadata),gm_sf,
                            0,sf_desc,0,CONSTANT_PADDING,
                            static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV));
                    }
                }
                asc_copy_gm2ub_align(ub_metadata+token_layout.topk_idx_offset()-X,
                    reinterpret_cast<__gm__ uint8_t*>(topk_idx+global*K),
                    static_cast<uint16_t>(num_tokens_in_stage),K*8,0,0,false,
                    static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV),uint64_t(K*8),M);
                asc_copy_gm2ub_align(ub_metadata+token_layout.topk_weights_offset()-X,
                    reinterpret_cast<__gm__ uint8_t*>(topk_weights+global*K),
                    static_cast<uint16_t>(num_tokens_in_stage),K*4,0,0,false,
                    static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV),uint64_t(K*4),M);
                asc_sync_notify(PIPE_MTE2,PIPE_S,event);
            };
            for(int stage_idx=0;stage_idx<num_active_stages;++stage_idx)
                preload_metadata(stage_idx*kNumTokensPerMetadataMTE,stage_idx);
            for(int batch_idx=0;batch_idx<num_metadata_batches;++batch_idx) {
                const int stage_idx=batch_idx%kNumMetadataMTEStages;
                const int first_metadata_idx=batch_idx*kNumTokensPerMetadataMTE;
                const int num_tokens_in_stage=math::min(kNumTokensPerMetadataMTE,int(size)-first_metadata_idx);
                const auto event=static_cast<event_t>(stage_idx+2);
                const auto ub_metadata=ub->metadata[stage_idx];
                asc_sync_wait(PIPE_MTE2,PIPE_S,event);
                for(int i=0;i<num_tokens_in_stage;++i) {
                    const auto meta=ub_metadata+i*M;
                    *reinterpret_cast<__ubuf__ uint32_t*>(meta+token_layout.src_token_idx_offset()-X)=tile*T+first_metadata_idx+i;
                    *reinterpret_cast<__ubuf__ uint32_t*>(meta+token_layout.valid_topk_mask_offset()-X)=
                        K==32?UINT32_MAX:(uint32_t(1)<<K)-1;
                }
                asc_sync_notify(PIPE_S,PIPE_MTE3,event);
                asc_sync_wait(PIPE_S,PIPE_MTE3,event);
                // Unlike gathered fields, completed metadata records are contiguous
                // in both UB and GM; one burst avoids any inter-burst stride/gap.
                const uint32_t num_metadata_bytes=num_tokens_in_stage*M;
                asc_copy_ub2gm_align(ws+w.metadata_send_buffer+uint64_t(tile*T+first_metadata_idx)*M,
                    ub_metadata,1,num_metadata_bytes,
                    asc_store_l2_cache_mode::NOTALLOC_CLEAN,uint64_t(num_metadata_bytes),num_metadata_bytes);
                asc_sync_notify(PIPE_MTE3,PIPE_MTE2,event);
                const int next_batch_idx=batch_idx+kNumMetadataMTEStages;
                if(next_batch_idx<num_metadata_batches)
                    preload_metadata(next_batch_idx*kNumTokensPerMetadataMTE,stage_idx);
            }
            // Last release for each stage is consumed exactly once, including tails.
            for(int stage_idx=0;stage_idx<num_active_stages;++stage_idx)
                asc_sync_wait(PIPE_MTE3,PIPE_MTE2,static_cast<event_t>(stage_idx+2));
            asc_sync_notify(PIPE_MTE2,PIPE_S,EVENT_ID0);
            asc_sync_wait(PIPE_MTE2,PIPE_S,EVENT_ID0);
            for(uint32_t i=0;i<size;++i) {
                const uint32_t row=tile*T+i,global=begin+row;
                const auto gm_meta=ws+w.metadata_send_buffer+uint64_t(row)*M;
                for(int k=0;k<K;++k) {
                    const int f=ub->dst_rank_idx[i*K+k];if(f<0)continue;
                    peer.peer_info=ub->peer_info+f;
                    const uint64_t off=addresses[lane_idx*N+f].records_offset+uint64_t(ub->dst_slot_idx[i*K+k])*token_layout.record_stride();
                    put_record_at_offset<Timeout>(jetty,peer,off,
                        static_cast<__gm__ uint8_t*>(x)+uint64_t(global)*X,gm_meta,X,M,inflight);
                }
            }
            flush_writes<Timeout>(jetty,inflight);
        }
        local_join();
        if(AscendC::ReadGmByPassDCache(error)!=OK) break;
        if(vec_core_idx==0) {
            uint64_t route_base=0;
            for(int f=0;f<N;++f) {
                uint32_t routes=0;
                for(int e=0;e<E/N;++e) {
                    routes+=AscendC::ReadGmByPassDCache(ecounts+f*(E/N)+e);
                    AscendC::WriteGmByPassDCache(eprefix+f*(E/N)+e,routes);
                }
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                peer.peer_info=ub->peer_info+f;
                const auto address=addresses[lane_idx*N+f];
                if(routes)put_at_offset<Timeout>(jetty,peer,address.route_reference_offset,refs+route_base,uint64_t(routes)*4,inflight);
                put_at_offset<Timeout>(jetty,peer,address.count_offset,ecounts+f*(E/N),(E/N)*4,inflight);
                put_at_offset<Timeout>(jetty,peer,address.prefix_offset,eprefix+f*(E/N),(E/N)*4,inflight);
                route_base+=routes;
            }
            flush_writes<Timeout>(jetty,inflight);
            load_control(stage,state);const AFDSendState saved=*reinterpret_cast<__ubuf__ AFDSendState*>(stage);
            for(int f=0;f<N;++f) {
                for(int i=0;i<16;++i)words[i]=0;
                auto inbox=reinterpret_cast<__ubuf__ AFDInbox*>(stage);
                inbox->microbatch_idx=task_info->microbatch_idx;inbox->layer_idx=task_info->layer_idx;
                inbox->round_idx=round;inbox->num_comm_rounds=task_info->num_comm_rounds;
                inbox->num_round_tokens=length;inbox->return_bit=saved.next_return_bit;
                inbox->num_recv_records=AscendC::ReadGmByPassDCache(rtotals+f);
                store_control(reinterpret_cast<__gm__ void*>(peers[f].control_base+addresses[lane_idx*N+f].inbox_offset),stage,64);
            }
            for(int i=0;i<16;++i)words[i]=0;
            auto out=reinterpret_cast<__ubuf__ AFDRoundLedger*>(stage);
            out->input_bit=saved.next_input_bit;out->return_bit=saved.next_return_bit;out->round_idx=round;
            out->token_begin=begin;out->num_round_tokens=length;out->final_round=round+1==task_info->num_comm_rounds;
            out->microbatch_idx=task_info->microbatch_idx;
            store_control(ledger,stage,64);words[0]=1;store_control(ledger,stage);
            for(int f=0;f<N;++f) {
                for(int i=0;i<8;++i)words[i]=0;words[0]=saved.next_input_bit;
                store_control(reinterpret_cast<__gm__ void*>(peers[f].control_base+addresses[lane_idx*N+f].input_ready_offset),stage);
            }
            for(int i=0;i<8;++i)words[i]=0;
            auto next=reinterpret_cast<__ubuf__ AFDSendState*>(stage);
            next->next_input_bit=saved.next_input_bit^1;next->next_return_bit=saved.next_return_bit^1;
            next->has_published=1;next->last_input_bit=saved.next_input_bit;store_control(state,stage);
        }
        local_join();
    }
    if(vec_core_idx==0) {
        const uint32_t code=AscendC::ReadGmByPassDCache(error);
        for(int i=0;i<8;++i)words[i]=0;words[0]=code==OK?2:3;words[1]=code;
        words[2]=current_round;store_control(send_status,stage);
    }
}
} // namespace deep_ep::afd
// ### PATCH END: bounded AFD sender
