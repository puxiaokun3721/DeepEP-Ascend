#pragma once
#include <kernel_operator.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>
#include <c_api/asc_simd.h>
#include <deep_ep/layout/afd/active.hpp>
#include <deep_ep/layout/afd/token.hpp>

// Upstream source: impls/ep/dispatch_copy_epilogue.hpp at 3b25377d.
// The upstream fused epilogue is unsuitable for delegation: it histograms the complete
// transaction and emits K+2 metadata. AFD consumes immutable selected segments only.
// ### PATCH START: AFD selected-piece expansion; no scheduling, polling or ACK
namespace deep_ep::afd {
// Copied from the pinned epilogue's fill_ubuf: padding remains vectorized.
__simd_vf__ void fill_ubuf(__ubuf__ int* ptr, const int num_elems, const int value) {
    using namespace __cce_simd;
    static constexpr int kNumElemsPerVector = sizeof(vector_int32_t) / sizeof(int);
    vector_int32_t value_vec;
    asc_duplicate_scalar(value_vec, value);
    uint32_t remaining = static_cast<uint32_t>(num_elems);
    for (int elem_offset = 0; elem_offset < num_elems; elem_offset += kNumElemsPerVector)
        asc_storealign(ptr + elem_offset, value_vec, asc_update_mask_b32(remaining));
}

__aicore__ inline bool recv_range(uint64_t offset,uint64_t bytes,uint64_t length) {
    return offset<=length && bytes<=length-offset;
}
__aicore__ inline void recv_load(__ubuf__ uint8_t* dst,__gm__ const void* src,uint32_t bytes,event_t id=EVENT_ID6) {
    asc_copy_gm2ub_align(dst,reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ void*>(src)),1,bytes,0,0,false,
        static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),uint64_t(bytes),align32(bytes));
    asc_sync_notify(PIPE_MTE2,PIPE_S,id);asc_sync_wait(PIPE_MTE2,PIPE_S,id);
}
__aicore__ inline void recv_store(__gm__ void* dst,__ubuf__ uint8_t* src,uint32_t bytes,event_t id=EVENT_ID6) {
    asc_sync_notify(PIPE_S,PIPE_MTE3,id);asc_sync_wait(PIPE_S,PIPE_MTE3,id);
    asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(dst),src,1,bytes,
        static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NOTALLOC_CI),uint64_t(bytes),bytes);
    asc_sync_notify(PIPE_MTE3,PIPE_S,id);asc_sync_wait(PIPE_MTE3,PIPE_S,id);
}
__aicore__ inline void recv_join() {
    asc_sync_notify(PIPE_MTE3,PIPE_V,EVENT_ID7);asc_sync_wait(PIPE_MTE3,PIPE_V,EVENT_ID7);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();AscendC::SyncAll<true>();
}
// Entries, members, segments and route tables change between launches. Explicit
// scalar bypass loads match the pinned sender's state reads; immutable Active
// publication keeps all fields stable for this kernel. A DDR barrier alone would
// not invalidate an old scalar DCache line.
template<typename T>
__aicore__ inline T recv_read(__gm__ const T* source) {
    T value{};
    auto input=reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ T*>(source));
    auto output=reinterpret_cast<uint32_t*>(&value);
    #pragma unroll
    for(int i=0;i<sizeof(T)/sizeof(uint32_t);++i)
        output[i]=AscendC::ReadGmByPassDCache(input+i);
    return value;
}
template<int D,int T> struct alignas(32) DispatchRecvUB {
    alignas(32) uint8_t token[2][D*T];
    alignas(32) uint8_t metadata[2][align32(16*T)];
    alignas(32) uint8_t refs[2][align32(4*T)];
    alignas(32) uint8_t segment[2][32];
    alignas(32) uint8_t zero[512];
    alignas(32) uint8_t control[256];
};

// Static trusted sizes come from AFDBuffer, never from a device-produced row count.
// control_buffer is the owner-local Inbox/Ready allocation, added to the design's
// internal signature because the existing 224B Context stores relative offsets.
template<int N,int E,int K,int NC,int X,int S,int L,int NL,int Q,int QA,int A,int T,int U>
__global__ __vector__ void dispatch_recv_impl(
    __gm__ void* buffer,__gm__ const AFDContextLayout* context,__gm__ const AFDActive* active,
    __gm__ AFDExecutionState* execution_state,__gm__ void* workspace,__gm__ void* recv_x,
    __gm__ int16_t* recv_sf,__gm__ int* recv_src_metadata,__gm__ void* control_buffer,
    const int rank_idx,const int recv_sf_token_stride,const int recv_sf_hidden_stride,
    const uint64_t payload_bytes,const uint64_t control_bytes,const uint64_t context_bytes) {
    AscendC::InitSocState();
    constexpr TokenLayout token_layout{X,S*2,K};
    constexpr int D=token_layout.record_stride(),M=token_layout.metadata_stride(),R=E/N;
    using UB=DispatchRecvUB<D,T>;
    static_assert(sizeof(UB)<=U && T>=1 && T<=16 && R<=512);
    const int vec_core_idx=AscendC::GetBlockIdx();
    auto ub=reinterpret_cast<__ubuf__ UB*>(0);
    auto tmp=ub->control;
    auto words=reinterpret_cast<__ubuf__ uint32_t*>(tmp);
    auto ws=reinterpret_cast<__gm__ uint8_t*>(workspace);
    auto result=reinterpret_cast<__gm__ uint32_t*>(ws+32*NC);
    recv_load(tmp,active,64);
    const AFDActive task=*reinterpret_cast<__ubuf__ AFDActive*>(tmp);
    // Immutable publication guarantees every launched AIV sees the same gate.
    // NOP/ABORT read no Context directory, Inbox, route or record.
    if(task.kind==ABORT && task.valid==0) return;
    if(vec_core_idx==0) {
        uint32_t error=OK;
        recv_load(tmp,execution_state,32);
        const AFDExecutionState state=*reinterpret_cast<__ubuf__ AFDExecutionState*>(tmp);
        if(task.kind==NOP && task.valid==0) {
            if(task.num_entries || task.num_members || task.num_segments || task.num_expanded_tokens ||
               task.num_padded_tokens || state.stage!=EMPTY || state.error_code!=OK || state.num_entries) error=STATE_ORDER;
        } else if(task.valid!=1 || (task.kind!=WORK && task.kind!=CONTROL)) error=BAD_TASK;
        else if(state.stage!=SELECTED || state.error_code!=OK || state.num_entries!=task.num_entries) error=STATE_ORDER;
        if(error==OK && task.kind!=NOP) {
            if(context->abi_version!=kAbiVersion || context->num_ranks!=N || context->num_experts!=E ||
               context->num_topk!=K || context->num_vec_cores!=NC || context->num_hidden_bytes!=X ||
               context->num_sf_packs!=S || context->num_source_lanes!=L || context->num_layers!=NL ||
               context->context_bytes!=context_bytes || context->payload_bytes!=payload_bytes ||
               context->control_bytes!=control_bytes || rank_idx<0 || rank_idx>=N) error=BAD_ABI;
            if(error==OK && (context->lanes_offset!=sizeof(AFDContextLayout) ||
               context->peers_offset!=context->lanes_offset+32*uint64_t(L) ||
               context->addresses_offset!=context->peers_offset+32*uint64_t(N) ||
               context->layers_offset!=context->addresses_offset+64*uint64_t(L)*N ||
               !recv_range(context->layers_offset,4*uint64_t(NL),context_bytes))) error=BAD_ABI;
            if(error==OK && (task.entries_offset!=active_entries_offset() ||
               task.members_offset!=active_members_offset(L) || task.segments_offset!=active_segments_offset(L) ||
               task.num_entries==0 || task.num_entries>L || task.num_members>L ||
               task.num_segments>uint64_t(L)*R || task.num_expanded_tokens>Q || task.num_padded_tokens>QA)) error=CAPACITY;
        }
        // Do not form any untrusted table pointer until the directory is checked.
        if(error==OK && task.kind!=NOP) {
            auto base=reinterpret_cast<__gm__ const uint8_t*>(context);
            auto lanes=reinterpret_cast<__gm__ const AFDLane*>(base+context->lanes_offset);
            auto addresses=reinterpret_cast<__gm__ const AFDAddress*>(base+context->addresses_offset);
            auto layers=reinterpret_cast<__gm__ const uint32_t*>(base+context->layers_offset);
            auto ab=reinterpret_cast<__gm__ const uint8_t*>(active);
            auto entries=reinterpret_cast<__gm__ const AFDActiveEntry*>(ab+task.entries_offset);
            auto members=reinterpret_cast<__gm__ const AFDActiveMember*>(ab+task.members_offset);
            auto segments=reinterpret_cast<__gm__ const AFDActiveSegment*>(ab+task.segments_offset);
            auto grants=reinterpret_cast<__gm__ const uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(execution_state)+32);
            bool legal_layer=false;for(int i=0;i<NL;++i)legal_layer|=layers[i]==task.layer_idx;
            if(!legal_layer)error=BAD_LAYER;
            // Freeze the sender's actual contiguous wire directory. Exact offsets also
            // reject aligned aliases/overlapping regions before any count/ref read.
            uint64_t wire_end=0;
            for(int l=0;l<L && error==OK;++l) {
                const auto lane=lanes[l];
                if(lane.capacity==0 || lane.capacity>kMaxLaneTokens || lane.src_rank_idx>=context->num_src_ranks ||
                   lane.payload_slot_idx>=context->num_slots || uint64_t(lane.capacity)*K>=UINT32_MAX) {error=BAD_ABI;break;}
                const uint64_t record_offset=wire_end;wire_end+=align512(uint64_t(lane.capacity)*D);
                const uint64_t ref_offset=wire_end;wire_end+=align32(4*uint64_t(lane.capacity)*(K<R?K:R));
                const uint64_t count_offset=wire_end;wire_end+=align32(4*uint64_t(R));
                const uint64_t prefix_offset=wire_end;wire_end+=align32(4*uint64_t(R));
                if(wire_end>payload_bytes) {error=BAD_ABI;break;}
                for(int f=0;f<N;++f) {
                    const auto address=addresses[uint64_t(l)*N+f];
                    if(address.records_offset!=record_offset || address.route_reference_offset!=ref_offset ||
                       address.count_offset!=count_offset || address.prefix_offset!=prefix_offset ||
                       address.inbox_offset!=uint64_t(l)*128 || address.input_ready_offset!=uint64_t(l)*128+64 ||
                       !recv_range(address.inbox_offset,128,control_bytes))error=BAD_ABI;
                }
            }
            if(error==OK && (wire_end!=payload_bytes || control_bytes!=128*uint64_t(L)))error=BAD_ABI;
            uint64_t next_member=0,next_segment=0,total=0;
            for(uint32_t i=0;i<task.num_entries && error==OK;++i) {
                const auto entry=recv_read(entries+(i));
                if(entry.pending_idx>=L || entry.member_begin!=next_member || entry.segment_begin!=next_segment ||
                   entry.num_members==0 || entry.num_members>task.num_members-next_member ||
                   entry.num_segments>task.num_segments-next_segment || entry.is_final>1 || entry.grant_state>HELD ||
                   recv_read(grants+(i))!=entry.grant_state) {error=BAD_TASK;break;}
                if(task.kind==CONTROL && entry.is_final!=1) {error=BAD_TASK;break;}
                for(uint32_t j=0;j<i;++j)if(recv_read(entries+(j)).pending_idx==entry.pending_idx)error=BAD_TASK;
                next_member+=entry.num_members;next_segment+=entry.num_segments;
                uint64_t rows=0;
                uint32_t first_dp=0,first_mb=0,first_round=0,first_rounds=0;
                for(uint32_t j=entry.member_begin;j<next_member && error==OK;++j) {
                    const auto member=recv_read(members+(j));
                    if(member.source_lane_idx>=L || member.input_bit>1 || member.return_bit>1 || member.num_experts!=R ||
                       member.delta_begin!=j*uint64_t(R)) {error=BAD_TASK;break;}
                    for(uint32_t k=0;k<j;++k)if(recv_read(members+(k)).source_lane_idx==member.source_lane_idx)error=BAD_TASK;
                    if(error!=OK)break;
                    const auto lane=lanes[member.source_lane_idx];
                    const auto address=addresses[uint64_t(member.source_lane_idx)*N+rank_idx];
                    if(lane.phase!=task.phase || lane.capacity==0 || lane.capacity>kMaxLaneTokens ||
                       !recv_range(address.records_offset,uint64_t(lane.capacity)*D,payload_bytes) ||
                       !recv_range(address.route_reference_offset,4*uint64_t(lane.capacity)*(K<R?K:R),payload_bytes) ||
                       !recv_range(address.count_offset,4*uint64_t(R),payload_bytes) ||
                       !recv_range(address.prefix_offset,4*uint64_t(R),payload_bytes) ||
                       !recv_range(address.inbox_offset,64,control_bytes) ||
                       !recv_range(address.input_ready_offset,32,control_bytes)) {error=BAD_ABI;break;}
                    recv_load(tmp,reinterpret_cast<__gm__ uint8_t*>(control_buffer)+address.inbox_offset,64);
                    const AFDInbox inbox=*reinterpret_cast<__ubuf__ AFDInbox*>(tmp);
                    recv_load(tmp,reinterpret_cast<__gm__ uint8_t*>(control_buffer)+address.input_ready_offset,32);
                    if(words[0]!=member.input_bit || inbox.return_bit!=member.return_bit || inbox.layer_idx!=task.layer_idx ||
                       inbox.num_round_tokens>lane.capacity || inbox.num_recv_records>inbox.num_round_tokens ||
                       inbox.num_comm_rounds==0 || inbox.round_idx>=inbox.num_comm_rounds) {error=BAD_TASK;break;}
                    if(task.kind==CONTROL && inbox.num_recv_records!=0) {error=BAD_TASK;break;}
                    if(j==entry.member_begin) {first_dp=lane.dp_idx;first_mb=inbox.microbatch_idx;first_round=inbox.round_idx;first_rounds=inbox.num_comm_rounds;}
                    else if(first_dp!=lane.dp_idx || first_mb!=inbox.microbatch_idx || first_round!=inbox.round_idx ||
                            first_rounds!=inbox.num_comm_rounds) {error=BAD_TASK;break;}
                    auto counts=reinterpret_cast<__gm__ const uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(buffer)+address.count_offset);
                    auto prefix=reinterpret_cast<__gm__ const uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(buffer)+address.prefix_offset);
                    uint64_t previous=0;
                    for(int e=0;e<R;++e) {
                        const uint64_t end=recv_read(prefix+(e));
                        if(end<previous || end-previous!=recv_read(counts+(e)) || end>uint64_t(lane.capacity)*(K<R?K:R)) {error=BAD_ROUTE;break;}
                        previous=end;
                    }
                    if(error==OK && task.kind==CONTROL && previous!=0)error=BAD_TASK;
                }
                for(uint32_t j=entry.segment_begin;j<next_segment && error==OK;++j) {
                    const auto segment=recv_read(segments+(j));
                    if(segment.entry_idx!=i || segment.source_lane_idx>=L || segment.local_expert_idx>=R ||
                       segment.num_routes==0) {error=BAD_TASK;break;}
                    bool found=false;for(uint32_t m=entry.member_begin;m<next_member;++m)found|=recv_read(members+(m)).source_lane_idx==segment.source_lane_idx;
                    if(!found) {error=BAD_TASK;break;}
                    const auto address=addresses[uint64_t(segment.source_lane_idx)*N+rank_idx];
                    auto prefix=reinterpret_cast<__gm__ const uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(buffer)+address.prefix_offset);
                    const uint64_t begin=segment.local_expert_idx==0?0:recv_read(prefix+(segment.local_expert_idx-1)),end=recv_read(prefix+(segment.local_expert_idx));
                    if(segment.route_begin<begin || segment.route_begin>end || segment.num_routes>end-segment.route_begin) {error=BAD_ROUTE;break;}
                    // The same input route cannot appear twice, even at different output rows.
                    for(uint32_t k=0;k<j;++k) {
                        const auto other=recv_read(segments+(k));
                        if(other.source_lane_idx==segment.source_lane_idx && other.local_expert_idx==segment.local_expert_idx &&
                           uint64_t(segment.route_begin)<uint64_t(other.route_begin)+other.num_routes &&
                           uint64_t(other.route_begin)<uint64_t(segment.route_begin)+segment.num_routes)error=BAD_ROUTE;
                    }
                    rows+=segment.num_routes;
                }
                if(error==OK && rows!=entry.num_rows)error=BAD_TASK;
                total+=rows;
            }
            if(error==OK && (next_member!=task.num_members || next_segment!=task.num_segments || total!=task.num_expanded_tokens))error=BAD_TASK;
            if(error==OK && ((task.kind==WORK && total==0) || (task.kind==CONTROL && (total || task.num_segments || task.num_padded_tokens))))error=BAD_TASK;
            uint64_t physical=0;
            // Repeated table scans bound scratch to the current computation piece.
            for(int e=0;e<R && error==OK;++e) {
                uint64_t count=0;for(uint32_t j=0;j<task.num_segments;++j)if(recv_read(segments+(j)).local_expert_idx==e)count+=recv_read(segments+(j)).num_routes;
                const uint64_t padded=(count+A-1)/A*A;
                if(physical>QA || padded>QA-physical) {error=CAPACITY;break;}
                for(uint32_t j=0;j<task.num_segments && error==OK;++j) {
                    const auto segment=recv_read(segments+(j));if(segment.local_expert_idx!=e)continue;
                    if(segment.dst_row_begin<physical || segment.dst_row_begin>physical+count ||
                       segment.num_routes>physical+count-segment.dst_row_begin) {error=BAD_TASK;break;}
                    for(uint32_t k=0;k<j;++k) {
                        const auto other=recv_read(segments+(k));
                        if(uint64_t(segment.dst_row_begin)<uint64_t(other.dst_row_begin)+other.num_routes &&
                           uint64_t(other.dst_row_begin)<uint64_t(segment.dst_row_begin)+segment.num_routes)error=BAD_TASK;
                    }
                }
                physical+=padded;
            }
            if(error==OK && physical!=task.num_padded_tokens)error=BAD_TASK;

        }
        for(int i=0;i<8;++i)words[i]=0;
        words[0]=error;recv_store(result,tmp,32);
        if(error!=OK) {
            recv_load(tmp,execution_state,32);words[0]=ABORTED;words[1]=error;recv_store(execution_state,tmp,32);
        }
    }
    recv_join();
    recv_load(tmp,result,32);
    if(words[0]!=OK || task.kind==NOP)return;
    if(task.kind==WORK) {
        auto ab=reinterpret_cast<__gm__ const uint8_t*>(active);
        auto segments=reinterpret_cast<__gm__ const AFDActiveSegment*>(ab+task.segments_offset);
        auto base=reinterpret_cast<__gm__ const uint8_t*>(context);
        auto addresses=reinterpret_cast<__gm__ const AFDAddress*>(base+context->addresses_offset);
        auto lanes=reinterpret_cast<__gm__ const AFDLane*>(base+context->lanes_offset);
        const uint32_t start=uint64_t(task.num_padded_tokens)*vec_core_idx/NC;
        const uint32_t end=uint64_t(task.num_padded_tokens)*(vec_core_idx+1)/NC;
        constexpr auto load_cache=static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN);
        constexpr auto store_cache=static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NOTALLOC_CI);

        // Phase 1 checked every directory and interval before any worker follows
        // a route. Phase 2 is parallel, metadata-only and commits no output.
        // Assign by destination intersection so skewed expert distributions still
        // divide across cores. The existing 32B/core scratch holds error votes.
        uint32_t error=OK;
        for(uint32_t j=0;j<task.num_segments && error==OK;++j) {
            const auto segment=recv_read(segments+j);
            const uint32_t begin=segment.dst_row_begin>start?segment.dst_row_begin:start;
            const uint32_t segment_end=segment.dst_row_begin+segment.num_routes;
            const uint32_t limit=segment_end<end?segment_end:end;
            if(begin>=limit)continue;
            const auto address=addresses[uint64_t(segment.source_lane_idx)*N+rank_idx];
            const auto lane=lanes[segment.source_lane_idx];
            recv_load(tmp,reinterpret_cast<__gm__ uint8_t*>(control_buffer)+address.inbox_offset,64);
            const auto inbox=*reinterpret_cast<__ubuf__ AFDInbox*>(tmp);
            auto refs=reinterpret_cast<__gm__ const uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(buffer)+address.route_reference_offset);
            for(uint32_t row=begin;row<limit && error==OK;) {
                const uint32_t num_stage_tokens=uint32_t(T)<limit-row?T:limit-row;
                recv_load(ub->refs[0],refs+segment.route_begin+row-segment.dst_row_begin,4*num_stage_tokens);
                auto stage_refs=reinterpret_cast<__ubuf__ uint32_t*>(ub->refs[0]);
                // Check ALL addresses before issuing this batch's indirect DMAs.
                for(uint32_t i=0;i<num_stage_tokens;++i)
                    if(stage_refs[i]==UINT32_MAX || stage_refs[i]/K>=inbox.num_recv_records)error=BAD_ROUTE;
                if(error!=OK)break;
                for(uint32_t i=0;i<num_stage_tokens;++i)
                    asc_copy_gm2ub_align(ub->token[0]+uint64_t(i)*D,
                        reinterpret_cast<__gm__ uint8_t*>(buffer)+address.records_offset+uint64_t(stage_refs[i]/K)*D+X,
                        1,M,0,0,false,load_cache,uint64_t(M),M);
                asc_sync_notify(PIPE_MTE2,PIPE_S,EVENT_ID0);
                asc_sync_wait(PIPE_MTE2,PIPE_S,EVENT_ID0);
                for(uint32_t i=0;i<num_stage_tokens;++i) {
                    const uint32_t slot=stage_refs[i]%K;
                    auto metadata=ub->token[0]+uint64_t(i)*D;
                    const auto idx=reinterpret_cast<__ubuf__ int64_t*>(metadata+token_layout.topk_idx_offset()-X)[slot];
                    const auto source=*reinterpret_cast<__ubuf__ uint32_t*>(metadata+token_layout.src_token_idx_offset()-X);
                    const auto mask=*reinterpret_cast<__ubuf__ uint32_t*>(metadata+token_layout.valid_topk_mask_offset()-X);
                    if(source>=inbox.num_round_tokens || source>=lane.capacity || !(mask&(uint32_t(1)<<slot)) ||
                       idx!=int64_t(rank_idx)*R+segment.local_expert_idx)error=BAD_ROUTE;
                    if constexpr(K<32)if((mask>>K)!=0)error=BAD_ROUTE;
                }
                row+=num_stage_tokens;
            }
        }
        for(int i=0;i<8;++i)words[i]=0;
        words[0]=error;recv_store(ws+32*vec_core_idx,tmp,32);
        recv_join();
        if(vec_core_idx==0) {
            uint32_t first_error=OK;
            for(int i=0;i<NC;++i) {
                recv_load(tmp,ws+32*i,32);
                if(first_error==OK)first_error=words[0];
            }
            for(int i=0;i<8;++i)words[i]=0;
            words[0]=first_error;recv_store(result,tmp,32);
            if(first_error!=OK) {
                recv_load(tmp,execution_state,32);words[0]=ABORTED;words[1]=first_error;
                recv_store(execution_state,tmp,32);
            }
        }
        recv_join();recv_load(tmp,result,32);
        if(words[0]!=OK)return;

        // Same stage ownership as the native epilogue: MTE2 -> S publishes a
        // loaded stage, MTE3 -> MTE2 protects refill, and a separate MTE3 -> S
        // event protects the four-column metadata stage from scalar overwrite.
        const auto issue_output = [&](int stage_idx,uint32_t row,uint32_t num_stage_tokens) __aicore__ {
            asc_sync_notify(PIPE_S,PIPE_MTE3,static_cast<event_t>(stage_idx));
            asc_sync_wait(PIPE_S,PIPE_MTE3,static_cast<event_t>(stage_idx));
            for(uint32_t i=0;i<num_stage_tokens;++i) {
                auto record=ub->token[stage_idx]+uint64_t(i)*D;
                asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(recv_x)+uint64_t(row+i)*X,
                    record,1,X,store_cache,uint64_t(X),X);
                if constexpr(S>0)if(recv_sf_hidden_stride==1)
                    asc_copy_ub2gm_align(recv_sf+uint64_t(row+i)*recv_sf_token_stride,
                        reinterpret_cast<__ubuf__ int16_t*>(record+X),1,S*2,store_cache,uint64_t(S*2),align32(S*2));
            }
            if constexpr(S>0)if(recv_sf_hidden_stride!=1) {
                // Native EP performs column conversion on host. AFD cannot add
                // a piece-sized HBM mirror: recycle the first record's DEAD
                // hidden area after hidden MTE3 completion, retaining all SF.
                // Each column has a 32B-aligned source and contains <= T packs.
                // The other token stage can be loading throughout this phase.
                static_assert(X>=32);
                constexpr int kNumColumnsPerStage=X/32;
                asc_sync_notify(PIPE_MTE3,PIPE_S,static_cast<event_t>(4+stage_idx));
                asc_sync_wait(PIPE_MTE3,PIPE_S,static_cast<event_t>(4+stage_idx));
                for(int sf_begin=0;sf_begin<S;sf_begin+=kNumColumnsPerStage) {
                    const int columns=kNumColumnsPerStage<S-sf_begin?kNumColumnsPerStage:S-sf_begin;
                    for(int p=0;p<columns;++p) {
                        auto column=reinterpret_cast<__ubuf__ int16_t*>(ub->token[stage_idx]+32*p);
                        for(uint32_t i=0;i<num_stage_tokens;++i)
                            column[i]=reinterpret_cast<__ubuf__ int16_t*>(ub->token[stage_idx]+uint64_t(i)*D+X)[sf_begin+p];
                    }
                    asc_sync_notify(PIPE_S,PIPE_MTE3,static_cast<event_t>(4+stage_idx));
                    asc_sync_wait(PIPE_S,PIPE_MTE3,static_cast<event_t>(4+stage_idx));
                    // One DMA per COLUMN TILE, not one DMA per token/pack.
                    // A one-block native copy avoids SDK-specific stride/gap interpretations.
                    for(int p=0;p<columns;++p)
                        asc_copy_ub2gm_align(recv_sf+uint64_t(row)*recv_sf_token_stride+uint64_t(sf_begin+p)*recv_sf_hidden_stride,
                            reinterpret_cast<__ubuf__ int16_t*>(ub->token[stage_idx]+32*p),
                            1,2*num_stage_tokens,store_cache,uint64_t(2*num_stage_tokens),32);
                    asc_sync_notify(PIPE_MTE3,PIPE_S,static_cast<event_t>(4+stage_idx));
                    asc_sync_wait(PIPE_MTE3,PIPE_S,static_cast<event_t>(4+stage_idx));
                }
            }
            asc_copy_ub2gm_align(recv_src_metadata+uint64_t(row)*4,
                reinterpret_cast<__ubuf__ int32_t*>(ub->metadata[stage_idx]),1,16*num_stage_tokens,
                store_cache,uint64_t(16*num_stage_tokens),align32(16*num_stage_tokens));
            asc_sync_notify(PIPE_MTE3,PIPE_MTE2,static_cast<event_t>(stage_idx));
            asc_sync_notify(PIPE_MTE3,PIPE_S,static_cast<event_t>(2+stage_idx));
        };

        // Segment-major traversal replaces O(padded_rows * segments). Segment
        // order is arbitrary; validated disjoint destination intervals allow
        // independent output. A segment crossing cores is clipped, not dropped.
        for(uint32_t j=0;j<task.num_segments;++j) {
            const auto segment=recv_read(segments+j);
            const uint32_t begin=segment.dst_row_begin>start?segment.dst_row_begin:start;
            const uint32_t segment_end=segment.dst_row_begin+segment.num_routes;
            const uint32_t limit=segment_end<end?segment_end:end;
            if(begin>=limit)continue;
            const auto address=addresses[uint64_t(segment.source_lane_idx)*N+rank_idx];
            auto refs=reinterpret_cast<__gm__ const uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(buffer)+address.route_reference_offset);
            uint32_t issue_row=begin;
            uint32_t stage_num_tokens[2]={0,0},stage_token_base_idx[2]={0,0};
            bool stage_has_output[2]={false,false};
            const auto issue_load = [&](int load_stage_idx,bool is_preload) __aicore__ {
                if(issue_row==limit)return false;
                const uint32_t num_stage_tokens=uint32_t(T)<limit-issue_row?T:limit-issue_row;
                recv_load(ub->refs[load_stage_idx],refs+segment.route_begin+issue_row-segment.dst_row_begin,4*num_stage_tokens);
                auto stage_refs=reinterpret_cast<__ubuf__ uint32_t*>(ub->refs[load_stage_idx]);
                // refs have no MTE3 consumer: fetch them BEFORE gating record
                // refill. A synchronous ref fetch behind this event would stall
                // S until current output completed and destroy stage overlap.
                if(!is_preload)
                    asc_sync_wait(PIPE_MTE3,PIPE_MTE2,static_cast<event_t>(load_stage_idx));
                // AFD Active is an indirect gather, unlike native PSumIterator's
                // contiguous stream. Coalesce adjacent record IDs when possible.
                for(uint32_t i=0;i<num_stage_tokens;) {
                    uint32_t run=1;
                    while(i+run<num_stage_tokens && stage_refs[i+run]/K==stage_refs[i]/K+run)++run;
                    // Internal record padding is the cost of coalescing; skip
                    // the final record's padding (and all padding for run == 1).
                    const uint32_t num_bytes=(run-1)*D+X+M;
                    asc_copy_gm2ub_align(ub->token[load_stage_idx]+uint64_t(i)*D,
                        reinterpret_cast<__gm__ uint8_t*>(buffer)+address.records_offset+uint64_t(stage_refs[i]/K)*D,
                        1,num_bytes,0,0,false,load_cache,uint64_t(num_bytes),num_bytes);
                    i+=run;
                }
                asc_sync_notify(PIPE_MTE2,PIPE_S,static_cast<event_t>(load_stage_idx));
                stage_num_tokens[load_stage_idx]=num_stage_tokens;
                stage_token_base_idx[load_stage_idx]=issue_row;
                issue_row+=num_stage_tokens;
                return true;
            };
            issue_load(0,true);issue_load(1,true);
            int stage_idx=0;
            uint32_t consumed=begin;
            while(consumed<limit) {
                asc_sync_wait(PIPE_MTE2,PIPE_S,static_cast<event_t>(stage_idx));
                if(stage_has_output[stage_idx])
                    asc_sync_wait(PIPE_MTE3,PIPE_S,static_cast<event_t>(2+stage_idx));
                const uint32_t num_stage_tokens=stage_num_tokens[stage_idx];
                auto stage_refs=reinterpret_cast<__ubuf__ uint32_t*>(ub->refs[stage_idx]);
                for(uint32_t i=0;i<num_stage_tokens;++i) {
                    auto out=reinterpret_cast<__ubuf__ int32_t*>(ub->metadata[stage_idx])+4*i;
                    out[0]=segment.source_lane_idx;
                    out[1]=*reinterpret_cast<__ubuf__ int32_t*>(ub->token[stage_idx]+uint64_t(i)*D+token_layout.src_token_idx_offset());
                    out[2]=stage_refs[i]%K;out[3]=segment.entry_idx;
                }
                issue_output(stage_idx,stage_token_base_idx[stage_idx],num_stage_tokens);
                stage_has_output[stage_idx]=true;
                consumed+=num_stage_tokens;
                // A completed stage's release event is consumed either by refill
                // here or the drain below, never twice and never left pending.
                if(!issue_load(stage_idx,false))
                    asc_sync_wait(PIPE_MTE3,PIPE_MTE2,static_cast<event_t>(stage_idx));
                stage_idx^=1;
            }
            for(int i=0;i<2;++i)if(stage_has_output[i])
                asc_sync_wait(PIPE_MTE3,PIPE_S,static_cast<event_t>(2+i));
        }

        // Padding is derived once per expert, rather than rediscovered for each
        // output row. Native vector fill is retained; no route or record is read.
        bool padding_initialized=false;
        uint32_t physical=0;
        for(int e=0;e<R;++e) {
            uint32_t count=0;
            for(uint32_t j=0;j<task.num_segments;++j) {
                const auto segment=recv_read(segments+j);
                if(segment.local_expert_idx==e)count+=segment.num_routes;
            }
            const uint32_t padded=(count+A-1)/A*A;
            uint32_t row=physical+count>start?physical+count:start;
            const uint32_t limit=physical+padded<end?physical+padded:end;
            while(row<limit) {
                if(!padding_initialized) {
                    asc_sync_notify(PIPE_MTE3,PIPE_V,EVENT_ID0);
                    asc_sync_wait(PIPE_MTE3,PIPE_V,EVENT_ID0);
                    asc_sync_notify(PIPE_MTE2,PIPE_V,EVENT_ID0);
                    asc_sync_wait(PIPE_MTE2,PIPE_V,EVENT_ID0);
                    fill_ubuf(reinterpret_cast<__ubuf__ int*>(ub->token[0]),D*T/4,0);
                    fill_ubuf(reinterpret_cast<__ubuf__ int*>(ub->metadata[0]),align32(16*T)/4,-1);
                    asc_sync_notify(PIPE_V,PIPE_S,EVENT_ID0);asc_sync_wait(PIPE_V,PIPE_S,EVENT_ID0);
                    asc_sync_notify(PIPE_V,PIPE_MTE3,EVENT_ID0);asc_sync_wait(PIPE_V,PIPE_MTE3,EVENT_ID0);
                    padding_initialized=true;
                }
                const uint32_t num_stage_tokens=uint32_t(T)<limit-row?T:limit-row;
                issue_output(0,row,num_stage_tokens);
                asc_sync_wait(PIPE_MTE3,PIPE_MTE2,EVENT_ID0);
                asc_sync_wait(PIPE_MTE3,PIPE_S,EVENT_ID2);
                row+=num_stage_tokens;
            }
            physical+=padded;
        }
    }
    recv_join(); // every output DMA has completed before INPUT_READY is published
    if(vec_core_idx==0) {
        recv_load(tmp,execution_state,32);words[0]=INPUT_READY;words[1]=OK;recv_store(execution_state,tmp,32);
    }
}
} // namespace deep_ep::afd
// ### PATCH END: AFD selected-piece expansion
