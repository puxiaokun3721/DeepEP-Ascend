#pragma once
#include <deep_ep/impls/afd/dispatch_recv.hpp>
#include <deep_ep/layout/afd/scheduler.hpp>

// ### PATCH START: AFD independent DP scheduler; native EP has no independent AF scheduler
namespace deep_ep::afd {
// Reuse the pinned-port DMA/cache helpers, not a second memory-ordering implementation.
template<class T> __aicore__ inline void scheduler_write(__gm__ T* dst,const T& value,__ubuf__ uint8_t* tmp) {
    ub_store(tmp,value);
    recv_store(dst,tmp,sizeof(T));
}
__aicore__ inline bool scheduler_same(const AFDLane& a,const AFDPendingLane& p,
                                      const AFDLane& b,const AFDPendingLane& q) {
    return q.valid && a.dp_idx==b.dp_idx && a.phase==b.phase &&
        p.inbox.microbatch_idx==q.inbox.microbatch_idx && p.inbox.layer_idx==q.inbox.layer_idx &&
        p.inbox.round_idx==q.inbox.round_idx && p.inbox.num_comm_rounds==q.inbox.num_comm_rounds;
}
// AFD-only directory: slots are not members. Rebuild in two quadratic passes
// once per invocation, rather than rediscovering representatives for every task.
// Scratch uses Pending's private reserved bytes; payload/Active budgets do not grow.
template<int L>
__aicore__ inline void scheduler_index_members(__gm__ const AFDLane* lanes,
                                               __gm__ AFDPendingLane* pending,uint32_t& error) {
    for(int i=0;i<L;++i) {
        const auto member=recv_read(lanes+i);
        const auto p=recv_read(pending+i);
        uint32_t representative=i;
        for(int j=0;j<i;++j) {
            const auto prior=recv_read(lanes+j);
            if(prior.dp_idx==member.dp_idx && prior.phase==member.phase && prior.member_idx==member.member_idx) {
                if(representative==uint32_t(i))representative=j;
                // Validate duplicates once for the immutable Pending snapshot.
                // No per-candidate GM marks or barriers are needed afterwards.
                if(p.valid && scheduler_same(member,p,prior,recv_read(pending+j)))error=BAD_TASK;
            }
        }
        AscendC::WriteGmByPassDCache(&pending[i].member_leader,representative);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    for(int i=0;i<L;++i) {
        const auto member=recv_read(lanes+i);
        uint32_t count=0;
        for(int j=0;j<L;++j) {
            const auto prior=recv_read(lanes+j);
            if(prior.dp_idx==member.dp_idx && prior.phase==member.phase &&
               recv_read(&pending[j].member_leader)==uint32_t(j))++count;
        }
        AscendC::WriteGmByPassDCache(&pending[i].num_group_members,count);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

// Count matching distinct physical members in one lane scan. Every matched lane
// belongs to the registered group, so equal cardinality + uniqueness proves
// exact membership (including zero-route members and inactive sibling slots).
// Uniqueness was checked by scheduler_index_members before any selection.
template<int L,bool PRECHECK>
__aicore__ inline bool scheduler_ready(int leader,__gm__ const AFDLane* lanes,
    __gm__ AFDPendingLane* pending,__gm__ const AFDReturnLane* returns,
    __gm__ const uint8_t* control,__ubuf__ uint8_t* tmp,uint32_t& error) {
    auto a=recv_read(lanes+leader);auto p=recv_read(pending+leader);
    if(!p.valid) return false;
    for(int i=0;i<leader;++i)
        if(scheduler_same(a,p,recv_read(lanes+i),recv_read(pending+i))) return false;
    bool has_routes=false;
    uint32_t matches=0;
    for(int j=0;j<L;++j) {
        auto b=recv_read(lanes+j);auto q=recv_read(pending+j);
        if(!scheduler_same(a,p,b,q))continue;
        ++matches;
        has_routes|=q.num_remaining_routes!=0;
    }
    if(matches!=p.num_group_members || error!=OK) return false;
    // AFD CONTROL has no useful compute to overlap. A missing credit must not
    // pin the single Active/workspace, even under COMPUTE_AHEAD. A member with
    // zero routes in a nonempty DP entry is not a separate CONTROL entry.
    if(PRECHECK || !has_routes) {
        for(int j=0;j<L;++j) {
            auto b=recv_read(lanes+j);auto q=recv_read(pending+j);
            if(!scheduler_same(a,p,b,q))continue;
            if(q.grant_state!=HELD) {
                auto ret=recv_read(returns+j);
                recv_load(tmp,control+ret.credit_offset,32);
                if(*reinterpret_cast<__ubuf__ uint32_t*>(tmp)!=q.inbox.return_bit)return false;
            }
        }
    }
    return true;
}

template<int N,int E,int K,int L,int NL,int Q,int QA,int A,bool PRECHECK>
__global__ __vector__ void ffn_scheduler_impl(
    __gm__ void* buffer,__gm__ const AFDContextLayout* context,__gm__ void* control_buffer,
    __gm__ void* workspace,__gm__ AFDActive* active,__gm__ AFDExecutionState* execution_state,
    __gm__ AFDCompletion* completion,__gm__ int* psum_num_recv_tokens_per_expert,
    __gm__ int* num_unaligned_recv_tokens_per_expert,const int rank_idx) {
    AscendC::InitSocState();
    if(AscendC::GetBlockIdx()!=0) return;
    constexpr int R=E/N;
    static_assert(R>0 && R<=512 && Q>0 && QA>=Q && QA>=A && A>0 && (A&(A-1))==0);
    // Five R-vectors, bounded independently of L, DP capacity and full maxBS.
    auto tmp=reinterpret_cast<__ubuf__ uint8_t*>(0);
    auto counts=reinterpret_cast<__ubuf__ uint32_t*>(256);
    auto consumed=reinterpret_cast<__ubuf__ uint32_t*>(256+align32(4*R));
    auto rows=reinterpret_cast<__ubuf__ uint32_t*>(256+2*align32(4*R));
    auto prefix=reinterpret_cast<__ubuf__ uint32_t*>(256+3*align32(4*R));
    auto delta=reinterpret_cast<__ubuf__ uint32_t*>(256+4*align32(4*R));
    auto base=reinterpret_cast<__gm__ const uint8_t*>(context);
    auto control=reinterpret_cast<__gm__ uint8_t*>(control_buffer);
    auto ws=reinterpret_cast<__gm__ uint8_t*>(workspace);
    auto pending=reinterpret_cast<__gm__ AFDPendingLane*>(ws+scheduler_pending_offset());
    auto totals=reinterpret_cast<__gm__ uint32_t*>(ws+scheduler_counts_offset(L));
    auto cursors=reinterpret_cast<__gm__ uint32_t*>(ws+scheduler_consumed_offset(L,R));
    auto state=recv_read(reinterpret_cast<__gm__ AFDSchedulerState*>(ws));
    auto done=recv_read(completion);
    auto exec=recv_read(execution_state);
    uint32_t error=OK;
    if(context->abi_version!=kAbiVersion || context->num_source_lanes!=L || context->num_ranks!=N ||
       context->num_experts!=E || context->num_layers!=NL || context->num_topk!=K || rank_idx<0 || rank_idx>=N)
        error=BAD_ABI;
    if(done.state==ARMED || done.error_code || exec.stage==ABORTED) error=STATE_ORDER;
    if(done.state>READY || done.num_members>L || done.num_entries>L || done.num_deltas>uint64_t(L)*R) error=STATE_ORDER;
    // Completion owns every field required after ACK: never recover old data from Inbox.
    auto cb=reinterpret_cast<__gm__ uint8_t*>(completion);
    auto ce=reinterpret_cast<__gm__ AFDActiveEntry*>(cb+completion_entries_offset());
    auto cm=reinterpret_cast<__gm__ AFDActiveMember*>(cb+completion_members_offset(L));
    auto cd=reinterpret_cast<__gm__ uint32_t*>(cb+completion_deltas_offset(L));
    if(error==OK && done.state==READY) {
        auto previous=recv_read(active);
        if(exec.stage!=COMMITTED || !previous.valid || !done.num_entries ||
           done.num_entries!=previous.num_entries || done.num_members!=previous.num_members ||
           done.num_deltas!=uint64_t(done.num_members)*R) error=STATE_ORDER;
        auto old_base=reinterpret_cast<__gm__ const uint8_t*>(active);
        auto old_members=reinterpret_cast<__gm__ const AFDActiveMember*>(old_base+active_members_offset(L));
        auto old_entries=reinterpret_cast<__gm__ const AFDActiveEntry*>(old_base+active_entries_offset());
        // Validate all deltas before mutating persistent cursors.
        for(uint32_t i=0;i<done.num_members && error==OK;++i) {
            auto m=recv_read(cm+i);auto original=recv_read(old_members+i);
            if(m.source_lane_idx!=original.source_lane_idx || m.input_bit!=original.input_bit ||
               m.return_bit!=original.return_bit || m.delta_begin!=original.delta_begin ||
               m.num_experts!=original.num_experts) {error=STATE_ORDER;break;}
            if(m.source_lane_idx>=L || m.num_experts!=R || m.delta_begin>done.num_deltas || R>done.num_deltas-m.delta_begin) {error=BAD_TASK;break;}
            for(uint32_t j=0;j<i;++j) if(recv_read(cm+j).source_lane_idx==m.source_lane_idx) error=BAD_TASK;
            auto p=recv_read(pending+m.source_lane_idx);
            if(!p.valid || m.input_bit!=p.expected_input_bit || m.return_bit!=p.inbox.return_bit) {error=STATE_ORDER;break;}
            recv_load(reinterpret_cast<__ubuf__ uint8_t*>(counts),totals+uint64_t(m.source_lane_idx)*R,4*R);
            recv_load(reinterpret_cast<__ubuf__ uint8_t*>(consumed),cursors+uint64_t(m.source_lane_idx)*R,4*R);
            recv_load(reinterpret_cast<__ubuf__ uint8_t*>(delta),cd+m.delta_begin,4*R);
            uint64_t remaining=0;
            for(int e=0;e<R;++e) {
                if(consumed[e]>counts[e] || delta[e]>counts[e]-consumed[e]) error=BAD_ROUTE;
                else remaining+=counts[e]-consumed[e];
            }
            if(error==OK && remaining!=p.num_remaining_routes)error=STATE_ORDER;
        }
        uint32_t next_member=0;
        for(uint32_t i=0;i<done.num_entries && error==OK;++i) {
            auto ent=recv_read(ce+i);auto original=recv_read(old_entries+i);
            if(ent.pending_idx!=original.pending_idx || ent.member_begin!=original.member_begin ||
               ent.num_members!=original.num_members || ent.segment_begin!=original.segment_begin ||
               ent.num_segments!=original.num_segments || ent.num_rows!=original.num_rows || ent.is_final!=original.is_final) {error=STATE_ORDER;break;}
            if(ent.member_begin!=next_member || !ent.num_members || ent.num_members>done.num_members-next_member ||
               ent.is_final>1 || (ent.is_final ? ent.grant_state!=FINISHED : ent.grant_state!=HELD)) {error=BAD_TASK;break;}
            next_member+=ent.num_members;
            if(ent.is_final) for(uint32_t j=0;j<ent.num_members;++j) {
                auto m=recv_read(cm+ent.member_begin+j);
                recv_load(reinterpret_cast<__ubuf__ uint8_t*>(counts),totals+uint64_t(m.source_lane_idx)*R,4*R);
                recv_load(reinterpret_cast<__ubuf__ uint8_t*>(consumed),cursors+uint64_t(m.source_lane_idx)*R,4*R);
                recv_load(reinterpret_cast<__ubuf__ uint8_t*>(delta),cd+m.delta_begin,4*R);
                for(int e=0;e<R;++e) if(consumed[e]+delta[e]!=counts[e]) error=BAD_ROUTE;
            }
        }
        if(next_member!=done.num_members) error=BAD_TASK;
        for(uint32_t i=0;i<done.num_entries && error==OK;++i) {
            auto ent=recv_read(ce+i);
            if(ent.member_begin>done.num_members || ent.num_members>done.num_members-ent.member_begin ||
               (ent.grant_state!=HELD && ent.grant_state!=FINISHED)) {error=BAD_TASK;break;}
            for(uint32_t j=0;j<ent.num_members;++j) {
                auto m=recv_read(cm+ent.member_begin+j);auto p=recv_read(pending+m.source_lane_idx);
                recv_load(reinterpret_cast<__ubuf__ uint8_t*>(consumed),cursors+uint64_t(m.source_lane_idx)*R,4*R);
                recv_load(reinterpret_cast<__ubuf__ uint8_t*>(delta),cd+m.delta_begin,4*R);
                for(int e=0;e<R;++e) {consumed[e]+=delta[e];p.num_remaining_routes-=delta[e];}
                recv_store(cursors+uint64_t(m.source_lane_idx)*R,reinterpret_cast<__ubuf__ uint8_t*>(consumed),4*R);
                p.grant_state=ent.grant_state;
                if(ent.grant_state==FINISHED) {p.valid=0;p.grant_state=NONE;p.expected_input_bit^=1;}
                scheduler_write(pending+m.source_lane_idx,p,tmp);
            }
        }
        if(error==OK) {done={};scheduler_write(completion,done,tmp);}
    }
    AFDActive out{};out.entries_offset=active_entries_offset();out.members_offset=active_members_offset(L);out.segments_offset=active_segments_offset(L);
    auto ab=reinterpret_cast<__gm__ uint8_t*>(active);
    auto entries=reinterpret_cast<__gm__ AFDActiveEntry*>(ab+out.entries_offset);
    auto members=reinterpret_cast<__gm__ AFDActiveMember*>(ab+out.members_offset);
    auto segments=reinterpret_cast<__gm__ AFDActiveSegment*>(ab+out.segments_offset);
    auto grants=reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(execution_state)+32);
    for(int e=0;e<R;++e) rows[e]=prefix[e]=0;
    // Context directory is constructed and bounds checked by the host, static across replay.
    auto lanes=reinterpret_cast<__gm__ const AFDLane*>(base+context->lanes_offset);
    auto addresses=reinterpret_cast<__gm__ const AFDAddress*>(base+context->addresses_offset);
    AFDReturnDirectory directory{};
    if(error==OK && !recv_range(context->reserved_address,sizeof(AFDReturnDirectory),context->context_bytes)) error=BAD_ABI;
    if(error==OK) directory=recv_read(reinterpret_cast<__gm__ const AFDReturnDirectory*>(base+context->reserved_address));
    auto returns=reinterpret_cast<__gm__ const AFDReturnLane*>(base+directory.lanes_offset);
    if(error==OK && (directory.version!=1 || !recv_range(directory.lanes_offset,32*uint64_t(L),context->context_bytes))) error=BAD_ABI;
    ++state.tick;
    // Snapshot only new matching Ready. ACK cannot occur until a later combineSend.
    for(int i=0;i<L && error==OK;++i) {
        auto p=recv_read(pending+i);if(p.valid) continue;
        auto address=recv_read(addresses+uint64_t(i)*N+rank_idx);
        recv_load(tmp,control+address.input_ready_offset,32);
        if(*reinterpret_cast<__ubuf__ uint32_t*>(tmp)!=p.expected_input_bit) continue;
        auto lane=recv_read(lanes+i);
        recv_load(tmp,control+address.inbox_offset,64);
        p.inbox=ub_load<AFDInbox>(tmp);
        if(!p.inbox.num_comm_rounds || p.inbox.round_idx>=p.inbox.num_comm_rounds || p.inbox.return_bit>1 ||
           p.inbox.num_round_tokens>lane.capacity || p.inbox.num_recv_records>p.inbox.num_round_tokens) {error=BAD_TASK;break;}
        bool layer_ok=false;
        auto layers=reinterpret_cast<__gm__ const uint32_t*>(base+context->layers_offset);
        for(int t=0;t<NL;++t) if(recv_read(layers+t)==p.inbox.layer_idx) layer_ok=true;
        if(!layer_ok) {error=BAD_LAYER;break;}
        recv_load(reinterpret_cast<__ubuf__ uint8_t*>(counts),reinterpret_cast<__gm__ uint8_t*>(buffer)+address.count_offset,4*R);
        uint64_t total=0;
        for(int e=0;e<R;++e) {total+=counts[e];consumed[e]=0;}
        if(total>uint64_t(p.inbox.num_round_tokens)*K) {error=BAD_ROUTE;break;}
        recv_store(totals+uint64_t(i)*R,reinterpret_cast<__ubuf__ uint8_t*>(counts),4*R);
        recv_store(cursors+uint64_t(i)*R,reinterpret_cast<__ubuf__ uint8_t*>(consumed),4*R);
        p.valid=1;p.grant_state=NONE;p.num_remaining_routes=uint32_t(total);
        p.first_seen=state.tick;scheduler_write(pending+i,p,tmp);
    }
    if(error==OK)scheduler_index_members<L>(lanes,pending,error);
    // Phase fairness: choose oldest ready alternative phase after every committed selection.
    int first=-1;uint64_t oldest=~uint64_t(0);bool alternative=false;
    for(int i=0;i<L && error==OK;++i) {
        if(!scheduler_ready<L,PRECHECK>(i,lanes,pending,returns,control,tmp,error)) continue;
        auto p=recv_read(pending+i);auto lane=recv_read(lanes+i);
        for(int j=0;j<L;++j) {auto q=recv_read(pending+j);if(scheduler_same(lane,p,recv_read(lanes+j),q) && q.first_seen<p.first_seen) p.first_seen=q.first_seen;}
        bool alt=state.has_phase && lane.phase!=state.last_phase;
        if(first<0 || (alt && !alternative) || (alt==alternative && p.first_seen<oldest)) {
            first=i;oldest=p.first_seen;alternative=alt;
        }
    }
    if(first>=0 && error==OK) {
        auto initial=recv_read(pending+first);auto initial_lane=recv_read(lanes+first);
        out.phase=initial_lane.phase;out.layer_idx=initial.inbox.layer_idx;
        // Oldest chosen group first; remaining groups in deterministic lane order.
        for(int step=0;step<L && error==OK;++step) {
            int i=(first+step)%L;
            auto p=recv_read(pending+i);auto lane=recv_read(lanes+i);
            if(!p.valid || lane.phase!=out.phase || p.inbox.layer_idx!=out.layer_idx ||
               !scheduler_ready<L,PRECHECK>(i,lanes,pending,returns,control,tmp,error)) continue;
            AFDActiveEntry ent{};ent.pending_idx=i;ent.member_begin=out.num_members;ent.segment_begin=out.num_segments;ent.is_final=1;
            ent.grant_state=PRECHECK?HELD:p.grant_state;
            for(int j=0;j<L;++j) {
                auto q=recv_read(pending+j);auto member_lane=recv_read(lanes+j);
                if(!scheduler_same(lane,p,member_lane,q)) continue;
                AFDActiveMember member{};member.source_lane_idx=j;member.input_bit=q.expected_input_bit;member.return_bit=q.inbox.return_bit;
                member.delta_begin=out.num_members*R;member.num_experts=R;
                scheduler_write(members+out.num_members,member,tmp);++out.num_members;++ent.num_members;
                recv_load(reinterpret_cast<__ubuf__ uint8_t*>(counts),totals+uint64_t(j)*R,4*R);
                recv_load(reinterpret_cast<__ubuf__ uint8_t*>(consumed),cursors+uint64_t(j)*R,4*R);
                uint32_t route_offset=0;
                for(int e=0;e<R;++e) {
                    uint32_t route_begin=route_offset+consumed[e];route_offset+=counts[e];
                    uint32_t remaining=counts[e]-consumed[e];
                    uint32_t old_padded=(rows[e]+A-1)/A*A;
                    uint32_t headroom=QA-out.num_padded_tokens;
                    uint32_t fit=(uint64_t(old_padded)+headroom)/A*A-rows[e];
                    uint32_t take=remaining;
                    if(take>Q-out.num_expanded_tokens) take=Q-out.num_expanded_tokens;
                    if(take>fit) take=fit;
                    if(take<remaining) ent.is_final=0;
                    if(!take) continue;
                    AFDActiveSegment seg{};seg.entry_idx=out.num_entries;seg.source_lane_idx=j;seg.local_expert_idx=e;
                    seg.route_begin=route_begin;seg.num_routes=take;seg.dst_row_begin=rows[e];
                    scheduler_write(segments+out.num_segments,seg,tmp);++out.num_segments;++ent.num_segments;ent.num_rows+=take;
                    rows[e]+=take;out.num_expanded_tokens+=take;out.num_padded_tokens+=(rows[e]+A-1)/A*A-old_padded;
                }
            }
            // Do not acquire a new zero-progress continuation. Truly empty entries must handshake.
            if(!ent.num_rows && !ent.is_final) {out.num_members=ent.member_begin;continue;}
            scheduler_write(entries+out.num_entries,ent,tmp);
            *reinterpret_cast<__ubuf__ uint32_t*>(tmp)=ent.grant_state;recv_store(grants+out.num_entries,tmp,4);
            ++out.num_entries;
        }
        uint32_t running=0;
        for(int e=0;e<R;++e) {prefix[e]=running;running+=(rows[e]+A-1)/A*A;}
        for(uint32_t i=0;i<out.num_segments;++i) {auto seg=recv_read(segments+i);seg.dst_row_begin+=prefix[seg.local_expert_idx];scheduler_write(segments+i,seg,tmp);}
        for(int e=0;e<R;++e) prefix[e]+=(rows[e]+A-1)/A*A;
        if(out.num_entries) {out.valid=1;out.kind=out.num_expanded_tokens?WORK:CONTROL;state.last_phase=out.phase;state.has_phase=1;}
    }
    exec={};done={};
    if(error!=OK) {out={};out.kind=ABORT;out.entries_offset=active_entries_offset();out.members_offset=active_members_offset(L);out.segments_offset=active_segments_offset(L);exec.stage=ABORTED;exec.error_code=error;for(int e=0;e<R;++e) rows[e]=prefix[e]=0;}
    else if(out.valid) {exec.stage=SELECTED;exec.num_entries=out.num_entries;done.state=ARMED;done.num_entries=out.num_entries;done.num_members=out.num_members;done.num_deltas=out.num_members*R;}
    recv_store(psum_num_recv_tokens_per_expert,reinterpret_cast<__ubuf__ uint8_t*>(prefix),4*R);
    recv_store(num_unaligned_recv_tokens_per_expert,reinterpret_cast<__ubuf__ uint8_t*>(rows),4*R);
    scheduler_write(reinterpret_cast<__gm__ AFDSchedulerState*>(ws),state,tmp);
    scheduler_write(execution_state,exec,tmp);scheduler_write(completion,done,tmp);
    // Every dependent table/prefix store completed before publishing valid.
    scheduler_write(active,out,tmp);
}

// Same-stream bridge after actual GMM2; it does not execute or stand in for GMM.
__global__ __vector__ void mark_compute_ready_impl(__gm__ const AFDActive* active,__gm__ AFDExecutionState* execution_state) {
    AscendC::InitSocState();if(AscendC::GetBlockIdx()!=0) return;
    auto task=recv_read(active);if(!task.valid || task.kind==ABORT || task.kind==NOP) return;
    auto state=recv_read(execution_state);
    if(state.stage!=INPUT_READY) {state.stage=ABORTED;state.error_code=STATE_ORDER;}
    else if(task.kind==WORK) state.stage=COMPUTE_READY;
    scheduler_write(execution_state,state,reinterpret_cast<__ubuf__ uint8_t*>(0));
}
}
// ### PATCH END: AFD independent DP scheduler
