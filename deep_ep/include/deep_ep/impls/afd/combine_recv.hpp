#pragma once
#include <deep_ep/impls/afd/dispatch_send.hpp>
#include <deep_ep/impls/afd/dispatch_recv.hpp>
#include <deep_ep/layout/afd/return.hpp>

// ### PATCH START: AFD weighted per-route epilogue, based on ep/combine_reduce_epilogue.hpp
namespace deep_ep::afd {
template<int K,int HC> struct alignas(32) CombineRecvUB {
    alignas(32) bfloat16_t input_stage[2][HC];
    alignas(32) float acc_fp32[HC], value_fp32[HC];
    alignas(32) bfloat16_t output_bf16[HC];
    alignas(32) float weights[(K+7)/8*8];
    alignas(32) int64_t expert_idx[(K+3)/4*4];
    alignas(32) AFDRoundLedger ledger;
    alignas(32) uint8_t control_stage[64], status_zero_stage[128];
};
constexpr uint64_t combine_recv_workspace_bytes(uint32_t nc) { return 256+align32(32*uint64_t(nc)); }
template<typename T> __aicore__ inline AscendC::LocalTensor<T> combine_local(__ubuf__ T* ptr,uint32_t size) {
    return AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,uint32_t(reinterpret_cast<uint64_t>(ptr)),size);
}
// Native ring reducer DMA shape and two-stage lifetime, with weighted FP32 accumulation.
// Rank-partial dedup and BF16 count=2 shortcut are deliberately absent: every route has its own weight.
template<int E,int N,int K,int H,int HC,int NC,int64_t Timeout>
__global__ __vector__ void combine_recv_impl(
    __gm__ bfloat16_t* combined_x,__gm__ int64_t* combined_topk_idx,
    __gm__ float* topk_weights,__gm__ uint8_t* token_mask,__gm__ uint8_t* buffer,
    __gm__ const AFDTaskInfo* task_info,__gm__ const AFDContextLayout* context,
    __gm__ void* control_buffer,__gm__ AFDRoundLedger* round_ledger,
    __gm__ void* workspace,__gm__ uint32_t* recv_status) {
    static_assert(H%128==0 && HC%128==0 && HC>0 && HC<=H && K>0 && K<=32);
    AscendC::InitSocState();
    const int vec_core_idx=AscendC::GetBlockIdx();
    const auto task=recv_read(task_info); // replay must not reuse scalar DCache task fields
    __ubuf__ CombineRecvUB<K,HC>* ub=0;
    auto stage=ub->control_stage;
    auto words=reinterpret_cast<__ubuf__ uint32_t*>(stage);
    auto ctx=reinterpret_cast<__gm__ const uint8_t*>(context);
    auto controls=static_cast<__gm__ uint8_t*>(control_buffer);
    auto error=static_cast<__gm__ uint32_t*>(workspace);
    auto core_error=reinterpret_cast<__gm__ uint32_t*>(static_cast<__gm__ uint8_t*>(workspace)+256+32*vec_core_idx);
    auto lanes=reinterpret_cast<__gm__ const AFDLane*>(ctx+context->lanes_offset);
    auto peers=reinterpret_cast<__gm__ const AFDPeer*>(ctx+context->peers_offset);
    auto addresses=reinterpret_cast<__gm__ const AFDAddress*>(ctx+context->addresses_offset);
    uint32_t lane_idx=context->num_source_lanes;
    for(uint32_t l=0;l<context->num_source_lanes;++l)
        if(lanes[l].src_rank_idx==task.src_rank_idx && lanes[l].payload_slot_idx==task.payload_slot_idx) lane_idx=l;
    if(vec_core_idx==0) {
        uint32_t code=OK;
        if(context->abi_version!=kAbiVersion || task.abi_version!=kAbiVersion || !context->reserved_address) code=BAD_ABI;
        else if(lane_idx==context->num_source_lanes || task.src_rank_idx!=context->src_rank_idx ||
            task.num_tokens>context->num_token_capacity || !task.num_comm_rounds) code=BAD_TASK;
        else {
            auto layers=reinterpret_cast<__gm__ const uint32_t*>(ctx+context->layers_offset);
            bool valid_layer=false;
            for(uint32_t l=0;l<context->num_layers;++l)valid_layer|=layers[l]==task.layer_idx;
            if(!valid_layer)code=BAD_LAYER;
        }
        AscendC::WriteGmByPassDCache(error,code);
        for(int i=0;i<8;++i)words[i]=0;
        words[0]=1;store_control(recv_status,stage);
    }
    local_join();
    if(AscendC::ReadGmByPassDCache(error)!=OK) {
        if(vec_core_idx==0){words[0]=3;words[1]=AscendC::ReadGmByPassDCache(error);store_control(recv_status,stage);}return;
    }
    const auto lane=lanes[lane_idx];
    auto directory=reinterpret_cast<__gm__ const AFDReturnDirectory*>(ctx+context->reserved_address);
    auto return_lanes=reinterpret_cast<__gm__ const AFDReturnLane*>(ctx+directory->lanes_offset);
    const auto return_lane=return_lanes[lane_idx];
    if(vec_core_idx==0) {
        if(directory->version!=1 || directory->return_stride<uint64_t(2)*H ||
            !lane.capacity || lane.capacity>kMaxLaneTokens ||
            task.num_comm_rounds<(uint64_t(task.num_tokens)+lane.capacity-1)/lane.capacity ||
            return_lane.return_offset>context->payload_bytes ||
            uint64_t(lane.capacity)*K*directory->return_stride>context->payload_bytes-return_lane.return_offset)
            AscendC::WriteGmByPassDCache(error,uint32_t(CAPACITY));
    }
    local_join();
    uint32_t current_round=0,last_input_bit=0;
    for(uint32_t round_idx=0;round_idx<task.num_comm_rounds;++round_idx) {
        current_round=round_idx;
        if(AscendC::ReadGmByPassDCache(error)!=OK)break;
        auto ledger=round_ledger+uint64_t(lane.payload_slot_idx)*2+round_idx%2;
        if(vec_core_idx==0 && !wait_control<Timeout>(stage,ledger,1)) AscendC::WriteGmByPassDCache(error,uint32_t(TIMEOUT));
        local_join();
        if(AscendC::ReadGmByPassDCache(error)!=OK)break;
        asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(&ub->ledger),reinterpret_cast<__gm__ uint8_t*>(ledger),
            1,64,0,0,true,asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,0,0);
        asc_sync_notify(PIPE_MTE2,PIPE_S,EVENT_ID3);asc_sync_wait(PIPE_MTE2,PIPE_S,EVENT_ID3);
        const auto saved=ub->ledger;
        last_input_bit=saved.input_bit;
        const uint32_t token_begin=uint32_t(math::min(uint64_t(task.num_tokens),uint64_t(round_idx)*lane.capacity));
        const uint32_t num_tokens=math::min(lane.capacity,task.num_tokens-token_begin);
        uint32_t code=OK;
        if(saved.round_idx!=round_idx || saved.microbatch_idx!=task.microbatch_idx ||
            saved.token_begin!=token_begin || saved.num_round_tokens!=num_tokens || saved.input_bit>1 || saved.return_bit>1 ||
            saved.final_round!=uint32_t(round_idx+1==task.num_comm_rounds))code=STATE_ORDER;
        // Preflight is parallel, bounded, and completes before any output or credit.
        const uint32_t num_tokens_per_core=(num_tokens+NC-1)/NC;
        const uint32_t core_begin=math::min(num_tokens,uint32_t(vec_core_idx)*num_tokens_per_core);
        const uint32_t core_end=math::min(num_tokens,core_begin+num_tokens_per_core);
        uint32_t mask_begin=UINT32_MAX;
        auto is_live=[&](uint32_t t) __aicore__ {
            if(!token_mask)return true;
            if(mask_begin==UINT32_MAX || t-mask_begin>=32) {
                mask_begin=t;
                // Exact byte count, including a short final tile; no allocation-tail overread.
                recv_load(ub->status_zero_stage,token_mask+uint64_t(token_begin)+t,math::min(uint32_t(32),core_end-t));
            }
            return ub->status_zero_stage[t-mask_begin]!=0;
        };
        for(uint32_t t=core_begin;t<core_end && code==OK;++t) {
            const uint32_t token_idx=token_begin+t;
            if(!is_live(t))continue;
            asc_copy_gm2ub_align(ub->expert_idx,combined_topk_idx+uint64_t(token_idx)*K,1,8*K,0,0,true,
                asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,0,0);
            asc_sync_notify(PIPE_MTE2,PIPE_S,EVENT_ID3);asc_sync_wait(PIPE_MTE2,PIPE_S,EVENT_ID3);
            for(int k=0;k<K;++k) {
                const auto expert_idx=ub->expert_idx[k];
                if(expert_idx<0 || expert_idx>=E)code=BAD_ROUTE;
                for(int j=0;j<k;++j) if(ub->expert_idx[j]==expert_idx)code=BAD_ROUTE;
            }
        }
        AscendC::WriteGmByPassDCache(core_error,code);local_join();
        if(vec_core_idx==0) {
            for(int i=0;i<NC;++i) {
                auto p=reinterpret_cast<__gm__ uint32_t*>(static_cast<__gm__ uint8_t*>(workspace)+256+32*i);
                const uint32_t c=AscendC::ReadGmByPassDCache(p);if(c)AscendC::WriteGmByPassDCache(error,c);
            }
            if(AscendC::ReadGmByPassDCache(error)==OK)
                for(int f=0;f<N;++f)if(!wait_control<Timeout>(stage,controls+return_lane.done_offset+64*f,saved.return_bit)) {
                    AscendC::WriteGmByPassDCache(error,uint32_t(TIMEOUT));break;
                }
        }
        local_join();if(AscendC::ReadGmByPassDCache(error)!=OK)break;
        mask_begin=UINT32_MAX;
        for(uint32_t t=core_begin;t<core_end;++t) {
            const uint32_t token_idx=token_begin+t;
            const bool live=is_live(t);
            if(live) {
                asc_copy_gm2ub_align(ub->weights,topk_weights+uint64_t(token_idx)*K,1,4*K,0,0,true,
                    asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,0,0);
                asc_sync_notify(PIPE_MTE2,PIPE_S,EVENT_ID3);asc_sync_wait(PIPE_MTE2,PIPE_S,EVENT_ID3);
            }
            for(uint32_t offset=0;offset<H;offset+=HC) {
                const uint32_t count=math::min(uint32_t(HC),uint32_t(H)-offset);
                auto acc=combine_local(ub->acc_fp32,HC),value=combine_local(ub->value_fp32,HC);
                auto output=combine_local(ub->output_bf16,HC);
                AscendC::Duplicate(acc,0.0f,count);AscendC::PipeBarrier<PIPE_V>();
                if(live) {
                    // Two independent stages: prefetch k+1 while V consumes k.
                    for(int k=0;k<math::min(K,2);++k) {
                        auto input=reinterpret_cast<__gm__ bfloat16_t*>(buffer+return_lane.return_offset+
                            (uint64_t(t)*K+k)*directory->return_stride)+offset;
                        asc_copy_gm2ub_align(ub->input_stage[k],input,1,2*count,0,0,true,
                            asc_load_l2_cache_mode::NOTALLOC_KEEP,0,0);
                        asc_sync_notify(PIPE_MTE2,PIPE_V,static_cast<event_t>(k));
                    }
                    for(int k=0;k<K;++k) {
                        const auto event=static_cast<event_t>(k%2);
                        asc_sync_wait(PIPE_MTE2,PIPE_V,event);
                        AscendC::Cast(value,combine_local(ub->input_stage[k%2],HC),AscendC::RoundMode::CAST_NONE,count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(value,value,ub->weights[k],count);AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(acc,acc,value,count);AscendC::PipeBarrier<PIPE_V>();
                        if(k+2<K) {
                            asc_sync_notify(PIPE_V,PIPE_MTE2,event);asc_sync_wait(PIPE_V,PIPE_MTE2,event);
                            auto input=reinterpret_cast<__gm__ bfloat16_t*>(buffer+return_lane.return_offset+
                                (uint64_t(t)*K+k+2)*directory->return_stride)+offset;
                            asc_copy_gm2ub_align(ub->input_stage[k%2],input,1,2*count,0,0,true,
                                asc_load_l2_cache_mode::NOTALLOC_KEEP,0,0);
                            asc_sync_notify(PIPE_MTE2,PIPE_V,event);
                        }
                    }
                    // Both input buffers have finished V reads before the next chunk can refill them.
                    asc_sync_notify(PIPE_V,PIPE_MTE2,EVENT_ID2);asc_sync_wait(PIPE_V,PIPE_MTE2,EVENT_ID2);
                }
                AscendC::Cast(output,acc,AscendC::RoundMode::CAST_RINT,count);
                asc_sync_notify(PIPE_V,PIPE_MTE3,EVENT_ID2);asc_sync_wait(PIPE_V,PIPE_MTE3,EVENT_ID2);
                asc_copy_ub2gm_align(combined_x+uint64_t(token_idx)*H+offset,ub->output_bf16,1,2*count,
                    asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,0,0);
                asc_sync_notify(PIPE_MTE3,PIPE_V,EVENT_ID2);asc_sync_wait(PIPE_MTE3,PIPE_V,EVENT_ID2);
            }
        }
        local_join();
        if(vec_core_idx==0) {
            for(int i=0;i<8;++i)words[i]=0;
            words[0]=saved.return_bit^1;
            for(int f=0;f<N;++f) store_control(reinterpret_cast<__gm__ void*>(peers[f].control_base+return_lane.credit_offset),stage);
            // SPSC bank is released only after all credit stores completed, never by send_status.
            for(int i=0;i<8;++i)words[i]=0;store_control(ledger,stage);
        }
        local_join();
    }
    if(vec_core_idx==0) {
        if(AscendC::ReadGmByPassDCache(error)==OK)
            for(int f=0;f<N;++f)if(!wait_control<Timeout>(stage,controls+addresses[lane_idx*N+f].consumed_offset,last_input_bit)) {
                AscendC::WriteGmByPassDCache(error,uint32_t(TIMEOUT));break;
            }
        for(int i=0;i<8;++i)words[i]=0;
        words[0]=AscendC::ReadGmByPassDCache(error)==OK?2:3;
        words[1]=AscendC::ReadGmByPassDCache(error);words[2]=current_round;store_control(recv_status,stage);
    }
}
} // namespace deep_ep::afd
// ### PATCH END: AFD weighted per-route epilogue
