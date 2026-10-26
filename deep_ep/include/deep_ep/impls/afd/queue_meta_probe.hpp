#pragma once
// ### PATCH START: AFD queue-metadata read-only probe (review candidate v4)
// Scope: the ChannelHandle table produced by acquire_jetties
// (HcommChannelCreateWithConfig with the COMM_PROTOCOL_UBC_CTP predicate:
// the production Jetty queue view for the AFDBuffer payload/control
// contexts). It is NOT the temporary UB_MEM mem-discovery channel and NOT
// HCCL-internal collective channels. The entity's engine/protocol fields are
// reported exactly as read; no channel-family label is assumed here.
// Reads only typed scalar fields of the SDK SqContext/CqContext plus the
// entity counts; never dereferences sq/head/tail/doorbell addresses, never
// writes SQ/CQ/peer state, and never emits addresses, EIDs, tokens, IDs or
// handles. Output layout per table block (uint32 words):
//   [0] valid (1 = fields below copied)
//   [1] engine, [2] protocol (as read from ChannelEntity)
//   [3] sqNum, [4] cqNum
//   [5] ubJfs.wqeSize, [6] ubJfs.sqDepth
//   [7] ubJfc.cqeSize, [8] ubJfc.cqDepth
//   [9] HCOMM_URMA_DEFAULT_QP_IDX used (0)
//   [10] error (0=none, 1=null/misaligned entity, 2=QP/count,
//               3=null/misaligned context, 4=wrong SQ/CQ type)
//   [11] SQ type, [12] CQ type; [13..15] reserved, zero
// Include chain: the production device header deep_ep/comm/handle.hpp brings
// the SDK device definitions consumed by the existing AFD kernels
// (<hcomm/hcomm.h> provides AscendC::ChannelEntity/SqContext/CqContext and
// HCOMM_URMA_DEFAULT_QP_IDX; the c_api headers provide ReadGmByPassDCache).
// The assertions pin the invariants this probe relies on and are delivered
// with the diff for review.
#include <deep_ep/comm/handle.hpp>

namespace deep_ep::afd {

static_assert(sizeof(decltype(AscendC::ChannelEntity::sqContextAddr)) == sizeof(uint64_t) and
              sizeof(decltype(AscendC::ChannelEntity::cqContextAddr)) == sizeof(uint64_t),
              "context pointer width must match the aligned pointer-word load");
static_assert(sizeof(AscendC::ChannelEntity) == 256,
              "ChannelEntity must match the SDK device-side 256-byte layout");
static_assert(offsetof(AscendC::ChannelEntity, sqContextAddr) % 8 == 0 and
              offsetof(AscendC::ChannelEntity, cqContextAddr) % 8 == 0,
              "sq/cq context pointers must be 8-byte aligned words in the entity");
static_assert(sizeof(AscendC::SqContext) == 128 and sizeof(AscendC::CqContext) == 128,
              "SQ/CQ context entries must be 128 bytes, matching the transport slab copies");
static_assert(std::is_same<decltype(((AscendC::SqContext*)nullptr)->contextInfo.ubJfs.sqDepth), uint32_t>::value,
              "ubJfs.sqDepth must be uint32_t");
static_assert(std::is_same<decltype(((AscendC::SqContext*)nullptr)->contextInfo.ubJfs.wqeSize), uint32_t>::value,
              "ubJfs.wqeSize must be uint32_t");
static_assert(std::is_same<decltype(((AscendC::CqContext*)nullptr)->contextInfo.ubJfc.cqDepth), uint32_t>::value,
              "ubJfc.cqDepth must be uint32_t");
static_assert(std::is_same<decltype(((AscendC::CqContext*)nullptr)->contextInfo.ubJfc.cqeSize), uint32_t>::value,
              "ubJfc.cqeSize must be uint32_t");

__aicore__ inline void queue_meta_probe_block(__gm__ uint64_t* jetties,
                                              __gm__ uint32_t* out) {
    #pragma unroll
    for (int i = 0; i < 16; ++i)
        out[i] = 0;
    const auto jetty_value = AscendC::ReadGmByPassDCache(jetties);
    if (jetty_value == 0 or jetty_value % alignof(AscendC::ChannelEntity) != 0) {
        out[10] = 1;
        return;
    }
    const auto entity = reinterpret_cast<__gm__ const AscendC::ChannelEntity*>(jetty_value);
    const auto sq_num = entity->sqNum;
    const auto cq_num = entity->cqNum;
    constexpr auto qp_idx = AscendC::HCOMM_URMA_DEFAULT_QP_IDX;
    if (qp_idx >= sq_num or qp_idx >= cq_num) {
        out[10] = 2;
        return;
    }
    // Only aligned pointer-width words; counts/types/union members stay typed.
    const auto entity_words = reinterpret_cast<__gm__ const uint64_t*>(entity);
    const auto sq_addr = entity_words[offsetof(AscendC::ChannelEntity, sqContextAddr) / sizeof(uint64_t)];
    const auto cq_addr = entity_words[offsetof(AscendC::ChannelEntity, cqContextAddr) / sizeof(uint64_t)];
    if (sq_addr == 0 or cq_addr == 0 or
        sq_addr % alignof(AscendC::SqContext) != 0 or
        cq_addr % alignof(AscendC::CqContext) != 0) {
        out[10] = 3;
        return;
    }
    const auto sq_context = reinterpret_cast<__gm__ const AscendC::SqContext*>(sq_addr) + qp_idx;
    const auto cq_context = reinterpret_cast<__gm__ const AscendC::CqContext*>(cq_addr) + qp_idx;
    out[11] = static_cast<uint32_t>(sq_context->type);
    out[12] = static_cast<uint32_t>(cq_context->type);
    if (sq_context->type != AscendC::SQ_CONTEXT_TYPE_UB_JFS or
        cq_context->type != AscendC::CQ_CONTEXT_TYPE_UB_JFC) {
        out[10] = 4;
        return;
    }
    out[1] = static_cast<uint32_t>(entity->engine);
    out[2] = static_cast<uint32_t>(entity->protocol);
    out[3] = sq_num;
    out[4] = cq_num;
    out[5] = sq_context->contextInfo.ubJfs.wqeSize;
    out[6] = sq_context->contextInfo.ubJfs.sqDepth;
    out[7] = cq_context->contextInfo.ubJfc.cqeSize;
    out[8] = cq_context->contextInfo.ubJfc.cqDepth;
    out[9] = AscendC::HCOMM_URMA_DEFAULT_QP_IDX;
    out[0] = 1;
}

__global__ __vector__ void queue_meta_probe_impl(__gm__ uint64_t* payload_jetties,
                                                 __gm__ uint64_t* control_jetties,
                                                 __gm__ uint32_t* out) {
    queue_meta_probe_block(payload_jetties, out);
    queue_meta_probe_block(control_jetties, out + 16);
}

}
// ### PATCH END: AFD queue-metadata read-only probe
