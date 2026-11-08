#pragma once
// T020 diag20 recorder (native EP dispatch/combine, exec/T020-send-coarse).
//
// One diagnostic build (D) serves both modes: mode 1 records, mode 0 is the
// same kernel with every recorder entry inactive. P is the frozen production
// build and never contains this header.
//
// Arena contract (host allocates, device writes):
//   byte 0                    128-byte arena header (host-owned)
//                             device writes only the out-of-range counter at +0x70
//   byte 128                  ring header/record region, one 4160-byte ring per
//                             (call_id, core, writer)
//   ring index                (call_id * num_cores + core) * 3 + writer
//
// Ring header (64 B):
//   +0   magic | version << 32
//   +8   op | core << 32
//   +16  writer | state << 32            state: 1 active, 2 complete
//   +24  capacity | written << 32
//   +32  dropped | call_id << 32
//   +40  ring byte offset from the arena base
//   +48  first timestamp (this writer, this call)
//   +56  last timestamp  (this writer, this call)
//
// Record (64 B):
//   +0   timestamp (scalar system counter, or SIMT clock64; domains never mix)
//   +8   aux0
//   +16  aux1
//   +24  aux2
//   +32  aux3
//   +40  point_id | kind << 16 | domain << 24 | seq << 32
//   +48  call_id
//   +56  core | writer << 32
//
// Call ids are never wrapped: a call_id >= kMaxCalls increments the arena
// out-of-range counter and records nothing, so the parser fails instead of
// silently reading overwritten rings.

#include <kernel_operator.h>
#include <simt_api/device_functions.h>
#include <utils/debug/asc_time.h>

namespace deep_ep::diag20 {

static constexpr uint64_t kMagic = 0x544F323044494147ULL;  // "T020DIAG"
static constexpr uint32_t kVersion = 2;

static constexpr uint32_t kArenaHeaderBytes = 128;
static constexpr uint32_t kArenaOobOffset = 0x70;

static constexpr uint32_t kMaxCalls = 256;
static constexpr uint32_t kRecordsPerWriter = 64;
static constexpr uint32_t kWritersPerCore = 3;
static constexpr uint32_t kRingHeaderBytes = 64;
static constexpr uint32_t kRecordBytes = 64;
static constexpr uint64_t kRingStride = uint64_t(kRingHeaderBytes) + uint64_t(kRecordsPerWriter) * kRecordBytes;

static constexpr uint32_t kWriterScalar = 0;
static constexpr uint32_t kWriterSimt = 1;
// CS issues each task through its own VF call, so the first and the last task
// envelope need separate rings: a re-init inside the last VF call would wrap
// onto the first call's records.
static constexpr uint32_t kWriterSimtLast = 2;

static constexpr uint32_t kOpDispatch = 1;
static constexpr uint32_t kOpCombine = 2;

static constexpr uint32_t kKindPoint = 0;
static constexpr uint32_t kKindExit = 1;
static constexpr uint32_t kDomainScalar = 0;
static constexpr uint32_t kDomainSimt = 1;

static constexpr uint32_t kStateActive = 1;
static constexpr uint32_t kStateComplete = 2;

static constexpr uint32_t kPointExit = 0x0001;
static constexpr uint32_t kPointExport = 0x0002;

__aicore__ __forceinline__ uint64_t scalar_clock() {
    return static_cast<uint64_t>(asc_get_system_cycle());
}

__aicore__ __forceinline__ void scalar_store(__gm__ uint8_t* ptr, uint64_t value) {
    AscendC::WriteGmByPassDCache(reinterpret_cast<__gm__ uint64_t*>(ptr), value);
}

struct ScalarState {
    __gm__ uint8_t* ring;
    __gm__ uint8_t* recs;
    uint64_t ring_offset;
    uint64_t first_ts;
    uint64_t last_ts;
    uint32_t call_id;
    uint32_t core;
    uint32_t op;
    uint32_t writer;
    uint32_t seq;
    uint32_t dropped;
    uint32_t active;

    __aicore__ __forceinline__ void store_header(uint32_t state) const {
        scalar_store(ring + 0, kMagic | (uint64_t(kVersion) << 32));
        scalar_store(ring + 8, uint64_t(op) | (uint64_t(core) << 32));
        scalar_store(ring + 16, uint64_t(writer) | (uint64_t(state) << 32));
        scalar_store(ring + 24, uint64_t(kRecordsPerWriter) | (uint64_t(seq) << 32));
        scalar_store(ring + 32, uint64_t(dropped) | (uint64_t(call_id) << 32));
        scalar_store(ring + 40, ring_offset);
        scalar_store(ring + 48, first_ts);
        scalar_store(ring + 56, last_ts);
    }

    __aicore__ __forceinline__ void emit(uint32_t kind, uint32_t point_id,
                                         uint64_t aux0 = 0, uint64_t aux1 = 0,
                                         uint64_t aux2 = 0, uint64_t aux3 = 0) {
        if (not active)
            return;
        if (seq >= kRecordsPerWriter) {
            ++dropped;
            return;
        }
        const uint64_t ts = scalar_clock();
        if (seq == 0)
            first_ts = ts;
        last_ts = ts;
        __gm__ uint8_t* r = recs + uint64_t(seq) * kRecordBytes;
        scalar_store(r + 0, ts);
        scalar_store(r + 8, aux0);
        scalar_store(r + 16, aux1);
        scalar_store(r + 24, aux2);
        scalar_store(r + 32, aux3);
        scalar_store(r + 40, uint64_t(point_id) | (uint64_t(kind) << 16) |
                             (uint64_t(kDomainScalar) << 24) | (uint64_t(seq) << 32));
        scalar_store(r + 48, uint64_t(call_id));
        scalar_store(r + 56, uint64_t(core) | (uint64_t(writer) << 32));
        ++seq;
    }

    __aicore__ __forceinline__ void point(uint32_t point_id,
                                          uint64_t aux0 = 0, uint64_t aux1 = 0,
                                          uint64_t aux2 = 0, uint64_t aux3 = 0) {
        emit(kKindPoint, point_id, aux0, aux1, aux2, aux3);
    }

    // Business exit, one DDR DSB, export marker, ring commit. The DSB is the
    // only added fence and it is issued once per writer per call.
    __aicore__ __forceinline__ void finish() {
        if (not active)
            return;
        emit(kKindExit, kPointExit, uint64_t(seq), uint64_t(dropped), 0, 0);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        emit(kKindPoint, kPointExport, uint64_t(seq), uint64_t(dropped), 0, 0);
        store_header(kStateComplete);
        active = 0;
    }
};

__aicore__ __forceinline__ ScalarState scalar_init(__gm__ uint8_t* arena, int mode, uint32_t call_id,
                                                   uint32_t op, uint32_t writer, int num_cores) {
    ScalarState s;
    s.ring = nullptr;
    s.recs = nullptr;
    s.ring_offset = 0;
    s.first_ts = 0;
    s.last_ts = 0;
    s.call_id = call_id;
    s.core = 0;
    s.op = op;
    s.writer = writer;
    s.seq = 0;
    s.dropped = 0;
    s.active = 0;
    if (arena == nullptr or mode != 1)
        return s;
    const auto core = static_cast<uint32_t>(AscendC::GetBlockIdx());
    s.core = core;
    if (static_cast<int>(core) >= num_cores)
        return s;
    if (call_id >= kMaxCalls) {
        // Never wrap onto an older call's ring; flag it for the parser instead.
        // A plain flag store keeps the scalar path free of SIMT-only atomics.
        scalar_store(arena + kArenaOobOffset, 1);
        return s;
    }
    const uint64_t idx =
        (uint64_t(call_id) * uint64_t(num_cores) + uint64_t(core)) * kWritersPerCore + uint64_t(writer);
    s.ring = arena + kArenaHeaderBytes + idx * kRingStride;
    s.recs = s.ring + kRingHeaderBytes;
    s.ring_offset = static_cast<uint64_t>(s.ring - arena);
    s.active = 1;
    s.store_header(kStateActive);
    return s;
}

// ---------------------------------------------------------------------------
// SIMT writer. Only the representative thread (thread_idx == 0) constructs an
// active state; every other thread's state stays inactive. Timestamps come
// from the SIMT clock64() domain and are never subtracted from scalar cycles.

struct SimtState {
    __gm__ uint8_t* ring;
    __gm__ uint8_t* recs;
    uint64_t ring_offset;
    uint64_t first_ts;
    uint64_t last_ts;
    uint32_t call_id;
    uint32_t core;
    uint32_t op;
    uint32_t writer;
    uint32_t seq;
    uint32_t dropped;
    uint32_t active;
};

__aicore__ __simt_callee__ __forceinline__ void simt_store(__gm__ uint8_t* ptr, uint64_t value) {
    asc_stcg(reinterpret_cast<__gm__ uint64_t*>(ptr), value);
}

__aicore__ __simt_callee__ __forceinline__ void simt_store_header(SimtState* s, uint32_t state) {
    simt_store(s->ring + 0, kMagic | (uint64_t(kVersion) << 32));
    simt_store(s->ring + 8, uint64_t(s->op) | (uint64_t(s->core) << 32));
    simt_store(s->ring + 16, uint64_t(s->writer) | (uint64_t(state) << 32));
    simt_store(s->ring + 24, uint64_t(kRecordsPerWriter) | (uint64_t(s->seq) << 32));
    simt_store(s->ring + 32, uint64_t(s->dropped) | (uint64_t(s->call_id) << 32));
    simt_store(s->ring + 40, s->ring_offset);
    simt_store(s->ring + 48, s->first_ts);
    simt_store(s->ring + 56, s->last_ts);
}

__aicore__ __simt_callee__ __forceinline__ void simt_init(SimtState* s, __gm__ uint8_t* arena, int mode,
                                                          uint32_t call_id, uint32_t op, uint32_t core,
                                                          int num_cores, uint32_t writer = kWriterSimt) {
    s->ring = nullptr;
    s->recs = nullptr;
    s->ring_offset = 0;
    s->first_ts = 0;
    s->last_ts = 0;
    s->call_id = call_id;
    s->core = core;
    s->op = op;
    s->writer = writer;
    s->seq = 0;
    s->dropped = 0;
    s->active = 0;
    if (arena == nullptr or mode != 1)
        return;
    if (static_cast<int>(core) >= num_cores)
        return;
    if (call_id >= kMaxCalls) {
        simt_store(arena + kArenaOobOffset, 1);
        return;
    }
    const uint64_t idx =
        (uint64_t(call_id) * uint64_t(num_cores) + uint64_t(core)) * kWritersPerCore + uint64_t(writer);
    s->ring = arena + kArenaHeaderBytes + idx * kRingStride;
    s->recs = s->ring + kRingHeaderBytes;
    s->ring_offset = static_cast<uint64_t>(s->ring - arena);
    s->active = 1;
    simt_store_header(s, kStateActive);
}

__aicore__ __simt_callee__ __forceinline__ void simt_emit(SimtState* s, uint32_t kind, uint32_t point_id,
                                                          uint64_t aux0 = 0, uint64_t aux1 = 0,
                                                          uint64_t aux2 = 0, uint64_t aux3 = 0) {
    if (not s->active)
        return;
    if (s->seq >= kRecordsPerWriter) {
        ++s->dropped;
        return;
    }
    const uint64_t ts = static_cast<uint64_t>(clock64());
    if (s->seq == 0)
        s->first_ts = ts;
    s->last_ts = ts;
    __gm__ uint8_t* r = s->recs + uint64_t(s->seq) * kRecordBytes;
    simt_store(r + 0, ts);
    simt_store(r + 8, aux0);
    simt_store(r + 16, aux1);
    simt_store(r + 24, aux2);
    simt_store(r + 32, aux3);
    simt_store(r + 40, uint64_t(point_id) | (uint64_t(kind) << 16) |
                          (uint64_t(kDomainSimt) << 24) | (uint64_t(s->seq) << 32));
    simt_store(r + 48, uint64_t(s->call_id));
    simt_store(r + 56, uint64_t(s->core) | (uint64_t(s->writer) << 32));
    ++s->seq;
}

__aicore__ __simt_callee__ __forceinline__ void simt_point(SimtState* s, uint32_t point_id,
                                                           uint64_t aux0 = 0, uint64_t aux1 = 0,
                                                           uint64_t aux2 = 0, uint64_t aux3 = 0) {
    simt_emit(s, kKindPoint, point_id, aux0, aux1, aux2, aux3);
}

__aicore__ __simt_callee__ __forceinline__ void simt_finish(SimtState* s) {
    if (not s->active)
        return;
    simt_emit(s, kKindExit, kPointExit, uint64_t(s->seq), uint64_t(s->dropped), 0, 0);
    // SIMT has no scalar DSB; order this thread's stores before the commit.
    asc_threadfence();
    simt_emit(s, kKindPoint, kPointExport, uint64_t(s->seq), uint64_t(s->dropped), 0, 0);
    asc_threadfence();
    simt_store_header(s, kStateComplete);
    s->active = 0;
}

}  // namespace deep_ep::diag20
