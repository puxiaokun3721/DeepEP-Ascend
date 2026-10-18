#pragma once
#include "active.hpp"

// ### PATCH START: AFD reverse-channel directory and local completion ABI
namespace deep_ep::afd {
// Optional Context extension at reserved_address (an offset, not a device pointer).
// Existing dispatch ABI/offsets stay unchanged. Reverse-channel users require version=1.
struct alignas(32) AFDReturnDirectory {
    uint64_t source_peers_offset, lanes_offset, return_stride;
    uint32_t version, reserved;
};
struct alignas(32) AFDReturnLane {
    uint64_t return_offset, done_offset, credit_offset, reserved;
};
enum CompletionState : uint32_t { COMPLETION_EMPTY=0, ARMED=1, READY=2 };
struct alignas(32) AFDCompletion {
    uint32_t state, error_code, num_entries, num_members;
    uint32_t num_deltas, reserved[3];
};
constexpr uint64_t completion_entries_offset() { return 32; }
constexpr uint64_t completion_members_offset(uint32_t l) { return 32+32*uint64_t(l); }
constexpr uint64_t completion_deltas_offset(uint32_t l) { return 32+64*uint64_t(l); }
constexpr uint64_t completion_bytes(uint32_t l,uint32_t r) { return completion_deltas_offset(l)+align32(4*uint64_t(l)*r); }
static_assert(sizeof(AFDReturnDirectory)==32 && sizeof(AFDReturnLane)==32 && sizeof(AFDCompletion)==32);
}
// ### PATCH END: AFD reverse-channel directory and local completion ABI
