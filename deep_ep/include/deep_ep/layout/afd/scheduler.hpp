#pragma once
#include "return.hpp"

// ### PATCH START: private bounded metadata scheduler state (no native EP equivalent)
namespace deep_ep::afd {
struct alignas(32) AFDSchedulerState {
    uint64_t tick;
    uint32_t last_phase, has_phase, reserved[4];
};
struct alignas(32) AFDPendingLane {
    uint32_t valid, expected_input_bit, grant_state, member_leader;
    uint64_t first_seen;
    // Private per-invocation scratch in former reserved bytes; rebuilt on replay.
    uint32_t num_group_members, reserved;
    AFDInbox inbox;
};
constexpr uint64_t scheduler_pending_offset() { return 32; }
constexpr uint64_t scheduler_counts_offset(uint32_t l) { return 32+96*uint64_t(l); }
constexpr uint64_t scheduler_consumed_offset(uint32_t l,uint32_t r) { return scheduler_counts_offset(l)+align32(4*uint64_t(l)*r); }
constexpr uint64_t scheduler_workspace_bytes(uint32_t l,uint32_t r) { return scheduler_consumed_offset(l,r)+align32(4*uint64_t(l)*r); }
constexpr uint64_t scheduler_ub_bytes(uint32_t r) { return 256+5*align32(4*uint64_t(r)); }
static_assert(offsetof(AFDPendingLane,inbox)==32);
static_assert(sizeof(AFDSchedulerState)==32 && sizeof(AFDPendingLane)==96);
}
// ### PATCH END: private bounded metadata scheduler state
