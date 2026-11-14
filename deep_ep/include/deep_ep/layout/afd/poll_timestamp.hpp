// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace deep_ep::afd {
// Caller-owned, zero-initialized sidecar. A lane's slot describes its most
// recent NEW_READY observation, even after the lane becomes Pending or idle.
struct alignas(32) AFDPollTimestamp {
    uint32_t valid;
    uint32_t input_bit;
    uint64_t previous_adopt_tick;
    uint64_t system_cycle;
    uint32_t microbatch_idx;
    uint32_t layer_idx;
    uint32_t round_idx;
    uint32_t num_comm_rounds;
    uint32_t num_round_tokens;
    uint32_t return_bit;
    uint32_t num_recv_records;
    uint32_t round_mode;
    uint32_t reserved[2];
};

constexpr uint32_t kPollTimestampBytes = sizeof(AFDPollTimestamp);
static_assert(kPollTimestampBytes == 64);
} // namespace deep_ep::afd
