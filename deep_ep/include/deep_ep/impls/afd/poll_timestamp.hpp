// SPDX-License-Identifier: MIT
#pragma once

#include <deep_ep/layout/afd/poll_timestamp.hpp>

namespace deep_ep::afd {
// Call after the matching Ready load completes. A repeated poll of the same
// publication must not overwrite the timestamp. The preceding adoption tick
// still identifies the publication when a prior round is retired early or the
// scheduler later clears its Pending entry.
__aicore__ inline void capture_ffn_poll_timestamp(__gm__ AFDPollTimestamp *destination,
    __gm__ const uint8_t *control, const AFDAddress &address, uint64_t control_bytes,
    uint32_t input_bit, uint64_t previous_adopt_tick, __ubuf__ uint8_t *scratch)
{
    if (AscendC::ReadGmByPassDCache(&destination->valid) == 1 &&
        AscendC::ReadGmByPassDCache(&destination->input_bit) == input_bit) {
        // Read only the first four words for the deduplication check.
        auto timestamp_words = reinterpret_cast<__gm__ uint32_t *>(destination);
        const uint64_t previous_tick =
            uint64_t(AscendC::ReadGmByPassDCache(timestamp_words + 2)) |
            (uint64_t(AscendC::ReadGmByPassDCache(timestamp_words + 3)) << 32);
        if (previous_tick == previous_adopt_tick) {
            return;
        }
    }

    if (!recv_range(address.inbox_offset, sizeof(AFDInbox), control_bytes)) {
        return;
    }
    const uint64_t cycle = static_cast<uint64_t>(asc_get_system_cycle());
    recv_load(scratch, control + address.inbox_offset, sizeof(AFDInbox));
    const auto mailbox = recv_read_inbox_ubuf(scratch);
    auto words = reinterpret_cast<__ubuf__ uint32_t *>(scratch);
    words[0] = 1;
    words[1] = input_bit;
    words[2] = uint32_t(previous_adopt_tick);
    words[3] = uint32_t(previous_adopt_tick >> 32);
    words[4] = uint32_t(cycle);
    words[5] = uint32_t(cycle >> 32);
    words[6] = mailbox.microbatch_idx;
    words[7] = mailbox.layer_idx;
    words[8] = mailbox.round_idx;
    words[9] = mailbox.num_comm_rounds;
    words[10] = mailbox.num_round_tokens;
    words[11] = mailbox.return_bit;
    words[12] = mailbox.num_recv_records;
    words[13] = mailbox.reserved[0];
    words[14] = words[15] = 0;
    recv_store(destination, scratch, kPollTimestampBytes);
}
} // namespace deep_ep::afd
