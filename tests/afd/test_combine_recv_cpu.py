"""CPU semantic oracle for combineRecv; not a CANN compilation or device test."""
import unittest

import numpy as np


def bf16_rn(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    # Keep NaNs as NaNs, including payloads which would round to infinity.
    rounded = np.where(np.isnan(values), bits | np.uint32(0x00400000), rounded).astype(np.uint32)
    return rounded.view(np.float32)


def reduce_routes(payload, weights, mask):
    rows, topk, hidden = payload.shape
    result = np.zeros((rows, hidden), dtype=np.float32)
    for token in range(rows):
        if not mask[token]:
            continue
        for slot_idx in range(topk):
            value = np.multiply(payload[token, slot_idx], weights[token, slot_idx], dtype=np.float32)
            result[token] = np.add(result[token], value, dtype=np.float32)
    return bf16_rn(result)


class RoundReceiver:
    """All-F closure oracle; credit stores precede ledger release and final ACK."""
    def __init__(self, num_ranks):
        self.num_ranks = num_ranks
        self.trace = []
        self.final_input_bit = None

    def consume(self, bit, done, input_bit, final, publish_credit):
        if done != [bit] * self.num_ranks:
            return False
        self.trace.append('read_payload')
        for rank in range(self.num_ranks):
            publish_credit(rank, bit ^ 1)
            self.trace.append(('credit_complete', rank, bit ^ 1))
        self.trace.append('ledger_empty')
        if final:
            self.final_input_bit = input_bit
        return True

    def complete(self, consumed):
        return self.final_input_bit is not None and consumed == [self.final_input_bit] * self.num_ranks


class CombineRecvTests(unittest.TestCase):
    def test_weight_once_for_every_route_same_rank_and_bf16_final(self):
        for topk in (1, 2, 8, 32):
            payload = np.ones((1, topk, 128), dtype=np.float32) * 2
            weights = np.full((1, topk), .25, dtype=np.float32)
            result = reduce_routes(payload, weights, [True])
            np.testing.assert_array_equal(result, np.full((1, 128), topk * .5))

    def test_mask_holes_ignore_nan_weights_and_invalid_payload(self):
        payload = np.full((3, 2, 256), np.nan, dtype=np.float32)
        payload[1] = 2
        weights = np.full((3, 2), np.nan, dtype=np.float32)
        weights[1] = .25
        result = reduce_routes(payload, weights, [False, True, False])
        np.testing.assert_array_equal(result[[0, 2]], 0)
        np.testing.assert_array_equal(result[1], 1)

    def test_fp32_accumulation_and_fixed_order_not_bf16_shortcut(self):
        payload = np.zeros((1, 3, 128), dtype=np.float32)
        payload[0, 0] = 256
        payload[0, 1] = 1
        payload[0, 2] = -256
        result = reduce_routes(payload, np.ones((1, 3), dtype=np.float32), [True])
        np.testing.assert_array_equal(result, 1)
        # Rounding after each BF16 addition would lose the middle +1.
        partial = bf16_rn(bf16_rn(payload[0, 0] + payload[0, 1]) + payload[0, 2])
        np.testing.assert_array_equal(partial, 0)

    def test_nan_not_suppressed_by_zero_weight(self):
        with np.errstate(invalid='ignore'):
            result = reduce_routes(np.full((1, 1, 128), np.inf, dtype=np.float32),
                                   np.zeros((1, 1), dtype=np.float32), [True])
        self.assertTrue(np.isnan(result).all())

    def test_all_f_delay_credit_before_ledger_and_final_consumed(self):
        receiver = RoundReceiver(2)
        credits = []
        emit = lambda rank, bit: credits.append((rank, bit))
        self.assertFalse(receiver.consume(1, [1, 0], 1, False, emit))
        self.assertFalse(credits)
        self.assertTrue(receiver.consume(1, [1, 1], 1, False, emit))
        self.assertEqual(receiver.trace[-2:], [('credit_complete', 1, 0), 'ledger_empty'])
        self.assertTrue(receiver.consume(0, [0, 0], 0, True, emit))
        self.assertFalse(receiver.complete([0, 1]))
        self.assertTrue(receiver.complete([0, 0]))

    def test_shared_rounds_zero_tails_and_independent_slot_bits(self):
        capacities, lengths = (32768, 32768), (262144, 17)
        rounds = max((n + c - 1) // c for n, c in zip(lengths, capacities))
        receivers = [RoundReceiver(2), RoundReceiver(2)]
        ranges = [[], []]
        for round_idx in range(rounds):
            bit = 1 ^ (round_idx & 1)
            for owner, (n, c) in enumerate(zip(lengths, capacities)):
                begin = min(n, round_idx * c)
                count = min(c, n - begin)
                ranges[owner].append((begin, count))
                self.assertTrue(receivers[owner].consume(bit, [bit, bit], bit,
                                                       round_idx == rounds - 1, lambda *_: None))
        self.assertEqual(rounds, 8)
        self.assertEqual(sum(count for _, count in ranges[0]), 262144)
        self.assertEqual(ranges[1], [(0, 17)] + [(17, 0)] * 7)
        self.assertTrue(all(r.complete([0, 0]) for r in receivers))


if __name__ == '__main__':
    unittest.main()
