"""FUNCTION-021 increment 3 CPU additions: same slot, three full transactions, bits 0->1->0.

Model-level only (golden.InputLane is an executable protocol SPECIFICATION, not a
DMA-visibility model); the device-side run_reuse_bits extension asserts the same
invariants on real ledger/control snapshots when cards are available.
"""
import unittest

from golden import InputLane


class BitsSequenceTests(unittest.TestCase):
    def test_three_full_transactions_bits_0_1_0(self):
        lane = InputLane(num_ranks=2)
        bits = []
        for round_idx in range(3):
            bit = lane.publish(round_idx)
            bits.append(bit)
            lane.ack(0)
            lane.ack(1)
            lane.recv_release(round_idx)
        self.assertEqual(bits, [0, 1, 0])

    def test_publish_blocked_without_ack(self):
        lane = InputLane(num_ranks=2)
        lane.publish(0)
        lane.ack(0)                      # rank 1 has not consumed the bit yet
        with self.assertRaises(BlockingIOError):
            lane.publish(1)
        lane.ack(1)
        lane.recv_release(0)
        self.assertEqual(lane.publish(1), 1)

    def test_ledger_needs_recv_release_between_rounds(self):
        lane = InputLane(num_ranks=2)
        lane.publish(0)
        lane.ack(0)
        lane.ack(1)
        with self.assertRaises(BlockingIOError):   # ledger[0] still held
            lane.publish(2)
        lane.recv_release(0)
        self.assertEqual(lane.publish(2), 1)

    def test_zero_length_transaction_keeps_done_ack_cycle(self):
        # Zero-route members still publish/complete with num_round_tokens=0;
        # the bit cycle must not be skipped (device side asserts ledger Done/ACK).
        lane = InputLane(num_ranks=2)
        bit = lane.publish(0)            # transaction with 0 tokens is still a round
        self.assertEqual(bit, 0)
        lane.ack(0)
        lane.ack(1)
        lane.recv_release(0)
        self.assertEqual(lane.publish(1), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
