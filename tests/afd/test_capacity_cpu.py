"""Capacity clarification regression: specification + existing byte/route oracles.

No scheduler, host extension, transport or Ascend kernel executes in these tests.
"""
import unittest

import numpy as np

from golden import golden, layout
from recv_reference import Segment, align, make_record, reference
from test_cpu import source_workspace


WINDOW = 32768
FULL_BATCH = 262144


def communication_rounds(lengths, capacities):
    """Select one lane per logical member; lengths are extents, not mask counts."""
    if len(lengths) != len(capacities) or not lengths:
        raise ValueError('member list')
    if any(b < 0 or c <= 0 for b, c in zip(lengths, capacities)):
        raise ValueError('member capacity')
    rounds = max(1, max((b+c-1)//c for b, c in zip(lengths, capacities)))
    return [[(min(b, r*c), min(c, max(0, b-r*c)))
             for b, c in zip(lengths, capacities)] for r in range(rounds)]


def f_input_bytes(capacities, hidden_bytes=256, k=8, experts=8):
    """One F's records, route refs and count/prefix arrays for ALL registered slots."""
    stride = layout(hidden_bytes, 0, k)['stride']
    return sum(align(c*stride, 512)+align(4*c*min(k, experts))+
               2*align(4*experts) for c in capacities)


class CapacityClarification(unittest.TestCase):
    def test_two_owners_64k_and_eight_rounds(self):
        rounds = communication_rounds([FULL_BATCH]*2, [WINDOW]*2)
        self.assertEqual(len(rounds), 8)
        self.assertTrue(all(sum(n for _, n in r) == 65536 for r in rounds))
        for member in range(2):
            spans = [r[member] for r in rounds]
            self.assertEqual(spans, [(i*WINDOW, WINDOW) for i in range(8)])
        # Actual existing route oracle: rank dedup keeps one record/token but
        # preserves eight local expert routes; no full hidden allocation needed.
        records, routes = golden(np.tile(np.arange(8), (WINDOW, 1)), 8, 1)
        self.assertEqual(len(records[0]), WINDOW)
        self.assertEqual(sum(sum(v.values()) for v in routes), WINDOW*8)
        self.assertEqual(2*sum(sum(v.values()) for v in routes), 524288)

    def test_heterogeneous_and_exhausted_members(self):
        rounds = communication_rounds([65537, 9, 0], [32768, 4, 8192])
        self.assertEqual(rounds, [[(0,32768),(0,4),(0,0)],
                                  [(32768,32768),(4,4),(0,0)],
                                  [(65536,1),(8,1),(0,0)]])
        rounds = communication_rounds([65537, 3], [32768, 4])
        self.assertEqual([r[1] for r in rounds], [(0,3),(3,0),(3,0)])

    def test_mask_holes_do_not_shorten_transaction_extent(self):
        topk = np.tile([0, 1], (9, 1))
        rounds = communication_rounds([9], [4])
        for mask, expected in (([1,0,0,0,0,0,0,0,1], [1,0,1]),
                               ([0]*9, [0,0,0])):
            observed = []
            for [(begin, count)] in rounds:
                records, _ = golden(topk[begin:begin+count], 2, 1,
                                    mask[begin:begin+count])
                observed.append(len(records[0]))
            self.assertEqual(observed, expected)
            self.assertEqual(len(rounds), 3)

    def test_tp_owner_and_decode_slots_have_different_dimensions(self):
        # TP nonowner has canonical B=0 but retains a positive registered slot.
        rounds = communication_rounds([FULL_BATCH, 0], [WINDOW, WINDOW])
        self.assertTrue(all(sum(n for _, n in r) == WINDOW for r in rounds))
        # Three D slots for the owner are three independent transactions, not
        # three members in one DP round. All slots still reserve F storage.
        self.assertEqual(len(communication_rounds([FULL_BATCH], [WINDOW])), 8)
        one_slot = f_input_bytes([WINDOW])
        self.assertEqual(f_input_bytes([WINDOW]*4), 4*one_slot)
        self.assertEqual(f_input_bytes([WINDOW]*2)-one_slot, 17825856)

    def test_hot_f_fixed_q_and_actual_partial_recv_expansion(self):
        q, alignment, experts = WINDOW, 16, 8
        q_alloc = q + experts*(alignment-1)
        remaining = 2*WINDOW*8
        pieces = []
        while remaining:
            valid = min(remaining, q)
            # Expert 0-heavy adversarial partition creates nonzero padding.
            counts = [valid-7]+[1]*7
            physical = sum(align(n, alignment) for n in counts)
            self.assertLessEqual(physical, q_alloc)
            pieces.append(valid)
            remaining -= valid
        self.assertEqual(len(pieces), 16)
        self.assertEqual(sum(pieces), 524288)
        # Existing recv byte oracle exercises nonzero source offsets, one
        # partial slice crossing owners, and original full-B token identity.
        hidden = bytes(range(128))
        records = [[make_record(hidden, b'', [0], FULL_BATCH-2)],
                   [make_record(hidden[::-1], b'', [0], FULL_BATCH-1)]]
        result = reference(records, [[[0]], [[0]]],
                           [Segment(0,0,0,0,1,0), Segment(0,1,0,0,1,1)],
                           hidden_bytes=128, sf_bytes=0, k=1, num_experts=1,
                           alignment=4, capacity=4, token_capacity=FULL_BATCH)
        self.assertEqual(result[0][0].tobytes(), hidden)
        self.assertEqual(result[0][1].tobytes(), hidden[::-1])
        self.assertEqual(result[2][:2, 1].tolist(), [FULL_BATCH-2, FULL_BATCH-1])

    def test_single_owner_and_workspace_regression(self):
        self.assertEqual(communication_rounds([WINDOW], [WINDOW]), [[(0,WINDOW)]])
        self.assertEqual(communication_rounds([0], [WINDOW]), [[(0,0)]])
        workspace = source_workspace(WINDOW,256,8,64,8,128,1)
        # Actual C++ workspace expressions do not include B, DP aggregate or
        # remote registered-lane count. More slots only add persistent states.
        extra_slots = source_workspace(WINDOW,256,8,64,8,128,3)
        self.assertEqual(extra_slots['num_bytes']-workspace['num_bytes'], 64)


if __name__ == '__main__':
    unittest.main(verbosity=2)
