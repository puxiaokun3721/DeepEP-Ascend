"""CPU contract oracle for combineSend; does not execute Ascend C or URMA."""
import unittest
from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class Lane:
    rank: int
    capacity: int
    return_offset: int


def route_writes(metadata, lanes, members, k, hidden, stride):
    """Preflight all rows before returning any transfer; no hidden mirror."""
    writes = []
    for row, (lane, token, slot, entry) in enumerate(metadata):
        if lane == -1:
            if (token, slot, entry) != (-1, -1, -1):
                raise ValueError('padding')
            continue
        if lane < 0 or lane >= len(lanes) or entry not in members or lane not in members[entry]:
            raise ValueError('source member')
        source = lanes[lane]
        if not 0 <= token < source.capacity or not 0 <= slot < k:
            raise ValueError('route')
        writes.append((source.rank, source.return_offset + (token*k+slot)*stride, row*hidden*2, hidden*2))
    return writes


class Entry:
    """Independent state machine for persistent grant and final closure order."""
    def __init__(self, members, bit):
        self.members, self.bit, self.held = members, bit, False
        self.consumed = [0]*len(members)

    def submit(self, credit, delta, final, drained=True):
        if not self.held:
            if any(credit[m] != self.bit for m in self.members):
                return ['WAIT_CREDIT']
            self.held = True
        if not drained:
            return ['WAIT_DRAIN']
        events = ['SNAPSHOT']
        self.consumed = [a+b for a, b in zip(self.consumed, delta)]
        if final:
            events += [('DONE', m, self.bit) for m in self.members]
            events += [('CONSUMED', m) for m in self.members]
        return events + ['COMPLETION_READY']


class CombineSendContract(unittest.TestCase):
    def test_scheduler_armed_send_ready_contract(self):
        # A source/ABI regression check, deliberately not a device execution test.
        root = Path(__file__).resolve().parents[2]
        kernel_dir = root / 'deep_ep/include/deep_ep/impls/afd'
        scheduler = (kernel_dir / 'ffn_scheduler.hpp').read_text(encoding='utf-8')
        sender = (kernel_dir / 'combine_send.hpp').read_text(encoding='utf-8')
        self.assertIn('done.state=ARMED', scheduler)
        self.assertIn('done.state!=ARMED', sender)
        self.assertIn('words[0]=READY', sender)
        header = struct.pack('<8I', 1, 0, 2, 3, 24, 0, 0, 0)
        state, error, entries, members, deltas, *_ = struct.unpack('<8I', header)
        self.assertEqual((state, error, entries, members, deltas), (1, 0, 2, 3, 24))
        self.assertEqual(len(header), 32)
    def test_address_above_four_gib_and_source_identity(self):
        lanes = [Lane(0, 32768, 0), Lane(1, 32768, 1 << 33)]
        writes = route_writes([(0, 32767, 31, 0), (-1, -1, -1, -1), (1, 32767, 31, 0)], lanes, {0: {0, 1}}, 32, 7168, 15360)
        self.assertGreater(writes[0][1], 1 << 32)
        self.assertEqual(writes[1][1]-writes[0][1], 1 << 33)
        self.assertEqual(writes[1][2], 2*7168*2)
        self.assertEqual(writes[1][3], 14336)

    def test_invalid_route_rejects_entire_preflight(self):
        lane = [Lane(0, 32768, 0)]
        for md in [(0, 32768, 0, 0), (0, 0, 32, 0), (0, 0, 0, 1), (-1, 0, -1, -1)]:
            with self.assertRaises(ValueError):
                route_writes([(0, 0, 0, 0), md], lane, {0: {0}}, 32, 7168, 15360)

    def test_continuation_retains_grant_until_final(self):
        entry = Entry([0, 1], 0)
        self.assertEqual(entry.submit({0: 0, 1: 1}, [1, 0], False), ['WAIT_CREDIT'])
        self.assertFalse(entry.held)
        first = entry.submit({0: 0, 1: 0}, [1, 0], False)
        self.assertEqual(first, ['SNAPSHOT', 'COMPLETION_READY'])
        self.assertEqual(entry.submit({0: 1, 1: 1}, [0, 2], False), first)
        final = entry.submit({}, [3, 0], True)
        self.assertEqual(entry.consumed, [4, 2])
        self.assertEqual(final, ['SNAPSHOT', ('DONE', 0, 0), ('DONE', 1, 0), ('CONSUMED', 0), ('CONSUMED', 1), 'COMPLETION_READY'])

    def test_zero_route_and_delayed_completion(self):
        entry = Entry([0, 1], 1)
        self.assertEqual(entry.submit({0: 1, 1: 1}, [0, 0], True, drained=False), ['WAIT_DRAIN'])
        self.assertEqual(entry.consumed, [0, 0])
        events = entry.submit({}, [0, 0], True)
        self.assertEqual(len([e for e in events if isinstance(e, tuple) and e[0] == 'CONSUMED']), 2)


if __name__ == '__main__':
    unittest.main()
