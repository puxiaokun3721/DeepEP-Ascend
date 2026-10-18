"""Executable scheduler specification. CPU model, not C++/Ascend execution evidence."""
from dataclasses import dataclass, field
import unittest
import random


@dataclass
class Lane:
    dp: int
    member: int
    phase: int = 0
    expected: int = 0
    task: tuple | None = None  # microbatch, layer, round, total_rounds
    counts: list = field(default_factory=list)
    consumed: list = field(default_factory=list)
    credit: bool = True
    held: bool = False
    age: int = 0


def indexed_ready(lanes, precheck=False):
    """CPU model of the two-pass private directory; not kernel execution."""
    representatives = []
    for i, lane in enumerate(lanes):
        representative = i
        for j in range(i):
            prior = lanes[j]
            if (prior.dp, prior.phase, prior.member) != (lane.dp, lane.phase, lane.member):
                continue
            if representative == i:
                representative = j
            if lane.task is not None and prior.task == lane.task:
                raise ValueError('duplicate member')
        representatives.append(representative)
    required = [sum(prior.dp == lane.dp and prior.phase == lane.phase and representatives[j] == j
                    for j, prior in enumerate(lanes)) for lane in lanes]
    ready = []
    for i, lane in enumerate(lanes):
        if lane.task is None:
            continue
        key = (lane.dp, lane.phase, lane.task)
        matches = [j for j, other in enumerate(lanes)
                   if other.task is not None and (other.dp, other.phase, other.task) == key]
        if matches[0] != i:
            continue
        if len(matches) == required[i] and (not precheck or all(lanes[j].credit or lanes[j].held for j in matches)):
            ready.append(tuple(matches))
    return ready


class Scheduler:
    def __init__(self, lanes, q, qa, alignment=1, precheck=False):
        self.lanes, self.q, self.qa, self.alignment = lanes, q, qa, alignment
        self.precheck, self.last_phase, self.active = precheck, None, None

    def publish(self, lane, task, counts, bit=0, age=0):
        item = self.lanes[lane]
        if item.task is not None or bit != item.expected:
            return False
        item.task, item.counts, item.consumed, item.age = task, counts[:], [0] * len(counts), age
        return True

    def select(self):
        if self.active is not None:
            raise ValueError('ARMED cannot be overwritten')
        groups = {}
        for i, lane in enumerate(self.lanes):
            if lane.task is not None:
                groups.setdefault((lane.dp, lane.phase, lane.task), []).append(i)
        ready = []
        for (dp, phase, task), indices in groups.items():
            required = {lane.member for lane in self.lanes if lane.dp == dp and lane.phase == phase}
            members = [self.lanes[i].member for i in indices]
            if len(set(members)) != len(members):
                raise ValueError('duplicate member')
            if set(members) != required:
                continue
            if self.precheck and not all(self.lanes[i].credit or self.lanes[i].held for i in indices):
                continue
            ready.append((phase, task[1], min(self.lanes[i].age for i in indices), indices))
        if not ready:
            return None
        ready.sort(key=lambda group: (group[0] == self.last_phase, group[2]))
        phase, layer = ready[0][:2]
        rows = [0] * len(self.lanes[ready[0][3][0]].counts)
        entries = []
        for candidate_phase, candidate_layer, _, indices in ready:
            if (candidate_phase, candidate_layer) != (phase, layer):
                continue
            deltas = {}
            final = True
            for i in indices:
                lane = self.lanes[i]
                delta = []
                for expert, count in enumerate(lane.counts):
                    used = sum((n + self.alignment - 1) // self.alignment * self.alignment for n in rows)
                    padded = (rows[expert] + self.alignment - 1) // self.alignment * self.alignment
                    fit = (padded + self.qa - used) // self.alignment * self.alignment - rows[expert]
                    left = count - lane.consumed[expert]
                    take = min(left, self.q - sum(rows), fit)
                    delta.append(take)
                    rows[expert] += take
                    final &= take == left
                deltas[i] = delta
            if any(sum(delta) for delta in deltas.values()) or final:
                entries.append((deltas, final))
        self.last_phase = phase
        self.active = (entries, rows)
        return self.active

    def complete(self):
        if self.active is None:
            raise ValueError('no ARMED completion')
        for deltas, final in self.active[0]:
            for i, delta in deltas.items():
                lane = self.lanes[i]
                lane.consumed = [a + b for a, b in zip(lane.consumed, delta)]
                lane.held = not final
                if final:
                    assert lane.consumed == lane.counts
                    lane.task = None
                    lane.expected ^= 1
        self.active = None


class SchedulerTests(unittest.TestCase):
    def test_dp_arrival_and_two_microbatches(self):
        s = Scheduler([Lane(0, 0), Lane(0, 1), Lane(0, 0), Lane(0, 1)], 100, 100)
        s.publish(0, (0, 2, 0, 1), [3])
        self.assertIsNone(s.select())
        for i, mb in [(1, 0), (2, 1), (3, 1)]:
            s.publish(i, (mb, 2, 0, 1), [4])
        self.assertEqual(len(s.select()[0]), 2)
        s.complete()
        self.assertTrue(all(lane.expected == 1 for lane in s.lanes))

    def test_hot_f_64k_two_owners(self):
        s = Scheduler([Lane(0, 0), Lane(0, 1)], 32768, 32768)
        for i in range(2):
            s.publish(i, (0, 2, 0, 8), [32768 * 8])
        slices = 0
        while s.select() is not None:
            self.assertLessEqual(sum(s.active[1]), 32768)
            s.complete()
            slices += 1
        self.assertEqual(slices, 16)

    def test_padding_and_continuation(self):
        s = Scheduler([Lane(0, 0)], 32, 32, alignment=16)
        s.publish(0, (0, 1, 0, 1), [17, 20])
        self.assertEqual(s.select()[1], [17, 0])
        s.complete()
        self.assertTrue(s.lanes[0].held)
        self.assertEqual(s.lanes[0].expected, 0)
        self.assertEqual(s.select()[1], [0, 20])
        s.complete()
        self.assertEqual(s.lanes[0].expected, 1)

    def test_credit_modes_and_bypass(self):
        for precheck in (False, True):
            s = Scheduler([Lane(0, 0, credit=False), Lane(1, 0)], 8, 8, precheck=precheck)
            s.publish(0, (0, 1, 0, 1), [8], age=0)
            s.publish(1, (0, 1, 0, 1), [8], age=1)
            entries, _ = s.select()
            self.assertEqual(next(iter(entries[0][0])), int(precheck))
            with self.assertRaises(ValueError):
                s.select()

    def test_zero_tail_eight_rounds_and_bit_reuse(self):
        s = Scheduler([Lane(0, 0), Lane(0, 1)], 8, 8)
        for round_idx in range(8):
            for i in range(2):
                s.publish(i, (0, 1, round_idx, 8), [0 if i else 1], bit=round_idx % 2)
            self.assertEqual(len(s.select()[0]), 1)
            s.complete()
        self.assertEqual([lane.expected for lane in s.lanes], [0, 0])
        self.assertFalse(s.publish(0, (0, 1, 0, 1), [1], bit=1))
        with self.assertRaises(ValueError):
            s.complete()

    def test_phase_fairness_and_layer_boundary(self):
        s = Scheduler([Lane(0, 0, 0), Lane(1, 0, 1), Lane(2, 0, 0)], 8, 8)
        for i in range(3):
            s.publish(i, (0, i + 1, 0, 1), [1], age=i)
        self.assertEqual(list(s.select()[0][0][0]), [0]); s.complete()
        self.assertEqual(list(s.select()[0][0][0]), [1]); s.complete()
        self.assertEqual(list(s.select()[0][0][0]), [2])

    def test_duplicate_member_rejected(self):
        s = Scheduler([Lane(0, 0), Lane(0, 0)], 8, 8)
        for i in range(2):
            s.publish(i, (0, 1, 0, 1), [1])
        with self.assertRaises(ValueError):
            s.select()




class IndexedDirectoryTests(unittest.TestCase):
    def test_randomized_set_oracle(self):
        rng = random.Random(950)
        for case_idx in range(300):
            lanes = [Lane(dp, member, phase=phase, task=rng.choice([None, (0, 2, 0, 3), (1, 2, 0, 3)]),
                          credit=rng.choice([False, True]), held=rng.choice([False, True]))
                     for dp in range(2) for phase in range(2) for member in range(3) for slot in range(3)]
            rng.shuffle(lanes)
            if case_idx % 2 == 0:
                seen = set()
                for lane in lanes:
                    key = (lane.dp, lane.phase, lane.member, lane.task)
                    if key in seen:
                        lane.task = None
                    elif lane.task is not None:
                        seen.add(key)
            groups = {}
            for i, lane in enumerate(lanes):
                if lane.task is not None:
                    groups.setdefault((lane.dp, lane.phase, lane.task), []).append(i)
            duplicate = any(len({lanes[i].member for i in group}) != len(group) for group in groups.values())
            for precheck in (False, True):
                if duplicate:
                    with self.assertRaises(ValueError):
                        indexed_ready(lanes, precheck)
                    continue
                expected = []
                for (dp, phase, task), group in groups.items():
                    required = {l.member for l in lanes if (l.dp, l.phase) == (dp, phase)}
                    if {lanes[i].member for i in group} == required and (not precheck or all(lanes[i].credit or lanes[i].held for i in group)):
                        expected.append(tuple(group))
                self.assertEqual(indexed_ready(lanes, precheck), expected)

    def test_multiple_slots_missing_members_and_held_credit(self):
        for slots in (2, 3):
            lanes = [Lane(0, member) for member in range(2) for slot in range(slots)]
            for slot in range(slots):
                task = (slot, 4, 2, 8)
                lanes[slot].task = task
                lanes[slots+slot].task = task
                lanes[slots+slot].counts = [0]  # Exhausted owner still participates.
            expected = [(slot, slots+slot) for slot in range(slots)]
            self.assertEqual(indexed_ready(lanes), expected)
            lanes[slots].credit = False
            self.assertEqual(indexed_ready(lanes, True), expected[1:])
            self.assertEqual(indexed_ready(lanes, False), expected)
            lanes[slots].held = True
            self.assertEqual(indexed_ready(lanes, True), expected)
            lanes[slots].task = None
            self.assertEqual(indexed_ready(lanes), expected[1:])

    def test_replay_rebuild_does_not_keep_stale_membership(self):
        lanes = [Lane(0, 0), Lane(0, 1), Lane(0, 0), Lane(0, 1)]
        for bit in (0, 1, 0):
            lanes[0].task = lanes[1].task = (bit, 7, 0, 1)
            self.assertEqual(indexed_ready(lanes), [(0, 1)])
            lanes[2].task = lanes[0].task
            with self.assertRaises(ValueError):
                indexed_ready(lanes)
            lanes[0].task = None
            self.assertEqual(indexed_ready(lanes), [(1, 2)])
            for lane in lanes:
                lane.task = None
            self.assertEqual(indexed_ready(lanes), [])


if __name__ == '__main__':
    unittest.main()
