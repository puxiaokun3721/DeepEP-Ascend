"""Deterministic CPU model of one F rank's fixed selection and whole-group slices."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import unittest


@dataclass(frozen=True)
class Lane:
    dp: int
    member: int
    phase: str
    capacity: int


@dataclass(frozen=True)
class Round:
    dp: int
    mb: int
    layer: int
    index: int = 0
    rounds: int = 1


@dataclass(frozen=True)
class Route:
    expert: int
    token: int
    topk: int
    value: int


def align_up(rows, alignment):
    return (rows + alignment - 1) // alignment * alignment


class Ffn:
    def __init__(self, rank, lanes, q, qa, alignment, backend_max_rows):
        self.rank, self.lanes, self.q, self.qa, self.alignment = rank, lanes, q, qa, alignment
        bound = max(sum(lane.capacity for lane in lanes if lane.phase == phase)
                    for phase in {lane.phase for lane in lanes})
        if not (alignment > 0 and alignment & (alignment - 1) == 0 and
                bound <= q <= qa <= backend_max_rows and align_up(bound, alignment) <= qa):
            raise ValueError(f"CAPACITY single-group bound={bound}, Q={q}, Q_alloc={qa}, backend={backend_max_rows}")
        self.expected = [0] * len(lanes)
        self.wire = {}
        self.pending = {}
        self.clock = 0
        self.last_phase = None
        self.tx_id = 0
        self.active = None
        self.trace = []
        self.results = {}
        self.done = []

    def emit(self, event, **fields):
        self.trace.append({"event": event, "f_rank": self.rank, **fields})

    def publish(self, lane, key, routes, bit=None):
        layout = self.lanes[lane]
        assert key.dp == layout.dp and 0 <= key.index < key.rounds
        assert lane not in self.pending, "cannot overwrite an unretired lane"
        assert lane not in self.wire, "cannot overwrite an unclaimed publication"
        assert all(0 <= route.token < layout.capacity for route in routes)
        assert len({(route.token, route.expert) for route in routes}) == len(routes)
        self.wire[lane] = (key, tuple(routes), self.expected[lane] if bit is None else bit, self.clock)
        self.clock += 1

    def poll(self):
        ready = [lane for lane, (_, _, bit, _) in sorted(self.wire.items())
                 if bit == self.expected[lane]]
        observed = {**self.pending, **{lane: self.wire[lane] for lane in ready}}
        groups = defaultdict(set)
        for lane, (key, _, _, _) in observed.items():
            groups[key, self.lanes[lane].phase].add(self.lanes[lane].member)
        partial = [{"round": vars(key), "phase": phase, "missing_members": sorted(
                    {lane.member for lane in self.lanes if lane.dp == key.dp and lane.phase == phase} - members)}
                   for (key, phase), members in groups.items()]
        self.emit("poll", ready=ready, pending=sorted(self.pending),
                  partial=[item for item in partial if item["missing_members"]])
        return ready

    def schedule(self):
        assert self.active is None
        for lane, publication in list(self.wire.items()):
            if publication[2] == self.expected[lane]:
                self.pending[lane] = self.wire.pop(lane)
        by_key = defaultdict(list)
        for lane, (key, _, _, _) in self.pending.items():
            by_key[key, self.lanes[lane].phase].append(lane)
        ready = []
        for (key, phase), members in by_key.items():
            ids = [self.lanes[lane].member for lane in members]
            if len(ids) != len(set(ids)):
                raise ValueError("BAD_TASK duplicate logical member")
            required = {lane.member for lane in self.lanes if lane.dp == key.dp and lane.phase == phase}
            if set(ids) == required:
                ready.append((min(self.pending[lane][3] for lane in members), key, phase, members))
        if not ready:
            self.emit("scheduler", result="idle")
            return False
        ready.sort(key=lambda item: (item[2] == self.last_phase, item[0]))
        phase = ready[0][2]
        entries = {key: lanes for _, key, entry_phase, lanes in ready if entry_phase == phase}
        groups = defaultdict(list)
        remaining = {}
        for key, lanes in entries.items():
            remaining[key] = 0
            for lane in lanes:
                for route in self.pending[lane][1]:
                    groups[key.layer, route.expert].append((key, lane, route))
                    remaining[key] += 1
        self.tx_id += 1
        self.last_phase = phase
        self.active = {"entries": entries, "groups": sorted(groups.items()), "remaining": remaining,
                       "cursor": 0, "slice": 0, "proposed": None}
        self.emit("scheduler", result="selected", txn=self.tx_id, phase=phase,
                  entries=[vars(key) for key in entries])
        return True

    def dispatch_recv(self):
        active = self.active
        assert active["proposed"] is None
        begin = end = active["cursor"]
        rows = padded = 0
        while end < len(active["groups"]):
            _, routes = active["groups"][end]
            cost = align_up(len(routes), self.alignment)
            if len(routes) > self.q or cost > self.qa:
                raise ValueError("CAPACITY group cannot fit an empty slice")
            if rows + len(routes) > self.q or padded + cost > self.qa:
                break
            rows += len(routes)
            padded += cost
            end += 1
        groups = active["groups"][begin:end]
        active["proposed"] = (end, groups)
        self.emit("dispatch_recv", txn=self.tx_id, slice=active["slice"], groups=[
            {"layer": group[0], "expert": group[1], "rows": len(routes),
             "padded": align_up(len(routes), self.alignment)} for group, routes in groups],
            rows=rows, padded=padded)
        return groups

    def compute(self, groups):
        if groups:
            self.emit("gmm", txn=self.tx_id, slice=self.active["slice"], groups=len(groups))
        # Scalar expert matrix for the CPU trace. Both layer and expert select its weight.
        return [(key, lane, route, route.value * (group[0] * 10 + group[1] + 1))
                for group, routes in groups for key, lane, route in routes]

    def combine_send(self, outputs, *, drain_ok=True):
        active = self.active
        if active.get("aborted"):
            raise RuntimeError("STATE_ORDER aborted transaction")
        end, groups = active["proposed"]
        if not drain_ok:
            active["aborted"] = True
            self.emit("combine_send", txn=self.tx_id, slice=active["slice"], status="aborted")
            raise RuntimeError("TRANSPORT failed before commit")
        planned = [(key, lane, route) for _, routes in groups for key, lane, route in routes]
        assert [(key, lane, route) for key, lane, route, _ in outputs] == planned
        for key, lane, route, result in outputs:
            address = (key, lane, route.token, route.topk)
            assert address not in self.results
            self.results[address] = result
            active["remaining"][key] -= 1
        retired = [key for key, count in active["remaining"].items() if not count]
        active["cursor"] = end
        self.emit("combine_send", txn=self.tx_id, slice=active["slice"], committed=True)
        for key in retired:
            lanes = active["entries"][key]
            for lane in lanes:
                del self.pending[lane]
                self.expected[lane] ^= 1
            del active["remaining"][key]
            self.done.append(key)
            self.emit("done_consumed", txn=self.tx_id, slice=active["slice"], round=vars(key), lanes=lanes)
        active["slice"] += 1
        active["proposed"] = None
        if not active["remaining"]:
            assert end == len(active["groups"])
            self.emit("transaction_end", txn=self.tx_id)
            self.active = None

    def step(self):
        self.combine_send(self.compute(self.dispatch_recv()))

    def drain(self):
        while self.active is not None:
            self.step()


def rows(expert, n, value=2, slot=0):
    return [Route(expert, token, slot, value) for token in range(n)]


def mixed_trace():
    # D: DP0 has two members (one sends zero to this F); DP1 is already at round 1.
    # P is independently ready and waits for the next effective selection.
    lanes = [Lane(0, 0, "D", 4), Lane(0, 1, "D", 4), Lane(1, 0, "D", 4), Lane(2, 0, "P", 4)]
    f = Ffn(0, lanes, q=12, qa=12, alignment=4, backend_max_rows=12)
    d0, d1, p = Round(0, 7, 12), Round(1, 9, 13, 1, 3), Round(2, 5, 12)
    f.poll()
    f.publish(0, d0, rows(0, 4))
    f.poll()
    assert not f.schedule()
    f.publish(1, d0, [])
    f.publish(2, d1, rows(0, 4) + rows(1, 4, slot=1) + rows(2, 4, slot=2))
    f.publish(3, p, rows(0, 1))
    f.poll()
    assert f.schedule()
    f.step()
    assert d0 in f.done and d1 not in f.done
    f.publish(0, Round(0, 8, 12), rows(1, 1))
    f.publish(1, Round(0, 8, 12), [])
    assert f.poll() == [0, 1]
    f.drain()
    assert f.schedule()
    assert f.last_phase == "P"
    f.drain()
    f.publish(2, Round(1, 9, 13, 2, 3), rows(0, 1))
    f.poll()
    assert f.schedule()
    f.drain()
    return f


class TransactionTests(unittest.TestCase):
    def test_fixed_mixed_selection_and_early_retirement(self):
        f = mixed_trace()
        selected = [e for e in f.trace if e["event"] == "scheduler" and e["result"] == "selected"]
        self.assertEqual([e["phase"] for e in selected], ["D", "P", "D"])
        self.assertEqual([(e["dp"], e["mb"], e["layer"], e["index"]) for e in selected[0]["entries"]],
                         [(0, 7, 12, 0), (1, 9, 13, 1)])
        slices = [e for e in f.trace if e["event"] == "dispatch_recv" and e["txn"] == 1]
        self.assertEqual([[(g["layer"], g["expert"]) for g in e["groups"]] for e in slices],
                         [[(12, 0), (13, 0), (13, 1)], [(13, 2)]])
        acks = [e for e in f.trace if e["event"] == "done_consumed" and e["txn"] == 1]
        self.assertEqual([(e["slice"], e["lanes"]) for e in acks], [(0, [0, 1]), (1, [2])])
        self.assertEqual(f.results[Round(0, 7, 12), 0, 0, 0], 242)
        self.assertEqual(f.results[Round(1, 9, 13, 1, 3), 2, 0, 0], 262)

    def test_exact_bound_skew_and_rejected_capacity(self):
        lanes = [Lane(dp, 0, "D", 4) for dp in range(4)]
        for q, qa, limit in ((15, 16, 16), (16, 16, 15), (17, 17, 17)):
            with self.subTest(q=q, qa=qa, limit=limit):
                with self.assertRaisesRegex(ValueError, "CAPACITY"):
                    # Alignment 8 raises the last case's 16 bound to 24 only with a 17-row lane sum.
                    adjusted = lanes if q != 17 else lanes + [Lane(4, 0, "D", 1)]
                    Ffn(0, adjusted, q, qa, 8, limit)
        f = Ffn(0, lanes, 16, 16, 8, 16)
        for dp in range(4):
            f.publish(dp, Round(dp, 0, 12), rows(0, 4))
        self.assertTrue(f.schedule())
        f.drain()
        self.assertEqual(len(f.results), 16)
        self.assertEqual([(e["rows"], e["padded"]) for e in f.trace if e["event"] == "dispatch_recv"], [(16, 16)])

    def test_padding_counts_against_physical_budget(self):
        f = Ffn(0, [Lane(0, 0, "D", 4)], 8, 8, 4, 8)
        f.publish(0, Round(0, 0, 12), rows(0, 1) + rows(1, 1, slot=1) + rows(2, 1, slot=2))
        self.assertTrue(f.schedule())
        f.drain()
        self.assertEqual([(e["rows"], e["padded"]) for e in f.trace if e["event"] == "dispatch_recv"],
                         [(2, 8), (1, 4)])

    def test_whole_groups_80_and_40_at_q100(self):
        f = Ffn(0, [Lane(0, 0, "D", 80)], 100, 100, 4, 100)
        f.publish(0, Round(0, 0, 12), rows(0, 80) + rows(1, 40, slot=1))
        self.assertTrue(f.schedule())
        f.drain()
        self.assertEqual([e["rows"] for e in f.trace if e["event"] == "dispatch_recv"], [80, 40])
        self.assertEqual(len(f.results), 120)

    def test_control_only_and_no_duplicate_ready(self):
        f = Ffn(0, [Lane(0, 0, "D", 4)], 4, 4, 4, 4)
        key = Round(0, 0, 12)
        f.publish(0, key, [])
        self.assertEqual(f.poll(), [0])
        self.assertEqual(f.poll(), [0])
        self.assertTrue(f.schedule())
        self.assertEqual(f.poll(), [])
        f.drain()
        self.assertEqual(f.done, [key])
        self.assertFalse(f.schedule())

    def test_failed_write_keeps_progress_and_does_not_ack(self):
        f = Ffn(0, [Lane(0, 0, "D", 4)], 4, 4, 4, 4)
        f.publish(0, Round(0, 0, 12), rows(0, 4))
        self.assertTrue(f.schedule())
        outputs = f.compute(f.dispatch_recv())
        with self.assertRaisesRegex(RuntimeError, "TRANSPORT"):
            f.combine_send(outputs, drain_ok=False)
        self.assertEqual((f.active["cursor"], f.expected, f.done, f.results), (0, [0], [], {}))
        with self.assertRaisesRegex(RuntimeError, "STATE_ORDER"):
            f.combine_send(outputs)

    def test_f_ranks_select_and_slice_independently(self):
        lanes = [Lane(0, 0, "D", 4), Lane(1, 0, "D", 4)]
        f0, f1 = (Ffn(rank, lanes, 8, 8, 4, 8) for rank in range(2))
        a, b = Round(0, 0, 12), Round(1, 0, 12)
        f0.publish(0, a, rows(0, 4) + rows(1, 4, slot=1) + rows(2, 4, slot=2))
        f0.publish(1, b, [])
        f1.publish(1, b, rows(0, 1))
        self.assertTrue(f0.schedule())
        self.assertTrue(f1.schedule())
        f1.drain()
        self.assertEqual(f1.done, [b])
        self.assertNotIn(b, set(f0.done) & set(f1.done))
        f0.drain()
        self.assertEqual(set(f0.done) & set(f1.done), {b})
        f1.publish(0, a, [])
        self.assertTrue(f1.schedule())
        f1.drain()
        self.assertEqual(set(f0.done) & set(f1.done), {a, b})
        self.assertEqual(sum(e["event"] == "dispatch_recv" and e["txn"] == 1 for e in f0.trace), 2)
        self.assertEqual(sum(e["event"] == "dispatch_recv" and e["txn"] == 1 for e in f1.trace), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="write the mixed-DP/layer trace as JSONL")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TransactionTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.trace:
        args.trace.write_text("".join(json.dumps(event) + "\n" for event in mixed_trace().trace), encoding="utf-8")
