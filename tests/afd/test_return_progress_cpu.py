"""CPU state-machine counterexample and round-step contract, not NPU execution."""
from dataclasses import dataclass, field
import ctypes
from pathlib import Path
import unittest

from test_cpu import parse_structs

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Progress:
    task: tuple | None = None
    state: str = 'IDLE'
    next_round: int = 0
    error: str | None = None
    trace: list = field(default_factory=list)

    def fail(self, error):
        self.state = 'ERROR'
        self.error = self.error or error
        return 'ERROR'

    def call(self, task, *, start=False, ready_rounds=frozenset(), ack=False):
        """One invocation, using immutable (microbatch, layer, actual_B, rounds)."""
        if self.state == 'ERROR':
            return 'ERROR'
        if start:
            if self.state not in ('IDLE', 'DONE'):
                return self.fail('STATE_ORDER')
            self.task, self.state, self.next_round = task, 'RUNNING', 0
        elif self.state == 'IDLE' or self.task != task:
            return self.fail('STATE_ORDER')
        if self.state == 'RUNNING' and self.next_round in ready_rounds:
            round_idx = self.next_round
            self.trace.extend([('read_payload', round_idx), ('all_credit_complete', round_idx),
                               ('ledger_empty', round_idx)])
            self.next_round += 1
            if self.next_round == task[3]:
                self.state = 'WAIT_ACK'
        if self.state == 'WAIT_ACK' and ack:
            self.state = 'DONE'
        return 'DONE' if self.state == 'DONE' else 'PENDING'


def legacy_admits(num_local_slots, num_rounds, round_step=False):
    return round_step or num_local_slots == 1 or num_rounds == 1


class ReturnProgressTests(unittest.TestCase):
    def test_old_transaction_recv_has_cycle_new_fair_sweep_breaks_it(self):
        # F0 Active1 returned four of six mb0/r1 routes and closed zero-route
        # mb1/r0. F1 has completed both. The sender has published mb1/r1.
        # F0 Active2 = remaining two mb0/r1 routes + one mb1/r1 route;
        # combineSend's all-entry grant acquisition needs mb1/r1 credit first.
        ready = {0: set(), 1: {0}}
        grant_mb1_r1 = False
        f0_active2_waits = not grant_mb1_r1
        legacy_waits_mb0_r1 = 1 not in ready[0]
        legacy_cannot_reach_mb1_r0 = legacy_waits_mb0_r1
        self.assertTrue(f0_active2_waits and legacy_cannot_reach_mb1_r0)
        # The prior mb0 round has already been merged; neither task may be reset.
        tasks = [(0, 7, 64, 2), (1, 7, 64, 2)]
        progress = [Progress(tasks[0], 'RUNNING', 1), Progress(tasks[1], 'RUNNING', 0)]
        self.assertEqual(progress[0].call(tasks[0], ready_rounds=ready[0]), 'PENDING')
        self.assertEqual(progress[1].call(tasks[1], ready_rounds=ready[1]), 'PENDING')
        grant_mb1_r1 = progress[1].next_round == 1
        self.assertTrue(grant_mb1_r1)
        # The same blocked Active can now complete without another compute workspace.
        ready[0].add(1)
        ready[1].add(1)
        for slot in range(2):
            self.assertEqual(progress[slot].call(tasks[slot], ready_rounds=ready[slot], ack=True), 'DONE')
        self.assertEqual(progress[1].trace[:3], [('read_payload', 0), ('all_credit_complete', 0), ('ledger_empty', 0)])

    def test_one_call_never_drains_multiple_ready_rounds(self):
        task = (0, 3, 262144, 8)
        progress = Progress()
        progress.call(task, start=True, ready_rounds=set(range(8)), ack=True)
        self.assertEqual(progress.next_round, 1)
        for expected in range(2, 9):
            progress.call(task, ready_rounds=set(range(8)), ack=True)
            self.assertEqual(progress.next_round, expected)
        self.assertEqual(progress.state, 'DONE')

    def test_missing_ledger_done_and_final_ack_are_nonblocking(self):
        p = [Progress(), Progress()]
        tasks = [(0, 0, 0, 1), (1, 0, 17, 1)]
        self.assertEqual(p[0].call(tasks[0], start=True), 'PENDING')
        self.assertEqual(p[1].call(tasks[1], start=True, ready_rounds={0}), 'PENDING')
        self.assertEqual(p[1].state, 'WAIT_ACK')
        self.assertEqual(p[0].call(tasks[0], ready_rounds={0}, ack=True), 'DONE')
        self.assertEqual(p[1].call(tasks[1]), 'PENDING')
        self.assertEqual(p[1].call(tasks[1], ack=True), 'DONE')

    def test_repeated_start_and_task_mutation_are_sticky_per_slot_errors(self):
        task = (4, 8, 32, 2)
        for mutation in ('start', 'task'):
            failed, healthy = Progress(), Progress()
            failed.call(task, start=True)
            if mutation == 'start':
                self.assertEqual(failed.call(task, start=True), 'ERROR')
            else:
                self.assertEqual(failed.call((4, 9, 32, 2)), 'ERROR')
            healthy.call(task, start=True, ready_rounds={0})
            self.assertEqual(failed.call(task, ready_rounds={0, 1}, ack=True), 'ERROR')
            self.assertEqual(failed.call(task, start=True), 'ERROR')
            self.assertEqual(failed.next_round, 0)
            self.assertEqual(failed.error, 'STATE_ORDER')
            # A later invalid task/admission error cannot replace the first diagnosis.
            self.assertEqual(failed.fail('BAD_TASK'), 'ERROR')
            self.assertEqual(failed.error, 'STATE_ORDER')

    def test_done_is_idempotent_until_explicit_new_transaction_start(self):
        task = (0, 0, 1, 1)
        p = Progress()
        self.assertEqual(p.call(task, start=True, ready_rounds={0}, ack=True), 'DONE')
        self.assertEqual(p.call(task, ready_rounds={0}, ack=True), 'DONE')
        self.assertEqual(len(p.trace), 3)
        # Same identifiers may be reused only after Etxn plus a new sender transaction.
        self.assertEqual(p.call(task, start=True), 'PENDING')
        self.assertEqual(p.next_round, 0)

    def test_legacy_admission_keeps_single_round_decode_and_single_slot_prefill(self):
        for slots in (2, 3):
            self.assertTrue(legacy_admits(slots, 1))
            self.assertFalse(legacy_admits(slots, 2))
            self.assertTrue(legacy_admits(slots, 8, True))
        self.assertTrue(legacy_admits(1, 8))

    def test_three_slots_zero_tails_and_delayed_notifications_remain_fair(self):
        tasks = [(0, 5, 262144, 8), (1, 5, 17, 8), (2, 5, 0, 8)]
        progress = [Progress() for _ in tasks]
        for slot, task in enumerate(tasks):
            progress[slot].call(task, start=True)
        # Lag differs by slot; all slots stay in every sweep, even WAIT_ACK or zero-tail.
        for sweep in range(30):
            for slot, task in enumerate(tasks):
                visible = set(range(max(0, min(8, sweep-slot))))
                progress[slot].call(task, ready_rounds=visible, ack=sweep>=15+slot)
        self.assertTrue(all(p.state == 'DONE' for p in progress))
        self.assertTrue(all(p.next_round == 8 and len(p.trace) == 24 for p in progress))

    def test_current_wire_sizes_and_progress_memory_are_explicit(self):
        folder = ROOT/'deep_ep/include/deep_ep/layout/afd'
        types = parse_structs((folder/'context.hpp').read_text(encoding='utf-8') + (folder/'return.hpp').read_text(encoding='utf-8'))
        self.assertEqual(ctypes.sizeof(types['AFDContextLayout']), 256)
        self.assertEqual(types['AFDContextLayout'].return_progress_mode.offset, 60)
        self.assertEqual(ctypes.sizeof(types['AFDReturnProgress']), 96)
        self.assertEqual(types['AFDReturnProgress'].state.offset, 64)
        self.assertEqual(types['AFDReturnProgress'].error_code.offset, 76)
        self.assertEqual(96*3, 288)  # Metadata only, no hidden rows or additional reduction scratch.


if __name__ == '__main__':
    unittest.main()
