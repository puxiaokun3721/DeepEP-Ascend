"""CPU contract tests only: never report these as NPU operator execution."""
import copy
import ctypes
from pathlib import Path
import re
import random
import struct
import unittest

import numpy as np

from recv_reference import (ABORT, ACTIVE, CONTROL, EMPTY, NOP, ROW, STATE,
                            Segment, UINT32_MAX, bit_errors, pack_active,
                            reference, sample_fixture, validate_segment_intervals, wire_layout)
from test_cpu import parse_structs


class RecvContractTests(unittest.TestCase):
    def test_actual_cpp_abi_matches_python_fixture(self):
        header = Path(__file__).resolve().parents[2]/'deep_ep/include/deep_ep/layout/afd/active.hpp'
        text = header.read_text(encoding='utf-8-sig')
        types = parse_structs(text)
        self.assertEqual(set(types), {'AFDActive', 'AFDActiveEntry', 'AFDActiveMember',
                                     'AFDActiveSegment', 'AFDExecutionState'})
        for name, size in (('AFDActive', 64), ('AFDActiveEntry', 32),
                           ('AFDActiveMember', 32), ('AFDActiveSegment', 32),
                           ('AFDExecutionState', 32)):
            self.assertEqual(ctypes.sizeof(types[name]), size)
        expected_fields = {
            'AFDActive': ['valid', 'kind', 'phase', 'layer_idx', 'num_entries', 'num_members',
                          'num_segments', 'num_expanded_tokens', 'num_padded_tokens', 'reserved'],
            'AFDActiveEntry': ['pending_idx', 'member_begin', 'num_members', 'segment_begin',
                               'num_segments', 'num_rows', 'is_final', 'grant_state'],
            'AFDActiveMember': ['source_lane_idx', 'input_bit', 'return_bit', 'delta_begin', 'num_experts'],
            'AFDActiveSegment': ['entry_idx', 'source_lane_idx', 'local_expert_idx', 'route_begin',
                                 'num_routes', 'dst_row_begin'],
            'AFDExecutionState': ['stage', 'error_code', 'num_entries']}
        for name, fields in expected_fields.items():
            for index, field in enumerate(fields):
                self.assertEqual(getattr(types[name], field).offset, index*4)
        for index, field in enumerate(('entries_offset', 'members_offset', 'segments_offset')):
            self.assertEqual(getattr(types['AFDActive'], field).offset, 40+index*8)
        methods = dict(re.findall(r'uint64_t (\w+)\([^)]*\) \{ return (.*?); \}', text))
        for lanes, experts, cores, q in ((1, 1, 1, 32), (2, 2, 8, 2), (33, 512, 64, 32768)):
            active, state = pack_active(lanes, experts, [], [], kind=NOP, valid=0)
            env = dict(l=lanes, r=experts, cores=cores, q=q, sizeof=ctypes.sizeof,
                       uint64_t=int, AFDActive=types['AFDActive'], align32=lambda n: (n+31)//32*32)
            for name, expression in methods.items():
                if '?' in expression:
                    condition, choices = expression.split('?')
                    yes, no = choices.split(':')
                    expression = f'({yes}) if ({condition}) else ({no})'
                env[name] = lambda *args, expression=expression: eval(expression, {'__builtins__': {}}, env)
            segment_capacity = min(lanes*experts, q)
            index_bytes = (4*segment_capacity+31)//32*32
            expected = dict(active_entries_offset=64, active_members_offset=64+32*lanes,
                active_segments_offset=64+64*lanes, active_bytes=len(active),
                execution_bytes=len(state), recv_workspace_bytes=32*cores+256,
                recv_segment_capacity=segment_capacity, recv_segment_index_bytes=index_bytes,
                dispatch_recv_workspace_bytes=32*cores+256+2*index_bytes)
            for name, value in expected.items():
                self.assertIn(name, methods)
                self.assertEqual(env[name](), value, name)

    def test_real_abi_sizes_and_offsets(self):
        self.assertEqual((ACTIVE.size, ROW.size, STATE.size), (64, 32, 32))
        active, state = pack_active(2, 2, sample_fixture()['segments'], [[0, 1]])
        self.assertEqual(len(active), 320)
        self.assertEqual(len(state), 64)
        self.assertEqual(struct.unpack_from('<3Q', active, 40), (64, 128, 192))
        self.assertEqual(struct.unpack_from('<8I', active, 64), (0, 0, 2, 0, 4, 6, 0, 0))
        self.assertEqual(struct.unpack_from('<8I', active, 160)[:5], (1, 0, 0, 2, 2))

    def test_frozen_wire_examples(self):
        for hidden, sf, stride, metadata in ((7168, 224, 7680, 352), (14336, 0, 14848, 128)):
            pos = wire_layout(hidden, sf, 8)
            self.assertEqual((pos['stride'], pos['metadata']), (stride, metadata))

    def test_exact_bytes_metadata_padding_both_dtypes(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                f = sample_fixture(fp8)
                x, sf, meta = reference(**f)
                np.testing.assert_array_equal(meta, [[0, 1, 0, 0], [0, 2, 0, 0],
                    [1, 1, 0, 0], [-1]*4, [1, 2, 1, 0], [1, 0, 1, 0],
                    [0, 2, 1, 0], [-1]*4])
                self.assertEqual(x[0].tobytes(), f['records'][0][2][:f['hidden_bytes']])
                self.assertFalse(x[3].any() or x[7].any())
                self.assertFalse(sf[3].any() or sf[7].any())
                self.assertEqual(bit_errors(x, x.copy()), 0)

    def test_two_sf_strides_preserve_all_bits(self):
        f = sample_fixture(True)
        _, sf_bytes, _ = reference(**f)
        expected = sf_bytes.copy().view(np.int16)
        for strides in ((expected.shape[1]*2, 2), (2, len(expected)*2)):
            storage = bytearray(expected.nbytes)
            view = np.ndarray(expected.shape, dtype=np.int16, buffer=storage, strides=strides)
            view[:] = expected
            self.assertEqual(bit_errors(view, expected), 0)

    def test_column_sf_multiburst_addresses_and_tail_poison(self):
        # 950 C API head-to-head byte strides, including T<16 non-32B bursts.
        # Exercise several UB column chunks and a padded destination stride.
        for stage_tokens in (1, 3, 8, 16):
            for hidden_bytes, packs in ((32, 7), (128, 17), (7168, 112)):
                columns_per_stage = hidden_bytes//32
                dst_rows, row = 37, 5
                dst = bytearray([0xCD]) * (2*dst_rows*packs)
                expected = dst.copy()
                inputs = np.arange(stage_tokens*packs, dtype=np.int16).reshape(stage_tokens, packs)
                for p in range(packs):
                    for token in range(stage_tokens):
                        struct.pack_into('<h', expected, 2*((row+token)+p*dst_rows), int(inputs[token, p]))
                transfers = 0
                for sf_begin in range(0, packs, columns_per_stage):
                    columns = min(columns_per_stage, packs-sf_begin)
                    ub = bytearray([0xEF]) * hidden_bytes
                    for p in range(columns):
                        for token in range(stage_tokens):
                            struct.pack_into('<h', ub, 32*p+2*token, int(inputs[token, sf_begin+p]))
                    # One instruction, columns bursts; dummy UB tail is not copied.
                    for p in range(columns):
                        begin = 2*(row+(sf_begin+p)*dst_rows)
                        dst[begin:begin+2*stage_tokens] = ub[32*p:32*p+2*stage_tokens]
                    transfers += 1
                self.assertEqual(dst, expected)
                self.assertEqual(transfers, (packs+columns_per_stage-1)//columns_per_stage)

    def test_unordered_intervals_match_independent_row_oracle(self):
        rng = random.Random(20261021)
        fixture = sample_fixture()
        for _ in range(40):
            segments = list(fixture['segments'])
            rng.shuffle(segments)
            self.assertEqual(validate_segment_intervals(segments, num_experts=2, alignment=4, capacity=8), 8)
            for actual, expected in zip(reference(**dict(fixture, segments=segments)), reference(**fixture)):
                np.testing.assert_array_equal(actual, expected)
        # Same lane/expert can have several disjoint Segments in arbitrary order.
        segments = [Segment(0, 0, 0, 2, 1, 2), Segment(0, 0, 0, 0, 2, 0)]
        self.assertEqual(validate_segment_intervals(segments, num_experts=2, alignment=4, capacity=4), 4)
        self.assertEqual(len(reference(**dict(fixture, segments=segments), capacity=4)[0]), 4)
        for bad in ([Segment(0, 0, 0, 0, 2, 0), Segment(0, 0, 0, 1, 1, 2)],
                    [Segment(0, 0, 0, 0, 2, 0), Segment(0, 1, 0, 0, 1, 1)],
                    [Segment(0, 0, 0, 0, 1, 0), Segment(0, 1, 0, 0, 1, 2)]):
            with self.assertRaises(ValueError):
                validate_segment_intervals(bad, num_experts=2, alignment=4, capacity=8)
            with self.assertRaises(ValueError):
                reference(**dict(fixture, segments=bad))

    def test_many_segments_and_cross_core_intersections(self):
        # A 4096-Segment piece spans 32k rows. No DP-wide 32k assumption:
        # these are effective rows selected from independent physical lanes.
        segments = [Segment(0, lane, 0, 0, 8, lane*8) for lane in range(4096)]
        random.Random(7).shuffle(segments)
        self.assertEqual(validate_segment_intervals(segments, num_experts=1, alignment=1, capacity=32768), 32768)
        coverage = np.zeros(32768, dtype=np.uint8)
        for core in range(7):
            start, end = 32768*core//7, 32768*(core+1)//7
            for segment in segments:
                begin, limit = max(start, segment.dst), min(end, segment.dst+segment.count)
                if begin < limit:
                    coverage[begin:limit] += 1
        np.testing.assert_array_equal(coverage, np.ones_like(coverage))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            validate_segment_intervals(segments, num_experts=1, alignment=1, capacity=32767)

    def test_multiple_pieces_same_stable_window(self):
        f = sample_fixture(True)
        for route in (0, 1, 2):
            piece = dict(f, segments=[Segment(0, 0, 0, route, 1, 0)])
            x, _, meta = reference(**piece)
            rec, slot = divmod(f['routes'][0][0][route], f['k'])
            self.assertEqual(x[0].tobytes(), f['records'][0][rec][:f['hidden_bytes']])
            self.assertEqual(meta[0, 2], slot)

    def test_empty_control_nop_abort_fixtures(self):
        f = sample_fixture()
        x, sf, meta = reference(**dict(f, segments=[]))
        self.assertEqual((x.shape[0], sf.shape[0], meta.shape[0]), (0, 0, 0))
        for kind, valid, entries in ((CONTROL, 1, [[0]]), (NOP, 0, []), (ABORT, 0, [])):
            active, state = pack_active(2, 2, [], entries, kind=kind, valid=valid)
            self.assertEqual(ACTIVE.unpack_from(active)[7:9], (0, 0))
            if valid == 0:
                self.assertEqual(STATE.unpack_from(state)[0], EMPTY)
            if kind == CONTROL:
                self.assertEqual(ROW.unpack_from(active, 64)[6], 1)

    def test_capacity_boundary(self):
        f = sample_fixture()
        self.assertEqual(len(reference(**f, capacity=8)[0]), 8)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            reference(**f, capacity=7)

    def test_record_route_and_slot_are_distinct(self):
        f = sample_fixture()
        _, _, meta = reference(**f)
        # route reference 4 -> record index 2 -> original token index 1, slot 0.
        self.assertEqual(f['routes'][0][0][0], 4)
        np.testing.assert_array_equal(meta[0], [0, 1, 0, 0])

    def test_invalid_routes(self):
        for ref in (UINT32_MAX, 100, -1, UINT32_MAX + 1):
            f = sample_fixture()
            f['routes'][0][0][0] = ref
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                reference(**f)

    def test_invalid_record_metadata(self):
        for field in ('token', 'valid', 'idx'):
            f = sample_fixture()
            raw = bytearray(f['records'][0][2])
            pos = wire_layout(f['hidden_bytes'], f['sf_bytes'], f['k'])
            struct.pack_into('<q' if field == 'idx' else '<I', raw, pos[field],
                             1 if field == 'idx' else (128 if field == 'token' else 0))
            f['records'][0][2] = bytes(raw)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'metadata'):
                reference(**f)

    def test_invalid_segment_bounds_and_duplicates(self):
        f = sample_fixture()
        for bad in (Segment(0, 0, 0, 3, 1, 0), Segment(0, 0, 0, 0, 1, 4),
                    Segment(0, 0, 0, 0, 0, 0), Segment(0, -1, 0, 0, 1, 0),
                    Segment(0, 0, -1, 0, 1, 0), Segment(0, 0, 2, 0, 1, 0)):
            with self.subTest(segment=bad), self.assertRaises(ValueError):
                reference(**dict(f, segments=[bad]))
        for segments in ([Segment(0, 0, 0, 0, 1, 0), Segment(0, 0, 0, 0, 1, 1)],
                         [Segment(0, 0, 0, 0, 1, 0), Segment(0, 0, 0, 1, 1, 0)]):
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                reference(**dict(f, segments=segments))

    def test_tail_and_multiple_entries(self):
        f = sample_fixture()
        segments = [Segment(0, 0, 0, 0, 1, 0), Segment(1, 1, 0, 0, 2, 1)]
        active, _ = pack_active(2, 2, segments, [[0], [1]], alignment=4)
        self.assertEqual(ACTIVE.unpack_from(active)[4:9], (2, 2, 2, 3, 4))
        _, _, meta = reference(**dict(f, segments=segments))
        np.testing.assert_array_equal(meta[:, 3], [0, 1, 1, -1])

    def test_tail_hole_poison_not_copied(self):
        f = sample_fixture()
        original = reference(**f)
        poisoned = copy.deepcopy(f)
        pos = wire_layout(f['hidden_bytes'], f['sf_bytes'], f['k'])
        end = f['hidden_bytes'] + pos['metadata']
        for lane in poisoned['records']:
            for i, raw in enumerate(lane):
                lane[i] = raw[:end] + bytes([0xEC]) * (len(raw) - end)
        for before, after in zip(original, reference(**poisoned)):
            self.assertEqual(bit_errors(before, after), 0)

    def test_byte_comparator_rejects_nan_payload_and_signed_zero_changes(self):
        a = np.array([0x8000, 0x7FC1], dtype=np.uint16)
        b = np.array([0x0000, 0x7FC0], dtype=np.uint16)
        self.assertEqual(bit_errors(a, b), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
