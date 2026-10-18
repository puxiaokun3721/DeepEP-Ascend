"""Actual single-card dispatchRecv tests; no HCCL or ffnScheduler required.

Run from repository root under an external watchdog:
timeout 300s python tests/afd/run_recv_a5.py --cores 8
Each local owner invokes the same JIT kernel as the real F receiver. Host fixture
uploads happen before each invocation, outside capture. No graph support claim.
"""
import argparse
import json
import struct

import numpy as np
import torch
import torch_npu  # Registers actual NPU backend.
import deep_ep

from recv_reference import (ABORT, ABORTED, CONTROL, EMPTY, NOP, Segment,
                            make_record, reference, sample_fixture, wire_layout)
from recv_npu_support import (assert_outputs, directory, install_active,
                              install_window, invoke, poison_outputs, put_bytes,
                              raw_bytes)


def unchanged(owner):
    tensors = (owner.recv_x, owner.recv_src_metadata)
    values = [raw_bytes(t) for t in tensors]
    if owner.recv_sf is not None:
        values.append(owner.recv_sf.cpu().numpy().tobytes())
    return values


def run_family(fp8, column_major, cores):
    fixture = sample_fixture(fp8)
    zero_fixture = dict(fixture, records=[[], []], routes=[[[], []], [[], []]], segments=[])
    owner = deep_ep.AFDBuffer.create_local_receiver(1, 0,
        [[0, 0, 0, 0, 0, 8], [1, 0, 0, 1, 0, 8]], [0],
        128, fixture['sf_bytes']//2, 2, 2, 8, fp8, cores, 64*1024**2)
    owner.prepare_recv(8, 12, 4, column_major)
    results = []

    def reset(segments=None, entry_lanes=None, **kwargs):
        install_window(owner, fixture)
        install_active(owner, fixture['segments'] if segments is None else segments,
                       [[0, 1]] if entry_lanes is None else entry_lanes, **kwargs)
        poison_outputs(owner)

    def success(name, segments=None, entry_lanes=None, grants=None):
        reset(segments, entry_lanes, grants=grants)
        before_control = raw_bytes(owner.control_buffer)
        before_payload = raw_bytes(owner.payload)
        invoke(owner)
        expected = reference(**dict(fixture, segments=fixture['segments'] if segments is None else segments))
        errors = assert_outputs(owner, expected, entries=1 if entry_lanes is None else len(entry_lanes), grants=grants)
        assert raw_bytes(owner.control_buffer) == before_control, 'dispatchRecv wrote ACK/control'
        assert raw_bytes(owner.payload) == before_payload, 'dispatchRecv modified input'
        results.append(dict(case=name, status='PASS', **errors))

    def error_case(name, expected_error, mutate):
        reset()
        mutate()
        before = unchanged(owner)
        before_control = raw_bytes(owner.control_buffer)
        before_payload = raw_bytes(owner.payload)
        invoke(owner)
        stage, error = struct.unpack_from('<2I', raw_bytes(owner.execution_state))
        assert (stage, error) == (ABORTED, expected_error), (name, stage, error)
        assert unchanged(owner) == before, f'{name}: output modified before validation'
        assert raw_bytes(owner.control_buffer) == before_control, f'{name}: ACK/control modified'
        assert raw_bytes(owner.payload) == before_payload, f'{name}: input modified'
        results.append(dict(case=name, status='PASS', stage=stage, error=error,
                            output_byte_errors=0))

    def set_u32(tensor, offset, value):
        raw = bytearray(raw_bytes(tensor))
        struct.pack_into('<I', raw, offset, value)
        put_bytes(tensor, raw)

    try:
        success('R01-multilane-expert-padding')
        success('R01-held-grant', grants=[1])
        success('R02-multiple-entries-tail', [Segment(0, 0, 0, 0, 1, 0),
            Segment(1, 1, 0, 0, 2, 1)], [[0], [1]])
        success('R02-work-mixed-zero-row-entry', [Segment(0, 0, 0, 0, 1, 0)], [[0], [1]])
        for route in range(3):
            # Same stable window; new Active fixture, no input reuse and no ACK.
            install_active(owner, [Segment(0, 0, 0, route, 1, 0)], [[0]])
            poison_outputs(owner)
            invoke(owner)
            assert_outputs(owner, reference(**dict(fixture,
                segments=[Segment(0, 0, 0, route, 1, 0)])))
        results.append(dict(case='R03-stable-window-three-pieces', status='PASS'))
        for kind, valid, entries, expected_stage in ((NOP, 0, [], EMPTY),
                (ABORT, 0, [], EMPTY), (CONTROL, 1, [[0, 1]], 2)):
            reset([], entries, kind=kind, valid=valid)
            if kind == CONTROL:
                # CONTROL is final zero-wire work, not a zero-row slice of a
                # nonempty input. It is still only an explicit test fixture.
                install_window(owner, zero_fixture)
            before = unchanged(owner)
            before_control = raw_bytes(owner.control_buffer)
            invoke(owner)
            assert unchanged(owner) == before
            assert raw_bytes(owner.control_buffer) == before_control
            assert struct.unpack_from('<2I', raw_bytes(owner.execution_state)) == (expected_stage, 0)
            results.append(dict(case=f'R04-kind-{kind}', status='PASS'))
        reset([], [], kind=ABORT, valid=0)
        set_u32(owner.execution_state, 0, ABORTED)
        set_u32(owner.execution_state, 4, 4)
        set_u32(owner.active, 40, 0xFFFFFFFF)
        before_state, before = raw_bytes(owner.execution_state), unchanged(owner)
        invoke(owner)
        assert raw_bytes(owner.execution_state) == before_state
        assert unchanged(owner) == before
        results.append(dict(case='R04-sticky-ABORTED-poisoned-table', status='PASS'))

        d = directory(owner)
        address = d['addresses'][0]
        pos = wire_layout(fixture['hidden_bytes'], fixture['sf_bytes'], 2)
        error_case('R05-bad-layer', 3, lambda: set_u32(owner.active, 12, 99))
        error_case('R06-prefix-count', 4, lambda: set_u32(owner.payload, address[3], 2))
        error_case('R07-invalid-ref', 4, lambda: set_u32(owner.payload, address[1], 0xFFFFFFFF))
        error_case('R08-record-oob', 4, lambda: set_u32(owner.payload, address[1], 100))
        record = address[0] + 2*pos['stride']
        error_case('R09-token-oob', 4, lambda: set_u32(owner.payload, record + pos['token'], 8))
        error_case('R10-mask-unset', 4, lambda: set_u32(owner.payload, record + pos['valid'], 0))
        error_case('R11-expert-mismatch', 4, lambda: set_u32(owner.payload, record + pos['idx'], 1))
        error_case('R11-late-record-expert-mismatch', 4,
                   lambda: set_u32(owner.payload, address[0] + pos['idx'] + 8, 0))
        error_case('R12-output-capacity', 5, lambda: set_u32(owner.active, 32, 13))
        error_case('R13-active-offset', 5, lambda: set_u32(owner.active, 40, 96))
        error_case('R14-state-order', 6, lambda: set_u32(owner.execution_state, 0, 2))
        error_case('R15-entry-count', 2, lambda: set_u32(owner.active, 64 + 20, 7))
        error_case('R15-grant-mismatch', 2, lambda: set_u32(owner.execution_state, 32, 1))
        error_case('R15-segment-member', 2, lambda: set_u32(owner.active, 192 + 4, 2))
        error_case('R15-segment-dst-hole-overlap', 2, lambda: set_u32(owner.active, 192 + 20, 1))
        error_case('R15-route-cross-expert', 4, lambda: set_u32(owner.active, 192 + 12, 3))
        error_case('R15-invalid-kind-gate', 2, lambda: set_u32(owner.active, 4, NOP))
        error_case('R15-input-bit-mismatch', 2, lambda: set_u32(owner.control_buffer, address[5], 1))
        error_case('R15-CONTROL-on-nonempty-wire', 2,
                   lambda: install_active(owner, [], [[0, 1]], kind=CONTROL))

        def nonfinal_control():
            install_window(owner, zero_fixture)
            install_active(owner, [], [[0, 1]], kind=CONTROL, finals=[0])

        error_case('R15-CONTROL-nonfinal', 2, nonfinal_control)
        success('R17-reset-after-error')
        before = unchanged(owner)
        invoke(owner)  # No SELECTED transition: duplicate use must fail.
        assert struct.unpack_from('<2I', raw_bytes(owner.execution_state)) == (ABORTED, 6)
        assert unchanged(owner) == before
        results.append(dict(case='R18-duplicate-state-order', status='PASS'))

        reset()
        before = unchanged(owner)
        # Construction is outside try: a malformed alias fixture is a test
        # failure, never evidence that dispatchRecv rejected aliasing.
        alias = owner.payload[:owner.recv_x.numel()*owner.recv_x.element_size()].view(owner.recv_x.dtype).view_as(owner.recv_x)
        assert alias.data_ptr() == owner.payload.data_ptr()
        try:
            # Output aliases the actual registered input payload; host must reject.
            invoke(owner, recv_x=alias)
        except RuntimeError:
            assert unchanged(owner) == before
        else:
            raise AssertionError('host accepted input/output alias')
        results.append(dict(case='R19-host-input-output-alias', status='PASS'))
        for name, kwargs in (
                ('R20-host-output-dtype', dict(recv_x=owner.recv_x.to(torch.float32))),
                ('R21-host-metadata-shape', dict(recv_src_metadata=owner.recv_src_metadata[:, :3])),
                ('R22-host-output-output-alias', dict(recv_src_metadata=owner.recv_x.view(torch.int32).reshape(-1)[:48].view(12, 4)))):
            reset()
            before = unchanged(owner)
            try:
                invoke(owner, **kwargs)
            except RuntimeError:
                assert unchanged(owner) == before
            else:
                raise AssertionError(f'{name}: invalid host tensor accepted')
            results.append(dict(case=name, status='PASS'))
        if fp8:
            reset()
            overlap = torch.as_strided(owner.recv_sf, owner.recv_sf.shape, (1, 1))
            try:
                invoke(owner, recv_sf=overlap)
            except RuntimeError:
                pass
            else:
                raise AssertionError('overlapping SF strides accepted')
            results.append(dict(case='R23-host-sf-overlap', status='PASS'))
            reset()
            wrong_sf = owner.recv_sf.to(torch.bfloat16)
            try:
                invoke(owner, recv_sf=wrong_sf)
            except RuntimeError:
                pass
            else:
                raise AssertionError('non-int16 SF accepted')
            results.append(dict(case='R23-host-sf-dtype', status='PASS'))
    finally:
        torch.npu.synchronize()
        owner.destroy()
    return dict(fp8=fp8, sf_column_major=column_major, cores=cores, cases=results)


def run_bad_context(cores):
    """Immutable Context fault injection occurs before this owner's first launch."""
    fixture = sample_fixture()
    owner = deep_ep.AFDBuffer.create_local_receiver(1, 0,
        [[0, 0, 0, 0, 0, 8], [1, 0, 0, 1, 0, 8]], [0],
        128, 0, 2, 2, 8, False, cores, 64*1024**2)
    owner.prepare_recv(8, 12, 4, False)
    try:
        install_window(owner, fixture)
        install_active(owner, fixture['segments'], [[0, 1]])
        poison_outputs(owner)
        raw = bytearray(raw_bytes(owner.context))
        struct.pack_into('<Q', raw, 80, 0xFFFFFFFFFFFFFFFF)
        put_bytes(owner.context, raw)
        before = unchanged(owner)
        invoke(owner)
        assert struct.unpack_from('<2I', raw_bytes(owner.execution_state)) == (ABORTED, 1)
        assert unchanged(owner) == before
        return dict(case='R16-first-launch-context-address', status='PASS', error=1)
    finally:
        torch.npu.synchronize()
        owner.destroy()


def run_edges(fp8, column_major, cores):
    """Exercise padding 0/1/a-1/a/a+1, stage tails, all-bit patterns and Q boundary."""
    hidden_bytes = 128 if fp8 else 256
    sf_bytes = 4 if fp8 else 0
    records = [make_record(bytes((i + token*17) % 256 for i in range(hidden_bytes)),
        bytes((token*7 + i*61) % 256 for i in range(sf_bytes)), [0, 1], token)
        for token in range(33)]
    # Include BF16 special encodings; all FP8 encodings and arbitrary SF bits
    # travel without float conversion.
    if not fp8:
        for token in range(33):
            bits = np.array([(token*128+i)*13 % 65536 for i in range(128)], dtype='<u2')
            bits[:8] = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 1, 0x8001, 0xFFFF]
            records[token] = make_record(bits.tobytes(), b'', [0, 1], token)
    routes = [[[2*i for i in reversed(range(33))], [2*i+1 for i in range(33)]]]
    fixture = dict(records=[records], routes=routes, hidden_bytes=hidden_bytes,
                   sf_bytes=sf_bytes, k=2, num_experts=2)
    owner = deep_ep.AFDBuffer.create_local_receiver(1, 0,
        [[0, 0, 0, 0, 0, 33]], [0], 128, sf_bytes//2, 2, 2, 33, fp8, cores, 64*1024**2)
    owner.prepare_recv(66, 72, 4, column_major)
    results = []
    try:
        install_window(owner, fixture)
        for count in (1, 3, 4, 5, 15, 16, 17, 33):
            segments = [Segment(0, 0, 0, 0, count, 0)]
            if count == 33:
                segments.append(Segment(0, 0, 1, 33, 33, 36))
            install_active(owner, segments, [[0]])
            poison_outputs(owner)
            invoke(owner)
            expected = reference(**dict(fixture, segments=segments), capacity=72, token_capacity=33)
            results.append(dict(case=f'R24-padding-tail-count-{count}',
                                **assert_outputs(owner, expected), status='PASS'))
    finally:
        torch.npu.synchronize()
        owner.destroy()
    return dict(fp8=fp8, sf_column_major=column_major, cores=cores, cases=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cores', type=int, default=8)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    results = [run_family(fp8, column, args.cores)
               for fp8, column in ((False, False), (True, False), (True, True))]
    results += [run_edges(fp8, column, cores) for cores in sorted({1, args.cores})
                for fp8, column in ((False, False), (True, False), (True, True))]
    results.append(run_bad_context(args.cores))
    print(json.dumps(dict(status='PASS', scope='actual single-card NPU kernel', results=results)))


if __name__ == '__main__':
    main()
