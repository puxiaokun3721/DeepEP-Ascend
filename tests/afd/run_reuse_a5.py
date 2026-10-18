"""Actual dispatchRecv stress matrix; CPU success is not evidence for this script.

timeout 1800s python tests/afd/run_reuse_a5.py --cores 8 --hidden 7168
Runs 16 counts x 3 dtype/layout families x distinct {1, requested cores}.
"""
import argparse
import json
import struct

import torch
import torch_npu  # Registers NPU backend.
import deep_ep

from recv_npu_support import (assert_outputs, directory, install_active, install_window,
                              invoke, poison_outputs, put_bytes, raw_bytes)
from recv_reference import ABORTED, align, reference
from reuse_fixture import REUSE_COUNTS, reuse_fixture
from run_recv_a5 import unchanged


def run_case(count, fp8, column_major, cores, hidden):
    fixture = reuse_fixture(count, fp8, hidden)
    capacity = 2*align(2*count, 4)+4  # Also check untouched allocation tail.
    owner = deep_ep.AFDBuffer.create_local_receiver(1, 0,
        [[0, 0, 0, 0, 0, count], [1, 0, 0, 1, 0, count]], [0],
        hidden, fixture['sf_bytes']//2, 2, 2, count, fp8, cores, 64*1024**2)
    owner.prepare_recv(4*count, capacity, 4, column_major)
    try:
        install_window(owner, fixture)
        install_active(owner, fixture['segments'], [[0], [1]])
        poison_outputs(owner)
        before_payload, before_control = raw_bytes(owner.payload), raw_bytes(owner.control_buffer)
        invoke(owner)
        expected = reference(**fixture, capacity=capacity, token_capacity=count)
        errors = assert_outputs(owner, expected, entries=2)
        assert raw_bytes(owner.payload) == before_payload
        assert raw_bytes(owner.control_buffer) == before_control

        # Corrupt the final route in the final lane/expert, outside early tiles.
        # Global precheck must prevent ANY output, even if other cores are valid.
        install_active(owner, fixture['segments'], [[0], [1]])
        poison_outputs(owner)
        payload = bytearray(before_payload)
        refs_offset = directory(owner)['addresses'][1][1]
        struct.pack_into('<I', payload, refs_offset+4*(2*count-1), 0xFFFFFFFF)
        put_bytes(owner.payload, payload)
        before_outputs = unchanged(owner)
        invoke(owner)
        assert struct.unpack_from('<2I', raw_bytes(owner.execution_state)) == (ABORTED, 4)
        assert unchanged(owner) == before_outputs, 'partial output on late invalid route'
        assert raw_bytes(owner.payload) == bytes(payload)
        assert raw_bytes(owner.control_buffer) == before_control
        return dict(count=count, fp8=fp8, column_major=column_major, cores=cores,
                    hidden=hidden, status='PASS', late_invalid_route='PASS', **errors)
    finally:
        torch.npu.synchronize()
        owner.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cores', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--counts', type=int, nargs='+', default=REUSE_COUNTS)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    results = []
    for cores in sorted({1, args.cores}):
        for fp8, column in ((False, False), (True, False), (True, True)):
            for count in args.counts:
                result = run_case(count, fp8, column, cores, args.hidden)
                results.append(result)
                print(json.dumps(result), flush=True)
    print(json.dumps(dict(scope='actual NPU kernel', cases=len(results), status='PASS')))


if __name__ == '__main__':
    main()
