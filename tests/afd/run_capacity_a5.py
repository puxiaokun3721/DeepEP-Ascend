"""Actual AFDBuffer local-receiver host validation, NOT a transport/scheduler test.

python tests/afd/run_capacity_a5.py --device 0
Requires compiled extension + A5 runtime. Does not dispatch a device kernel.
"""
import argparse
import json
import struct

import torch
import torch_npu  # Registers NPU backend.
import deep_ep

from recv_npu_support import directory, raw_bytes


WINDOW = 32768
FULL_BATCH = 262144
RETURN_LAYOUT_CASES = ((128, 512), (256, 512), (384, 1024),
                       (7168, 14336), (8192, 16384))
RETURN_TEST_TOPK = 2
LOCAL_TEST_BUDGET_BYTES = 128 * 1024**2
CONTEXT_ABI2_PREFIX = '<18I8Q'
RETURN_DIRECTORY_FORMAT = '<3Q2I'
RETURN_LANE_FORMAT = '<4Q'



def create(capacities):
    lanes = [[i,0,0,i,0,c] for i,c in enumerate(capacities)]
    return deep_ep.AFDBuffer.create_local_receiver(
        1,0,lanes,[0],128,0,8,8,FULL_BATCH,False,1,128*1024**2)


def check_return_layout():
    """Observe the real host constructor's directory, without launching kernels."""
    results = []
    # Two physical slots of the same source prove that compact rows do not overlap.
    lanes = [[0, 0, 0, 0, 0, 2], [0, 1, 0, 1, 0, 2]]
    for hidden, expected_stride in RETURN_LAYOUT_CASES:
        owner = deep_ep.AFDBuffer.create_local_receiver(
            1, 0, lanes, [0], hidden, 0, RETURN_TEST_TOPK, 2,
            5, False, 1, LOCAL_TEST_BUDGET_BYTES)
        try:
            context = raw_bytes(owner.context)
            prefix = struct.unpack_from(CONTEXT_ABI2_PREFIX, context)
            assert prefix[0] == 2, 'return-layout test requires context ABI 2'
            _, lane_offset, stride, version, _ = struct.unpack_from(
                RETURN_DIRECTORY_FORMAT, context, prefix[-1])
            assert version == 1 and stride == expected_stride, (hidden, stride)
            lane_size = struct.calcsize(RETURN_LANE_FORMAT)
            offsets = [struct.unpack_from(RETURN_LANE_FORMAT, context,
                                         lane_offset + i * lane_size)[0]
                       for i in range(len(lanes))]
            assert offsets == [0, 2 * RETURN_TEST_TOPK * expected_stride], offsets
            results.append(dict(hidden=hidden, stride=stride, offsets=offsets))
        finally:
            torch.npu.synchronize()
            owner.destroy()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    sizes = []
    for capacities in ([WINDOW], [WINDOW,WINDOW], [WINDOW,16384]):
        owner = create(capacities)
        try:
            d = directory(owner)
            assert [v[5] for v in d['lanes']] == list(capacities)
            context = struct.unpack_from('<16I', raw_bytes(owner.context))
            assert context[9] == FULL_BATCH
            assert context[10] == max(capacities)
            owner.prepare_recv(WINDOW, WINDOW+8*15, 16, False)
            assert tuple(owner.recv_x.shape) == (WINDOW+8*15,128)
            sizes.append(owner.payload.numel())
        finally:
            torch.npu.synchronize()
            owner.destroy()
    assert sizes[1] == 2*sizes[0], sizes
    try:
        owner = create([WINDOW+1])
    except RuntimeError:
        pass
    else:
        owner.destroy()
        raise AssertionError('per-lane C=32769 must remain rejected')
    return_layout = check_return_layout()
    print(json.dumps(dict(status='PASS', positive_cases=3, negative_cases=1,
                         return_layout_cases=return_layout,
                         payload_bytes=sizes,
                         scope='real local receiver host constructor; no kernel/transport')))


if __name__ == '__main__':
    main()
