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


def create(capacities):
    lanes = [[i,0,0,i,0,c] for i,c in enumerate(capacities)]
    return deep_ep.AFDBuffer.create_local_receiver(
        1,0,lanes,[0],128,0,8,8,FULL_BATCH,False,1,128*1024**2)


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
    print(json.dumps(dict(status='PASS', positive_cases=3, negative_cases=1,
                         payload_bytes=sizes,
                         scope='real local receiver host constructor; no kernel/transport')))


if __name__ == '__main__':
    main()
