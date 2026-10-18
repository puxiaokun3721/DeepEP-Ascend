"""Actual A dispatchSend -> synchronize -> explicit F fixture -> dispatchRecv.

timeout 300s torchrun --standalone --nproc_per_node=2 \
  tests/afd/run_send_recv_a5.py --batch 33 --piece-size 7 [--fp8 --transposed-sf]

This does NOT validate online scheduling, asynchronous Ready visibility, input
ACK, transaction reuse or graph replay. Exactly one stable input window is used;
multiple pieces consume explicit slices without recycling the input slot.
"""
import argparse
from datetime import timedelta
import json
import os
import struct

import numpy as np
import torch
import torch.distributed as dist
import torch_npu  # Registers NPU backend.
import deep_ep
from deep_ep.utils.envs import get_hccl_group_name

from gen_data import generate
from golden import verify_snapshot
from recv_reference import CONTROL, Segment, reference
from recv_npu_support import (assert_outputs, install_active, invoke,
                              poison_outputs, raw_bytes, snapshot_active)


def pieces(fixture, piece_size, alignment=4):
    """Explicit test slices of a stable table; no runtime task selection."""
    routes = fixture['routes'][0]
    count = max((len(r) for r in routes), default=0)
    for begin in range(0, max(count, 1), piece_size):
        segments, dst = [], 0
        for expert, refs in enumerate(routes):
            n = min(piece_size, max(0, len(refs)-begin))
            if n:
                route_begin = sum(len(v) for v in routes[:expert]) + begin
                segments.append(Segment(0, 0, expert, route_begin, n, dst))
                dst += (n + alignment-1)//alignment*alignment
        yield segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=33)
    parser.add_argument('--piece-size', type=int, default=7)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--fp8', action='store_true')
    parser.add_argument('--transposed-sf', action='store_true')
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument('--all-masked', action='store_true')
    mask_group.add_argument('--all-active', action='store_true')
    parser.add_argument('--poison-masked-topk', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.batch <= 32768 or args.piece_size <= 0:
        parser.error('single stable round: batch in [0,32768], piece-size > 0')
    if args.transposed_sf and not args.fp8:
        parser.error('--transposed-sf requires --fp8')
    if args.hidden <= 0 or args.hidden % 128:
        parser.error('--hidden must be positive and divisible by 128')
    if not 1 <= args.cores <= 64:
        parser.error('--cores must be in [1,64] and within actual device capacity')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=180))
    if dist.get_world_size() != 2:
        raise ValueError('exactly two ranks: A=0 and F=1')
    rank = dist.get_rank()
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    hidden, k, experts, alignment = args.hidden, 8, 8, 4
    capacity = max(1, args.batch)
    data = generate(batch=capacity, hidden=hidden, k=k, experts=experts,
                    ranks=1, fp8=args.fp8)
    if args.all_masked:
        data['mask'][:] = False
    elif args.all_active:
        data['mask'][:] = True
    if args.poison_masked_topk:
        data['topk'][~data['mask']] = -1
    sf_packs = data['sf_bytes'].shape[1]//2
    owner = deep_ep.AFDBuffer(rank, group_name, [0], [1],
        [[0, 0, 0, 0, 0, capacity]], [0], hidden, sf_packs, k, experts,
        capacity, args.fp8, args.cores, 60, 30, 8*1024**3, 256*1024**2)
    if rank == 1:
        q = max(1, min(capacity, args.piece_size)*k)
        q_alloc = max(4, ((min(capacity, args.piece_size)+3)//4*4)*experts)
        owner.prepare_recv(q, q_alloc, alignment, args.transposed_sf)
    torch.npu.synchronize()
    dist.barrier()
    if rank == 0:
        x = torch.from_numpy(data['x_bytes'].copy()).view(
            torch.float8_e4m3fn if args.fp8 else torch.bfloat16).npu()
        topk = torch.from_numpy(data['topk'].copy()).npu()
        weights = torch.from_numpy(data['weight_bits'].copy().view(np.float32)).npu()
        mask = torch.from_numpy(data['mask'].copy()).npu()
        sf = None
        if args.fp8:
            sf = torch.from_numpy(data['sf_bytes'].copy().view(np.int16)).npu()
            if args.transposed_sf:
                sf = sf.T.contiguous().T
        task = torch.from_numpy(np.frombuffer(struct.pack('<16I',
            1, 0, 0, 0, 0, args.batch, 1, *([0]*9)), dtype=np.uint8).copy()).npu()
        owner.dispatchSend(x, sf, topk, weights, mask, task, owner.context,
            owner.control_buffer, owner.round_ledger, owner.workspace, owner.send_status)
        torch.npu.synchronize()
        assert owner.send_status.cpu().tolist()[:3] == [2, 0, 0], 'sender not PUBLISHED'
    dist.barrier()  # send has actually finished before any F fixture is built.
    if rank == 1:
        fixture, _ = snapshot_active(owner, alignment=alignment)
        # Check source bits independently against generated A inputs. This also
        # catches send corruption that a snapshot-only receive oracle would miss.
        context = raw_bytes(owner.context)
        addresses = struct.unpack_from('<Q', context, 80)[0]
        record_offset, refs_offset, counts_offset, prefix_offset, _, _, _, _ = struct.unpack_from('<8Q', context, addresses)
        payload = raw_bytes(owner.payload)
        counts = np.frombuffer(payload, dtype='<u4', count=experts, offset=counts_offset)
        prefixes = np.frombuffer(payload, dtype='<u4', count=experts, offset=prefix_offset)
        refs = np.frombuffer(payload, dtype='<u4', count=int(prefixes[-1]), offset=refs_offset)
        verify_snapshot(b''.join(fixture['records'][0]), refs, counts, prefixes, 0,
            data['x_bytes'][:args.batch], data['sf_bytes'][:args.batch], data['topk'][:args.batch],
            data['weight_bits'][:args.batch], experts, 1, data['mask'][:args.batch])
        before_control = raw_bytes(owner.control_buffer)
        results = []
        for index, segments in enumerate(pieces(fixture, args.piece_size)):
            kwargs = {} if segments else dict(kind=CONTROL)
            install_active(owner, segments, [[0]], alignment=alignment, **kwargs)
            poison_outputs(owner)
            invoke(owner)
            expected = reference(**dict(fixture, segments=segments), alignment=alignment,
                capacity=owner.recv_x.shape[0], token_capacity=capacity)
            results.append(dict(piece=index, **assert_outputs(owner, expected)))
            assert raw_bytes(owner.control_buffer) == before_control, 'dispatchRecv issued input ACK'
        print(json.dumps(dict(status='PASS', scope='actual send -> sync -> fixture -> recv',
            batch=args.batch, fp8=args.fp8, transposed_sf=args.transposed_sf, pieces=results)), flush=True)
    dist.barrier()
    torch.npu.synchronize()
    owner.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
