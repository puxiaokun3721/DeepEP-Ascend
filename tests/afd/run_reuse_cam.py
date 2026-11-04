"""CAM AFD owner lifecycle test: construct/prepare/destroy repeatedly in one process.

Verifies that the CAM four-operator owner can be created, prepared and destroyed
repeatedly inside a single process (context/workspace reuse + teardown), which
the single-transaction bring-up harness never exercises.  With --cycle-shapes the
hidden/topk/experts/capacity/window change between cycles, checking that a
re-created owner picks up the new tiling instead of stale state.  Each cycle also
records torch NPU allocator counters so a teardown leak shows up as growth.
--microbatches/--round-step exercise multi-slot buffers and fair round-robin prep.
"""
import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu
import deep_ep
import umdk_cam_op_lib  # noqa: F401 - registers umdk_cam_op_lib torch ops
from deep_ep.utils.envs import get_hccl_group_name


def parse_shapes(spec):
    shapes = []
    for part in spec.split(';'):
        part = part.strip()
        if not part:
            continue
        fields = [int(x) for x in part.split(':')]
        if len(fields) != 5:
            raise ValueError('each --cycle-shapes entry needs batch:hidden:topk:experts:window')
        shapes.append(tuple(fields))
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--experts', type=int, default=0)
    parser.add_argument('--microbatches', type=int, default=1)
    parser.add_argument('--round-step', action='store_true')
    parser.add_argument('--q', type=int, default=7)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--cycle-shapes', default='')
    parser.add_argument('--a-budget-mib', type=int, default=8192)
    parser.add_argument('--f-budget-mib', type=int, default=8192)
    parser.add_argument('--reserve-mib', type=int, default=256)
    parser.add_argument('--run-tag', default='')
    args = parser.parse_args()
    shapes = parse_shapes(args.cycle_shapes)
    if args.round_step and args.microbatches < 1:
        raise ValueError('microbatches must be >= 1')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=args.timeout_seconds))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (2, 4):
        raise ValueError('2 ranks = one A/one F; 4 ranks = two A/two F')
    num_sources = world // 2
    sources, receivers = list(range(num_sources)), list(range(num_sources, world))
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    role = 'A' if rank in sources else 'F'
    budget = (args.a_budget_mib if role == 'A' else args.f_budget_mib) * 1024 * 1024
    reserve = args.reserve_mib * 1024 * 1024
    for cyc in range(args.cycles):
        batch, hidden, k, experts_arg, window = (
            shapes[cyc % len(shapes)] if shapes
            else (args.batch, args.hidden, args.topk, args.experts, args.window))
        capacity = max(1, batch)
        experts = experts_arg if experts_arg > 0 else 2 * len(receivers)
        mem_before = [int(torch.npu.memory_allocated()), int(torch.npu.memory_reserved())]
        t0 = time.monotonic()
        owner = deep_ep.AFDBuffer(rank, group_name, sources, receivers,
            [[a, mb, 0, a, 0, window] for a in sources for mb in range(args.microbatches)], [0],
            hidden, 0, k, experts, capacity, False, 1, 120, args.timeout_seconds,
            budget, reserve)
        cfg = owner.camConfig()
        if role == 'A':
            owner.prepare_combine_recv(round_step=args.round_step,
                                       fair_round_robin=args.round_step)
            cfg.update({key: int(value) for key, value in owner.camConfig().items()})
        else:
            owner.prepare_recv(args.q, args.q, 1)
            cfg.update({key: int(value) for key, value in owner.camConfig().items()})
        torch.npu.synchronize()
        t1 = time.monotonic()
        owner_usage = None
        try:
            owner_usage = [int(x) for x in owner.memory_usage()]
        except Exception:
            pass
        progress_state = None
        if role == 'A' and args.round_step:
            try:
                import struct
                raw = bytes(owner.combine_recv_progress.cpu().numpy().tobytes())
                progress_state = [struct.unpack_from('<4I', raw, 96 * mb)[0]
                                  for mb in range(args.microbatches)]
            except Exception as exc:
                progress_state = 'ERR:' + repr(exc)[:80]
        print('REUSE_CYCLE ' + str(dict(cycle=cyc, rank=rank, role=role,
              batch=batch, hidden=hidden, topk=k, experts=experts, window=window,
              microbatches=args.microbatches, round_step=args.round_step,
              construct_s=round(t1 - t0, 3),
              num_hidden_bytes=cfg.get('num_hidden_bytes'),
              num_vec_cores=cfg.get('num_vec_cores'),
              progress_state=progress_state,
              owner_memory_usage=owner_usage,
              torch_alloc=int(torch.npu.memory_allocated()),
              torch_reserved=int(torch.npu.memory_reserved()))), flush=True)
        owner.destroy()
        del owner
        torch.npu.synchronize()
        t2 = time.monotonic()
        print('REUSE_DESTROYED ' + str(dict(cycle=cyc, rank=rank,
              destroy_s=round(t2 - t1, 3),
              torch_alloc_after=int(torch.npu.memory_allocated()),
              torch_reserved_after=int(torch.npu.memory_reserved()),
              alloc_delta_vs_before=int(torch.npu.memory_allocated()) - mem_before[0],
              reserved_delta_vs_before=int(torch.npu.memory_reserved()) - mem_before[1])), flush=True)
        dist.barrier()
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print('REUSE_PASS cycles=%d world=%d' % (args.cycles, world), flush=True)


if __name__ == '__main__':
    main()
