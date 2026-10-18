"""Actual five-operator BF16 identity-expert bring-up; requires A5 and compiled extension.

timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py
Uses real scheduler/ACK/credit. Identity experts replace GMM numerics, not protocol.
Host reads are intentional bring-up diagnostics: this is NOT a graph/performance test.
"""
import argparse
from datetime import timedelta
import json
import os
import struct
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
import deep_ep
from deep_ep.utils.envs import get_hccl_group_name


def bytes_cpu(tensor):
    return tensor.cpu().numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=41)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--q', type=int, default=7)
    parser.add_argument('--short-second', type=int)
    parser.add_argument('--precheck', action='store_true')
    parser.add_argument('--all-masked', action='store_true')
    args = parser.parse_args()
    if args.batch < 0 or not 1 <= args.window <= 32768 or args.q < 1:
        parser.error('batch >= 0, window in [1,32768], q > 0 required')
    if args.short_second is not None and not 0 <= args.short_second <= args.batch:
        parser.error('short-second must be within [0,batch]')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=600))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (2, 4):
        raise ValueError('2 ranks = one A/one F; 4 ranks = two A/two F')
    num_sources = world // 2
    sources, receivers = list(range(num_sources)), list(range(num_sources, world))
    lengths = [args.batch] * num_sources
    if num_sources == 2 and args.short_second is not None:
        lengths[1] = args.short_second
    rounds = max(1, max((b+args.window-1)//args.window for b in lengths))
    capacity, hidden, k = max(1, args.batch), 128, 2
    experts = 2 * len(receivers)
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    owner = deep_ep.AFDBuffer(rank, group_name, sources, receivers,
        [[a, 0, 0, a, 0, args.window] for a in sources], [0],
        hidden, 0, k, experts, capacity, False, 1, 120, 600,
        12*1024**3, 256*1024**2)
    if rank in sources:
        owner.prepare_combine_recv()
        # Exactly representable BF16 and weights make the identity oracle bit-exact.
        x_cpu = (torch.arange(capacity*hidden).reshape(capacity, hidden) % 17).to(torch.bfloat16)
        x = x_cpu.npu()
        idx = torch.tensor([0, 1], dtype=torch.int64).repeat(capacity, 1).npu()
        weights = torch.full((capacity, k), 0.5, dtype=torch.float32).npu()
        mask_cpu = torch.arange(capacity) % 5 != 0
        if args.all_masked:
            mask_cpu[:] = False
        mask = mask_cpu.npu()
        task = torch.from_numpy(np.frombuffer(struct.pack('<16I',
            1, rank, 0, 0, 0, lengths[rank], rounds, *([0]*9)), dtype=np.uint8).copy()).npu()
        output = torch.full_like(x, -7)
        send_stream, recv_stream = torch.npu.Stream(), torch.npu.Stream()
    else:
        owner.prepare_recv(args.q, args.q, 1)
        owner.prepare_ffn(args.precheck)
    torch.npu.synchronize()
    dist.barrier()
    if rank in sources:
        with torch.npu.stream(send_stream):
            owner.dispatchSend(x, None, idx, weights, mask, task, owner.context,
                owner.control_buffer, owner.round_ledger, owner.workspace, owner.send_status)
        with torch.npu.stream(recv_stream):
            owner.combineRecv(idx, weights, mask, task, owner.context, owner.payload,
                owner.control_buffer, owner.round_ledger, owner.combine_recv_workspace, output, owner.recv_status)
        torch.npu.synchronize()
        assert owner.send_status.cpu().tolist()[:2] == [2, 0]
        assert owner.recv_status.cpu().tolist()[:2] == [2, 0]
        expected = x_cpu.clone()
        expected[~mask_cpu] = 0
        b = lengths[rank]
        actual = output.cpu()
        assert torch.equal(actual[:b], expected[:b]), 'weighted return mismatch'
        assert torch.all(actual[b:] == -7), 'capacity tail modified'
    else:
        finished = 0
        deadline = time.monotonic() + 600
        while finished < rounds:
            if time.monotonic() > deadline:
                raise TimeoutError('scheduler did not finish all DP rounds')
            owner.ffnScheduler(owner.context, owner.control_buffer, owner.scheduler_workspace,
                owner.active, owner.execution_state, owner.completion,
                owner.psum_num_recv_tokens_per_expert, owner.num_unaligned_recv_tokens_per_expert)
            header = struct.unpack_from('<10I3Q', bytes_cpu(owner.active))
            if header[1] == 3:
                raise RuntimeError(f'scheduler abort: {list(bytes_cpu(owner.execution_state))}')
            if not header[0]:
                continue
            owner.dispatchRecv(owner.payload, owner.context, owner.active, owner.execution_state,
                owner.recv_workspace, owner.recv_x, owner.recv_sf, owner.recv_src_metadata)
            # Actual identity-expert result; no manual completion or control writes.
            if header[1] == 1:
                owner.combine_x.copy_(owner.recv_x)
            owner.computeReady()
            owner.combineSend(owner.combine_x, owner.recv_src_metadata, owner.context,
                owner.active, owner.control_buffer, owner.execution_state,
                owner.combine_send_workspace, owner.completion)
            completed = bytes_cpu(owner.completion)
            state, error, entries = struct.unpack_from('<3I', completed)
            assert state == 2 and error == 0, 'combineSend did not commit'
            finished += sum(struct.unpack_from('<8I', completed, 32+32*i)[6] for i in range(entries))
        # Apply last Completion exactly once, retiring the old Pending state.
        owner.ffnScheduler(owner.context, owner.control_buffer, owner.scheduler_workspace,
            owner.active, owner.execution_state, owner.completion,
            owner.psum_num_recv_tokens_per_expert, owner.num_unaligned_recv_tokens_per_expert)
        torch.npu.synchronize()
    dist.barrier()
    if rank == 0:
        print(json.dumps(dict(status='PASS', scope='actual five operators, identity experts, no graph',
            lengths=lengths, window=args.window, rounds=rounds, q=args.q, precheck=args.precheck)))
    owner.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
