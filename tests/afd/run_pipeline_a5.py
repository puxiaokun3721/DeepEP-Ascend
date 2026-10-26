"""Actual five-operator BF16 identity-expert bring-up; requires A5 and compiled extension.

timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py
Uses real scheduler/ACK/credit. Identity experts replace GMM numerics, not protocol.
Host reads are intentional bring-up diagnostics: this is NOT a graph/performance test.
"""
import argparse
from dataclasses import dataclass
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


@dataclass
class SourceTask:
    x_cpu: torch.Tensor
    mask_cpu: torch.Tensor
    x: torch.Tensor
    idx: torch.Tensor
    weights: torch.Tensor
    mask: torch.Tensor
    task: torch.Tensor
    output: torch.Tensor


def receive(owner, source_task, *, start=False):
    owner.combineRecv(source_task.idx, source_task.weights, source_task.mask,
        source_task.task, owner.context, owner.payload, owner.control_buffer,
        owner.round_ledger, owner.combine_recv_workspace, source_task.output,
        owner.recv_status, start=start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=41)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--q', type=int, default=7)
    parser.add_argument('--short-second', type=int)
    parser.add_argument('--precheck', action='store_true')
    parser.add_argument('--all-masked', action='store_true')
    parser.add_argument('--microbatches', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--round-step', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--max-sweeps', type=int, default=100000)
    args = parser.parse_args()
    if args.batch < 0 or not 1 <= args.window <= 32768 or args.q < 1:
        parser.error('batch >= 0, window in [1,32768], q > 0 required')
    if args.short_second is not None and not 0 <= args.short_second <= args.batch:
        parser.error('short-second must be within [0,batch]')
    if args.timeout_seconds <= 0 or args.max_sweeps <= 0:
        parser.error('timeout-seconds and max-sweeps must be positive')
    if args.microbatches > 1 and args.batch > args.window and not args.round_step:
        parser.error('multiple slots with multiple rounds require --round-step')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=args.timeout_seconds))
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
        [[a, mb, 0, a, 0, args.window] for a in sources for mb in range(args.microbatches)], [0],
        hidden, 0, k, experts, capacity, False, 1, 120, args.timeout_seconds,
        8*1024**3, 256*1024**2)
    if rank in sources:
        owner.prepare_combine_recv(round_step=args.round_step, fair_round_robin=args.round_step)
        source_tasks = []
        for mb in range(args.microbatches):
            # Different, exactly representable BF16 data detects cross-slot output confusion.
            x_cpu = ((torch.arange(capacity*hidden).reshape(capacity, hidden) + 3*mb + rank) % 17).to(torch.bfloat16)
            x = x_cpu.npu()
            idx = torch.tensor([0, 1], dtype=torch.int64).repeat(capacity, 1).npu()
            weights = torch.full((capacity, k), 0.5, dtype=torch.float32).npu()
            mask_cpu = torch.arange(capacity) % 5 != 0
            if args.all_masked:
                mask_cpu[:] = False
            mask = mask_cpu.npu()
            # Task ABI word must match AFD kAbiVersion (context.hpp; 2 with the channel_stride layout).
            task = torch.from_numpy(np.frombuffer(struct.pack('<16I',
                2, rank, mb, mb, 0, lengths[rank], rounds, *([0]*9)), dtype=np.uint8).copy()).npu()
            source_tasks.append(SourceTask(x_cpu, mask_cpu, x, idx, weights, mask, task, torch.full_like(x, -7)))
        send_stream, recv_stream = torch.npu.Stream(), torch.npu.Stream()
        send_complete = torch.npu.Event()
        send_results = [torch.empty_like(owner.send_status) for _ in source_tasks]
    else:
        owner.prepare_recv(args.q, args.q, 1)
        owner.prepare_ffn(args.precheck)
    # Prewarm all real specializations serially in world rank order before any
    # communication: exactly one rank compiles per round (true single job).
    for warm_rank in range(world):
        if rank == warm_rank:
            if rank in sources:
                owner.compileDispatchSendOnly()
                owner.compileCombineRecvOnly()
            else:
                owner.compileFFNSchedulerOnly()
                owner.compileDispatchRecvOnly()
                owner.compileComputeReadyOnly()
                owner.compileCombineSendOnly()
        dist.barrier()
    torch.npu.synchronize()
    dist.barrier()
    sweeps = 0
    if rank in sources:
        # One send stream owns the shared scratch; copy tiny statuses into preallocated
        # buffers after each Epub. No allocation or host wait goes behind a live send.
        with torch.npu.stream(send_stream):
            for source_task, send_result in zip(source_tasks, send_results):
                owner.dispatchSend(source_task.x, None, source_task.idx, source_task.weights,
                    source_task.mask, source_task.task, owner.context, owner.control_buffer,
                    owner.round_ledger, owner.workspace, owner.send_status)
                send_result.copy_(owner.send_status)
            send_complete.record(send_stream)
        with torch.npu.stream(recv_stream):
            if args.round_step:
                # Initialization executes exactly once for this transaction set.
                for source_task in source_tasks:
                    receive(owner, source_task, start=True)
                deadline = time.monotonic() + args.timeout_seconds
                while True:
                    # Host observation is queued on recv_stream, never behind the live send.
                    # This is deliberately bring-up code, not a graph/performance loop.
                    progress = bytes_cpu(owner.combine_recv_progress)
                    states = [struct.unpack_from('<4I', progress, 96*mb+64)
                              for mb in range(args.microbatches)]
                    if any(state == 4 for state, _, _, _ in states):
                        raise RuntimeError(f'combineRecv round-step failed: {states}')
                    if all(state == 3 for state, _, _, _ in states):
                        break
                    if time.monotonic() > deadline or sweeps >= args.max_sweeps:
                        raise TimeoutError(f'fair return sweep did not close all slots: {states}, sweeps={sweeps}')
                    for source_task in source_tasks:
                        receive(owner, source_task, start=False)
                    sweeps += 1
            else:
                # The parser and device admission forbid unsafe multi-slot/multi-round use.
                for source_task in source_tasks:
                    receive(owner, source_task)
                    status = owner.recv_status.cpu().tolist()
                    assert status[:2] == [2, 0], f'combineRecv failed: {status}'
            # The last recv-stream copy/event and Epub must both finish before Etxn reuse.
            recv_stream.synchronize()
        send_complete.synchronize()
        with torch.npu.stream(recv_stream):
            for mb, (source_task, result) in enumerate(zip(source_tasks, send_results)):
                assert result.cpu().tolist()[:2] == [2, 0], f'dispatchSend failed for microbatch {mb}'
                expected = source_task.x_cpu.clone()
                expected[~source_task.mask_cpu] = 0
                b = lengths[rank]
                actual = source_task.output.cpu()
                if not torch.equal(actual[:b], expected[:b]):
                    # Failure-only host-side diagnostic (no kernel/protocol change).
                    ab=actual[:b].view(torch.int16).cpu().numpy()
                    eb=expected[:b].view(torch.int16).cpu().numpy()
                    dm=(ab!=eb)
                    idxs=list(zip(*dm.nonzero()))[:16]
                    sample=[f'[{int(i)},{int(j)}] a=0x{int(ab[i,j])&0xffff:04x} e=0x{int(eb[i,j])&0xffff:04x}' for i,j in idxs]
                if not torch.equal(actual[:b], expected[:b]):
                    # Failure-only host-side DIAG (no kernel/protocol change).
                    ab=actual[:b].view(torch.int16).cpu().numpy()
                    eb=expected[:b].view(torch.int16).cpu().numpy()
                    dm=(ab!=eb)
                    idxs=list(zip(*dm.nonzero()))[:16]
                    sample=[f'[{int(i)},{int(j)}] a=0x{int(ab[i,j])&0xffff:04x} e=0x{int(eb[i,j])&0xffff:04x}' for i,j in idxs]
                    print(f'DIAG weighted mismatch mb={mb} total_diff={int(dm.sum())} sum_a={float(actual[:b].float().sum()):.6f} sum_e={float(expected[:b].float().sum()):.6f} sample:', '; '.join(sample), flush=True)
                    try:
                        ctxb=bytes_cpu(owner.context)
                        reserved=struct.unpack_from('<Q', ctxb, 120)[0]
                        spso,lanes_off,stride,dver,_=struct.unpack_from('<3Q2I', ctxb, reserved)
                        ret_off,done_off,credit_off,_rl=struct.unpack_from('<4Q', ctxb, lanes_off)
                        pay=bytes_cpu(owner.payload)
                        print(f'DIAGDIR reserved={reserved} spso={spso} lanes_off={lanes_off} stride={stride} ret_off={ret_off} pay_len={len(pay)}', flush=True)
                        for kk in range(2):
                            off=ret_off+(1*2+kk)*stride
                            vals=struct.unpack_from('<8H', pay, off)
                            print(f'DIAGRAW t=1 k={kk} off={off} first8={[hex(v) for v in vals]}', flush=True)
                        print(f'DIAGW weights_host={source_task.weights[1].tolist()}', flush=True)
                    except Exception as exc:
                        print('DIAGRAW failed', repr(exc)[:400], flush=True)
                assert torch.equal(actual[:b], expected[:b]), f'weighted return mismatch for microbatch {mb}'
                assert torch.all(actual[b:] == -7), f'capacity tail modified for microbatch {mb}'
    else:
        finished = 0
        deadline = time.monotonic() + args.timeout_seconds
        expected_finished = rounds * args.microbatches
        while finished < expected_finished:
            if time.monotonic() > deadline:
                raise TimeoutError(f'scheduler completed {finished}/{expected_finished} DP round entries')
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
        assert finished == expected_finished, 'scheduler completed duplicate DP round entries'
        # Apply last Completion exactly once, retiring the old Pending state.
        owner.ffnScheduler(owner.context, owner.control_buffer, owner.scheduler_workspace,
            owner.active, owner.execution_state, owner.completion,
            owner.psum_num_recv_tokens_per_expert, owner.num_unaligned_recv_tokens_per_expert)
        torch.npu.synchronize()
    def stage(name, **kw):
        # Teardown stage marker: rank/stage/monotonic time/rc only; no addresses or handles.
        rec = dict(stage=name, rank=rank, t=round(time.monotonic(), 3))
        rec.update(kw)
        print('STAGE ' + json.dumps(rec), flush=True)
    stage('pre_function_barrier')
    dist.barrier()
    stage('post_function_barrier')
    if rank == 0:
        print(json.dumps(dict(status='FUNCTION_PASS', scope='actual five operators, identity experts, no graph',
            lengths=lengths, window=args.window, rounds=rounds, q=args.q, precheck=args.precheck,
            microbatches=args.microbatches, round_step=args.round_step, sweeps_on_rank0=sweeps)))
    stage('owner_destroy_begin')
    owner.destroy()
    stage('owner_destroy_end')
    stage('final_barrier_begin')
    dist.barrier()
    stage('final_barrier_end')
    stage('pg_destroy_begin')
    dist.destroy_process_group()
    stage('pg_destroy_end')
    if rank == 0:
        print(json.dumps(dict(status='RUN_PASS', scope='actual five operators, identity experts, no graph',
            lengths=lengths, window=args.window, rounds=rounds, q=args.q, precheck=args.precheck,
            microbatches=args.microbatches, round_step=args.round_step, sweeps_on_rank0=sweeps)))
    stage('main_return')


if __name__ == '__main__':
    main()
