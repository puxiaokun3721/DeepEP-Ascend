"""Actual five-operator BF16 identity-expert bring-up; requires A5 and compiled extension.

timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py
Uses real scheduler/ACK/credit. Identity experts replace GMM numerics, not protocol.
Host reads are intentional bring-up diagnostics: this is NOT a graph/performance test.
"""
import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
import sys
import struct
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
import torch_npu.profiler
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
    parser.add_argument('--weights', choices=('unit', 'nonunit', 'signed', 'zero01', 'mix'),
                        default='unit')
    parser.add_argument('--routes', choices=('both01', 'balanced', 'hotspot'), default='both01')
    parser.add_argument('--profile-trace', action='store_true')
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--a-budget-mib', type=int, default=8192)
    parser.add_argument('--f-budget-mib', type=int, default=8192)
    parser.add_argument('--reserve-mib', type=int, default=256)
    args = parser.parse_args()
    if args.batch < 0 or not 1 <= args.window <= 32768 or args.q < 1:
        parser.error('batch >= 0, window in [1,32768], q > 0 required')
    if args.short_second is not None and not 0 <= args.short_second <= args.batch:
        parser.error('short-second must be within [0,batch]')
    if args.timeout_seconds <= 0 or args.max_sweeps <= 0:
        parser.error('timeout-seconds and max-sweeps must be positive')
    if args.microbatches > 1 and args.batch > args.window and not args.round_step:
        parser.error('multiple slots with multiple rounds require --round-step')
    if args.a_budget_mib <= 0 or args.f_budget_mib <= 0 or args.reserve_mib <= 0:
        parser.error('budgets and reserve must be positive')
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
    def _align(n, a):
        return (n + a - 1) // a * a

    _role = 'A' if rank in sources else 'F'
    _lanes = 0
    _num_lanes = len(sources) * args.microbatches
    _R = _align(2 * hidden, 512)
    _record_stride = _align(hidden * 2 + 0 + _align(8 * k, 32) + 4 * k + 4 + 4, 512)
    _nf = max(1, experts // len(receivers))
    _nffn = len(receivers)
    # metadata_stride: align32(align32(8K)+4K+4+4); record_stride already computed.
    _mstride = _align(_align(8 * k, 32) + 4 * k + 8, 32)
    if _role == 'A':
        _payload = args.microbatches * _align(args.window * k * _R, 512)
    else:
        _payload = 0
        for _ in range(_num_lanes):
            _payload += _align(args.window * _record_stride, 512)
            _payload += _align(4 * args.window * min(k, _nf), 32)
            _payload += 2 * _align(4 * _nf, 32)
    _control = _align(_num_lanes * _nffn * 64, 32) if _role == 'A' else _align(_num_lanes * 128, 32)
    _ctx_base = (256 + _align(_num_lanes * 32, 32) + _align(_nffn * 32, 32)
        + _align(_num_lanes * _nffn * 64, 32) + _align(1 * 4, 32))
    _ctx = _ctx_base + 32 + 32 * len(sources) + 32 * _num_lanes
    _c = args.window
    _slots = args.microbatches
    _j = (_c + 31) // 32
    _ws_upper = (_align(4 * _c * k, 32) + _align(_c * _mstride, 32)
        + 2 * _align(4 * _j * (_nffn + experts), 32) + _align(4 * _c * k, 32)
        + 2 * _align(4 * experts, 32) + _align(4 * _nffn, 32) + 32 * _slots + 256)
    _ledger = 128 * _slots
    # num_vec_cores=1 and ChannelHandle is uint64_t (hcomm_res_defs.h).
    _ccap = (world - 1) if world > 2 else max(1, world - 1)
    _tables = 2 * _ccap * 8
    if _role == 'A':
        _prep = (256 + _align(32 * 1, 32)) + 32 + (96 * _slots if args.round_step else 0)
    else:
        _active = 64 + 64 * _num_lanes + 32 * _num_lanes * _nf
        _execution = 32 + _align(4 * _num_lanes, 32)
        _drecv = 32 * 1 + 256 + 2 * _align(4 * min(_num_lanes * _nf, args.q), 32)
        _add = _active + _execution + _drecv + args.q * (2 * hidden + 16)
        _sched = 32 + 96 * _num_lanes + 2 * _align(4 * _num_lanes * _nf, 32)
        _compl = 32 + 64 * _num_lanes + _align(4 * _num_lanes * _nf, 32)
        _prep = _add + _sched + _compl + 8 * _nf + args.q * hidden * 2 + (32 * 1 + 256)
    _required = _payload + _control + _ctx + _ws_upper + _ledger + 32 + _tables + _prep
    _B = max(1, args.batch)
    _caller = (args.microbatches * (2 * _B * hidden * 2 + _B * k * 8
        + _B * k * 4 + _B + 64) + 32 * args.microbatches) if _role == 'A' else 0
    _budget = (args.a_budget_mib if _role == 'A' else args.f_budget_mib) * 1024 * 1024
    _reserve = args.reserve_mib * 1024 * 1024
    _info = {'rank': rank, 'role': _role, 'required_bytes': _required,
             'payload_bytes': _payload, 'control_bytes': _control,
             'context_bytes': _ctx, 'workspace_upper_bytes': _ws_upper,
             'ledger_bytes': _ledger, 'tables_bytes': _tables,
             'prepare_bytes': _prep, 'caller_estimate_bytes': _caller,
             'workspace_note': 'j=ceil(C/32) upper bound (runtime tile>=32); includes prepare allocations',
             'model_note': 'prepare_F uses nf=E/num_F; caller uses B=max(1,batch)',
             'task_cap_bytes': 8 * 1024**3,
             'budget_bytes': _budget, 'reserve_bytes': _reserve,
             'fits': _required + _reserve <= _budget,
             'fits_task': _required + _caller + _reserve <= 8 * 1024**3}
    _gathered = [None] * world
    dist.all_gather_object(_gathered, _info)
    if not all(g and g['fits'] and g['fits_task'] for g in _gathered):
        if rank == 0:
            print('PREFLIGHT_REJECT ' + json.dumps(_gathered), flush=True)
        sys.exit(2)
    if rank == 0:
        print('PREFLIGHT_OK ' + json.dumps(_gathered), flush=True)
    try:
        import deep_ep._C as _c
        _so = _c.__file__
        _h = hashlib.sha256()
        with open(_so, 'rb') as _f:
            for _chunk in iter(lambda: _f.read(1 << 20), b''):
                _h.update(_chunk)
        _so_desc = {'path': _so, 'bytes': os.path.getsize(_so), 'sha256': _h.hexdigest()}
    except Exception as _exc:
        _so_desc = {'error': repr(_exc)[:200]}
    print('RUNCFG ' + json.dumps({'rank': rank, 'argv': sys.argv, 'run_tag': args.run_tag,
        'B': args.batch,
        'C': args.window, 'rounds': rounds, 'Q': args.q, 'MB': args.microbatches,
        'weights': args.weights, 'routes': args.routes, 'so': _so_desc}), flush=True)

    def _mem(stage, _owner=None):
        rec = {'stage': stage, 'rank': rank, 'role': _role,
               'torch_alloc': int(torch.npu.memory_allocated()),
               'torch_reserved': int(torch.npu.memory_reserved()),
               'torch_peak': int(torch.npu.max_memory_allocated())}
        if _owner is not None:
            try:
                rec['owner_memory_usage'] = [int(x) for x in _owner.memory_usage()]
            except Exception:
                rec['owner_memory_usage'] = None
            try:
                rec['payload_nbytes'] = int(_owner.payload.numel() * _owner.payload.element_size())
                rec['control_nbytes'] = int(_owner.control_buffer.numel() * _owner.control_buffer.element_size())
            except Exception:
                pass
        print('MEM ' + json.dumps(rec), flush=True)

    _mem('pg_ready')
    owner = deep_ep.AFDBuffer(rank, group_name, sources, receivers,
        [[a, mb, 0, a, 0, args.window] for a in sources for mb in range(args.microbatches)], [0],
        hidden, 0, k, experts, capacity, False, 1, 120, args.timeout_seconds,
        _budget, _reserve)
    _mem('constructed', owner)
    def _stride_dir(tag):
        try:
            _raw = owner.context.cpu().numpy().tobytes()
            _exp_stride = ((2 * hidden + 511) // 512) * 512
            _abi = struct.unpack_from('<I', _raw, 0)[0]
            _res = struct.unpack_from('<Q', _raw, 128)[0]
            _in_range = int(0 <= _res and _res + 32 <= len(_raw))
            if _in_range:
                _spso, _loff, _stride, _ver, _ = struct.unpack_from('<3Q2I', _raw, _res)
            else:
                _spso = _loff = _stride = _ver = None
            _a_payload = int(owner.payload.numel() * owner.payload.element_size()) if _role == 'A' else None
            _expect = (args.microbatches * args.window * k * _stride) if (_role == 'A' and _stride) else None
            _match = (int(_a_payload == _expect) if (_role == 'A' and _stride) else None)
            print('STRIDEDIR ' + json.dumps({'tag': tag, 'rank': rank, 'role': _role,
                'abi_version': _abi, 'abi_ok': int(_abi == 2),
                'dir_offset': _res, 'dir_in_range': _in_range,
                'return_stride': _stride, 'expected_stride_align512_2h': _exp_stride,
                'stride_ok': (int(_stride == _exp_stride) if _stride is not None else None),
                'dir_version': _ver, 'version_ok': (int(_ver == 1) if _ver is not None else None),
                'source_peers_offset': _spso, 'lanes_offset': _loff,
                'a_payload_nbytes': _a_payload, 'a_expected_mb_c_k_stride': _expect,
                'a_payload_matches_expected': _match,
                'f_layout_unchanged': True if _role == 'F' else None}), flush=True)
        except Exception as _exc:
            print('STRIDEDIR_ERR ' + repr(_exc)[:200], flush=True)
    _stride_dir('constructed')
    if rank in sources:
        owner.prepare_combine_recv(round_step=args.round_step, fair_round_robin=args.round_step)
        source_tasks = []
        for mb in range(args.microbatches):
            # Different, exactly representable BF16 data detects cross-slot output confusion.
            x_cpu = ((torch.arange(capacity*hidden).reshape(capacity, hidden) + 3*mb + rank) % 17).to(torch.bfloat16)
            x = x_cpu.npu()
            pairs = [(2 * i, 2 * i + 1) for i in range(experts // 2)]
            if args.routes == 'balanced':
                sel = [pairs[t % len(pairs)] for t in range(capacity)]
            elif args.routes == 'hotspot':
                sel = [pairs[0] if (len(pairs) == 1 or t % 4 != 3) else pairs[1]
                       for t in range(capacity)]
            else:
                sel = [pairs[0]] * capacity
            idx = torch.tensor(sel, dtype=torch.int64).npu()
            WEIGHT_VECTORS = {'unit': (0.5, 0.5), 'nonunit': (0.25, 0.25),
                              'signed': (-0.5, 1.5), 'zero01': (0.0, 1.0), 'mix': (0.25, 0.75)}
            weights = torch.tensor(WEIGHT_VECTORS[args.weights], dtype=torch.float32)
            weights = weights.repeat(capacity, 1).npu()
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
    _mem('prepared', owner)
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
    _mem('prewarmed', owner)
    prof = None
    if args.profile_trace:
        trace_dir = os.path.join(
            os.path.dirname(os.path.abspath(os.environ['EP_JIT_CACHE_DIR'])),
            'trace_profile', f'rank{rank}')
        os.makedirs(trace_dir, exist_ok=False)
        prof = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False, profile_memory=False, with_stack=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir))
        prof.start()
    _seg_t0 = None
    if prof is not None:
        torch.npu.synchronize()
        _seg_t0 = time.monotonic()
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
                w0, w1 = WEIGHT_VECTORS[args.weights]
                xf = source_task.x_cpu.to(torch.float32)
                expected = (xf * w0 + xf * w1).to(torch.bfloat16)
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
                        abi=struct.unpack_from('<I', ctxb, 0)[0]
                        reserved=struct.unpack_from('<Q', ctxb, 128)[0]
                        spso,lanes_off,stride,dver,_=struct.unpack_from('<3Q2I', ctxb, reserved)
                        ret_off,done_off,credit_off,_rl=struct.unpack_from('<4Q', ctxb, lanes_off)
                        pay=bytes_cpu(owner.payload)
                        print(f'DIAGDIR abi={abi} reserved={reserved} spso={spso} lanes_off={lanes_off} stride={stride} ret_off={ret_off} pay_len={len(pay)}', flush=True)
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
    if _seg_t0 is not None:
        torch.npu.synchronize()
        _seg_t1 = time.monotonic()
        print('FUNCSEG ' + json.dumps({'rank': rank, 'role': _role,
            't0': _seg_t0, 't1': _seg_t1, 'dur': _seg_t1 - _seg_t0}), flush=True)
    def stage(name, **kw):
        # Teardown stage marker: rank/stage/monotonic time/rc only; no addresses or handles.
        rec = dict(stage=name, rank=rank, t=round(time.monotonic(), 3))
        rec.update(kw)
        print('STAGE ' + json.dumps(rec), flush=True)
    stage('pre_function_barrier')
    dist.barrier()
    stage('post_function_barrier')
    _mem('function_done', owner)
    _stride_dir('function_done')
    if prof is not None:
        prof.stop()
        print(f'TRACE_EXPORT_RETURNED rank={rank} directory={trace_dir}', flush=True)
    if rank == 0:
        print(json.dumps(dict(status='FUNCTION_PASS', scope='actual five operators, identity experts, no graph',
            lengths=lengths, window=args.window, rounds=rounds, q=args.q, precheck=args.precheck,
            microbatches=args.microbatches, round_step=args.round_step, sweeps_on_rank0=sweeps,
            weights=args.weights, routes=args.routes)))
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
