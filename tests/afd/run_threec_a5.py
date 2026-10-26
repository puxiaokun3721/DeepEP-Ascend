"""2A2F same-owner/same-slot timed transaction reuse (AUDIT025 step 4, three-C).

torchrun --standalone --nproc_per_node=4 tests/afd/run_threec_a5.py --batch 32768 --window 32768 --q 1024 --round-step --warmup 3 --samples 20
Design: every active token routes to two experts, one on EACH F rank; per-expert
deterministic transform x*S[g]+C[g]; per-token unequal weights; inputs distinct per
(source, token, microbatch) and exactly BF16; the independent CPU oracle in
discrim_oracle.py (no device implementation imported) computes the expected rows.
Uses real scheduler/ACK/credit. Host reads are intentional bring-up diagnostics.
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
from discrim_oracle import (S as _S, C as _C, expected_rows, route_table,
                            weight_table, make_inputs)


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
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--samples', type=int, default=20)
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
    _proc_t0 = time.monotonic()
    _work_deadline = _proc_t0 + args.timeout_seconds
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
        _TXN = args.warmup + args.samples
        send_stream, recv_stream = torch.npu.Stream(), torch.npu.Stream()
        send_complete = torch.npu.Event()
        send_results = [torch.empty_like(owner.send_status) for _ in range(_TXN)]
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
        # Three transactions on the SAME owner and SAME physical slot; the ABI task
        # words stay (2, rank, 0, 0, 0, B, rounds); data differs per transaction.
        _txn_bits = []
        _txn_times = []
        for txn in range(args.warmup + args.samples):
            _xint = make_inputs(capacity, hidden, txn, rank)
            x_cpu = torch.from_numpy(_xint.astype(np.float32)).to(torch.bfloat16)
            x = x_cpu.npu()
            idx = torch.tensor(route_table(capacity, txn), dtype=torch.int64).npu()
            weights = torch.tensor(np.array(weight_table(capacity, txn), dtype=np.float32)).npu()
            mask_cpu = torch.ones(capacity, dtype=torch.bool)
            mask = mask_cpu.npu()
            task = torch.from_numpy(np.frombuffer(struct.pack('<16I',
                2, rank, 0, 0, 0, lengths[rank], rounds, *([0]*9)), dtype=np.uint8).copy()).npu()
            source_task = SourceTask(x_cpu, mask_cpu, x, idx, weights, mask, task, torch.full_like(x, -7))
            # Preparation for this transaction (CPU tables, H2D copies, full_like output)
            # must be complete before t0: one per-rank synchronize (no cross-rank sync).
            torch.npu.synchronize()
            _t0 = time.monotonic()
            with torch.npu.stream(send_stream):
                owner.dispatchSend(source_task.x, None, source_task.idx, source_task.weights,
                    source_task.mask, source_task.task, owner.context, owner.control_buffer,
                    owner.round_ledger, owner.workspace, owner.send_status)
                send_results[txn].copy_(owner.send_status)
                send_complete.record(send_stream)
            with torch.npu.stream(recv_stream):
                # start=True admits this transaction (kernel: RETURN_IDLE/RETURN_DONE).
                receive(owner, source_task, start=True)
                deadline = time.monotonic() + args.timeout_seconds
                _txn_sweeps = 0
                while True:
                    progress = bytes_cpu(owner.combine_recv_progress)
                    states = [struct.unpack_from('<4I', progress, 96*mb+64)
                              for mb in range(args.microbatches)]
                    if any(state == 4 for state, _, _, _ in states):
                        raise RuntimeError(f'combineRecv round-step failed: {states}')
                    if all(state == 3 for state, _, _, _ in states):
                        break
                    if (time.monotonic() > deadline or time.monotonic() > _work_deadline
                            or _txn_sweeps >= args.max_sweeps):
                        raise TimeoutError(f'fair return sweep did not close all slots: {states}, '
                                           f'txn_sweeps={_txn_sweeps} total_sweeps={sweeps} txn={txn}')
                    receive(owner, source_task, start=False)
                    sweeps += 1
                    _txn_sweeps += 1
                recv_stream.synchronize()
            send_complete.synchronize()
            _txn_times.append((txn, time.monotonic() - _t0))
            with torch.npu.stream(recv_stream):
                assert send_results[txn].cpu().tolist()[:2] == [2, 0], f'dispatchSend failed for txn {txn}'
                _pg = bytes_cpu(owner.combine_recv_progress)
                _pstate, _next_round, _last_bit, _perr = struct.unpack_from('<4I', _pg, 64)
                _led = bytes_cpu(owner.round_ledger)
                _l0 = struct.unpack_from('<I', _led, 0)[0]
                _l1 = struct.unpack_from('<I', _led, 64)[0]
                _txn_bits.append((int(_pstate), int(_last_bit), int(_next_round), int(_perr),
                                  int(_l0), int(_l1)))
                print('REUSE_TXN ' + json.dumps({'txn': txn, 'progress_state': int(_pstate),
                    'last_input_bit': int(_last_bit), 'next_round': int(_next_round),
                    'progress_error': int(_perr), 'ledger_bank_states': [int(_l0), int(_l1)],
                    'sweeps_total': sweeps}), flush=True)
                _xint = source_task.x_cpu.to(torch.float32).numpy()
                expected = torch.from_numpy(
                    expected_rows(_xint, route_table(capacity, txn),
                                  weight_table(capacity, txn))).to(torch.bfloat16)
                expected[~source_task.mask_cpu] = 0
                b = lengths[rank]
                actual = source_task.output.cpu()
                _ab = actual[:b].view(torch.int16).to(torch.int64) & 0xFFFF
                _eb = expected[:b].view(torch.int16).to(torch.int64) & 0xFFFF
                _oa = torch.where(_ab >= 0x8000, 0x8000 - (_ab & 0x7FFF), 0x8000 + _ab)
                _oe = torch.where(_eb >= 0x8000, 0x8000 - (_eb & 0x7FFF), 0x8000 + _eb)
                _bd = (_oa - _oe).abs()
                _ok = bool((_bd == 0).all())
                print('CHECK ' + json.dumps({'txn': txn, 'max_bd': int(_bd.max()),
                    'sum_a': round(float(actual[:b].float().sum()), 3),
                    'sum_e': round(float(expected[:b].float().sum()), 3)}), flush=True)
                print('TIMING ' + json.dumps({'txn': txn, 's': _txn_times[-1][1],
                    'warm': bool(txn < args.warmup), 'rank': rank,
                    'run_tag': args.run_tag, 'C': args.window}), flush=True)
                if not _ok:
                    ab = actual[:b].view(torch.int16).cpu().numpy()
                    eb = expected[:b].view(torch.int16).cpu().numpy()
                    dm = (ab != eb)
                    idxs = list(zip(*dm.nonzero()))[:16]
                    sample = [f'[{int(i)},{int(j)}] a=0x{int(ab[i,j])&0xffff:04x} e=0x{int(eb[i,j])&0xffff:04x}' for i, j in idxs]
                    print(f'DIAG reuse mismatch txn={txn} total_diff={int(dm.sum())} '
                          f'sum_a={float(actual[:b].float().sum()):.6f} sum_e={float(expected[:b].float().sum()):.6f} '
                          f'max_bd={int(_bd.max())} sample:', '; '.join(sample), flush=True)
                assert _ok, f'reuse weighted mismatch (strict) txn={txn} max_bd={int(_bd.max())}'
                assert torch.all(actual[b:] == -7), f'capacity tail modified for txn {txn}'
        _bits = [v[1] for v in _txn_bits]
        _expect_bits = [(((i + 1) * rounds) - 1) % 2 for i in range(len(_txn_bits))]
        assert all(v[0] == 3 and v[2] == rounds and v[3] == 0 for v in _txn_bits), f'progress not DONE: {_txn_bits}'
        assert _bits == _expect_bits, f'last_input_bit mismatch: {_bits} expected {_expect_bits}'
        assert all(v[4] == 0 and v[5] == 0 for v in _txn_bits), f'ledger banks not released: {[v[4:] for v in _txn_bits]}'
        print('REUSE_BITS ' + json.dumps({'last_input_bits': _bits, 'expected': _expect_bits,
            'progress_done': [v[0] for v in _txn_bits],
            'ledger_bank_states': [[v[4], v[5]] for v in _txn_bits]}), flush=True)
        _times = sorted(t for (i, t) in _txn_times if i >= args.warmup)
        _n = len(_times)
        if _n:
            _med = _times[_n // 2] if _n % 2 else 0.5 * (_times[_n // 2 - 1] + _times[_n // 2])
            _p90 = _times[min(_n - 1, int(round(0.9 * (_n - 1))))]
            print('TIMING_STATS ' + json.dumps({'n': _n, 'median_s': _med,
                'p90_s': _p90, 'max_s': _times[-1],
                'warmup': args.warmup, 'C': args.window, 'B': args.batch,
                'rank': rank, 'run_tag': args.run_tag}), flush=True)
    else:
        _recv_index = rank - len(sources)
        finished = 0
        deadline = _work_deadline
        expected_finished = (args.warmup + args.samples) * rounds * args.microbatches
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
                # Per-expert deterministic transform x*S[g]+C[g]; segments from the
                # per-expert aligned prefix sums (alignment=1 in prepare_recv).
                _psum = owner.psum_num_recv_tokens_per_expert.cpu().numpy().astype(np.int64)
                _unal = owner.num_unaligned_recv_tokens_per_expert.cpu().numpy().astype(np.int64)
                _nlocal = len(_psum)
                for _l in range(_nlocal):
                    _g = _recv_index * _nlocal + _l
                    _s0 = int(_psum[_l]) - int(_unal[_l])
                    _s1 = int(_psum[_l])
                    if _s1 > _s0:
                        _seg = owner.recv_x[_s0:_s1].float()
                        owner.combine_x[_s0:_s1].copy_(
                            (_seg * float(_S[_g]) + float(_C[_g])).to(torch.bfloat16))
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
    _teardown_left = _proc_t0 + args.timeout_seconds + 30 - time.monotonic()
    print('BUDGET ' + json.dumps({'rank': rank, 'role': _role,
        'work_used_s': time.monotonic() - _proc_t0,
        'teardown_left_s': _teardown_left}), flush=True)
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
    if _teardown_left > 0:
        _mem('function_done', owner)
        _stride_dir('function_done')
    if prof is not None:
        prof.stop()
        print(f'TRACE_EXPORT_RETURNED rank={rank} directory={trace_dir}', flush=True)
    if rank == 0:
        print(json.dumps(dict(status='FUNCTION_PASS', scope='actual five operators, per-expert deterministic transforms, timed transaction reuse',
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
        print(json.dumps(dict(status='RUN_PASS', scope='actual five operators, per-expert deterministic transforms, timed transaction reuse',
            lengths=lengths, window=args.window, rounds=rounds, q=args.q, precheck=args.precheck,
            microbatches=args.microbatches, round_step=args.round_step, sweeps_on_rank0=sweeps)))
    stage('main_return')


if __name__ == '__main__':
    main()
