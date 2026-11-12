"""CAM-port five-operator BF16 identity-expert bring-up; requires A5 and the CAM run package.

timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_cam.py

AFDBuffer still owns the device context, the transport registration and the
scheduler side (ffnScheduler/computeReady). The four communication operators are
the CAM custom ops (torch.ops.umdk_cam_op_lib.cam_deepep_*), launched with the
values camConfig() reports for this owner so the harness cannot drift from the
JIT-launched production path. The default milestone is single round
(round_step=0); multi-round stays available through --round-step.
Identity experts replace GMM numerics, not protocol.
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
import umdk_cam_op_lib  # noqa: F401 - registers umdk_cam_op_lib torch ops
from deep_ep.utils.envs import get_hccl_group_name
from ffn_transaction_runner import run_transaction_ffn

CAM = torch.ops.umdk_cam_op_lib


def bytes_cpu(tensor):
    return tensor.cpu().numpy().tobytes()


WEIGHT_SCHEMES = ('unit', 'nonunit', 'signed', 'zero01', 'mix')


def weight_vector(name, k):
    """Deterministic k weights; the k=2 spellings are the validated bring-up ones."""
    if name == 'unit':
        return (1.0 / k,) * k
    if name == 'nonunit':
        return (0.25,) * k
    if name == 'signed':
        return tuple(-0.5 if j % 2 == 0 else 1.5 for j in range(k))
    if name == 'zero01':
        return tuple(0.0 if j % 2 == 0 else 1.0 for j in range(k))
    return tuple(0.25 if j % 2 == 0 else 0.75 for j in range(k))


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


def receive(owner, cfg, group_name, source_task, status_i32, progress, *, start=False):
    """CAM combine_recv in place of owner.combineRecv; outputs are the owner buffers."""
    CAM.cam_deepep_combine_recv(
        owner.payload, source_task.task, owner.context, owner.control_buffer,
        owner.round_ledger, owner.combine_recv_workspace, source_task.mask,
        source_task.output, source_task.idx, source_task.weights, status_i32, progress,
        group_name,
        cfg['num_experts'], cfg['num_ranks'], cfg['num_topk'],
        cfg['round_step'], int(start), cfg['num_hidden'],
        cfg['num_token_capacity'], 8, cfg['return_progress_bytes'],
        cfg['timeout_cycles'], cfg['num_vec_cores'])


def send_dispatch(owner, cfg, group_name, source_task, sf, jetty, status_i32):
    """CAM dispatch_send in place of owner.dispatchSend."""
    CAM.cam_deepep_dispatch_send(
        source_task.x, sf, source_task.idx, source_task.weights, source_task.mask,
        source_task.task, owner.context, owner.control_buffer, owner.round_ledger,
        owner.workspace, jetty, status_i32, group_name,
        cfg['num_ranks'], cfg['num_experts'], cfg['num_topk'],
        cfg['num_tokens_per_tile'], 0, 0, cfg['num_sf_packs'],
        cfg['timeout_cycles'], 8, cfg['num_vec_cores'])


def recv_dispatch(owner, cfg, group_name, recv_sf):
    """CAM dispatch_recv in place of owner.dispatchRecv."""
    CAM.cam_deepep_dispatch_recv(
        owner.payload, owner.context, owner.active, owner.execution_state,
        owner.recv_workspace, owner.control_buffer,
        owner.recv_x, recv_sf, owner.recv_src_metadata, group_name,
        cfg['num_ranks'], cfg['num_experts'], cfg['num_topk'],
        cfg['num_source_lanes'], cfg['num_layers'],
        cfg['recv_q'], cfg['recv_q_alloc'], cfg['recv_alignment'], cfg['recv_tile'],
        cfg['recv_rank'], 0, 0,
        cfg['payload_bytes'], cfg['control_bytes'], cfg['context_bytes'],
        cfg['num_hidden'], cfg['num_sf_packs'], cfg['num_vec_cores'])


def send_combine(owner, cfg, group_name, jetty):
    """CAM combine_send in place of owner.combineSend; completion is an owner buffer."""
    CAM.cam_deepep_combine_send(
        owner.combine_x, owner.recv_src_metadata, owner.context, owner.active,
        owner.control_buffer, owner.execution_state, owner.combine_send_workspace,
        jetty, owner.completion, group_name,
        cfg['num_src_ranks'], cfg['num_ranks'], cfg['num_experts'], cfg['num_topk'],
        cfg['num_source_lanes'], cfg['num_layers'], cfg['recv_q_alloc'],
        cfg['combine_tile'], cfg['recv_rank'], cfg['timeout_cycles'],
        cfg['completion_bytes'], cfg['num_vec_cores'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=41)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=128,
                        help='BF16 hidden size (elements); matrix cases use 7168')
    parser.add_argument('--topk', type=int, default=2,
                        help='routed experts per token; matrix cases use 1/2/4/6/8')
    parser.add_argument('--experts', type=int, default=0,
                        help='total experts over all F ranks; 0 = two per F rank')
    parser.add_argument('--q', default='auto',
                        help='second-layer split capacity (max expanded rows per slice); '
                             '"auto" = ceil(1.2 * num_sources * window * topk / num_receivers), '
                             'capped at 32768; an integer reproduces historical runs')
    parser.add_argument('--short-second', type=int)
    parser.add_argument('--ffn-transaction', action='store_true', help='fixed selection and whole-expert slices')
    parser.add_argument('--transaction-cohort', action='store_true',
                        help='diagnostic: wait for all single-round lanes before the first selection')
    parser.add_argument('--transaction-gmm', action='store_true', help='BF16 grouped GEMM diagnostic (H=128)')
    parser.add_argument('--transaction-layers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--transaction-alignment', type=int, default=1)
    parser.add_argument('--backend-max-rows', type=int, default=0,
                        help='explicit padded-row capacity of the selected FFN backend')
    parser.add_argument('--precheck', action='store_true')
    parser.add_argument('--all-masked', action='store_true')
    parser.add_argument('--microbatches', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--round-step', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--max-sweeps', type=int, default=100000)
    parser.add_argument('--weights', choices=WEIGHT_SCHEMES, default='unit')
    parser.add_argument('--routes', choices=('both01', 'balanced', 'hotspot'), default='both01')
    parser.add_argument('--num-vec-cores', type=int, default=1,
                        help='AIV cores used by dispatch/combine; 0 = all platform AIVs; '
                             'the pipeline needs 2*n <= total AIVs (dispatchSend and '
                             'combineRecv co-reside), so the practical max is half')
    parser.add_argument('--profile-trace', action='store_true')
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--neg-qalign', action='store_true',
                        help='F ranks call dispatch_recv with q_alloc < alignment and expect a host '
                             'rejection; A ranks only join the barriers')
    parser.add_argument('--a-budget-mib', type=int, default=8192)
    parser.add_argument('--f-budget-mib', type=int, default=8192)
    parser.add_argument('--reserve-mib', type=int, default=256)
    args = parser.parse_args()
    if args.batch < 0 or not 1 <= args.window <= 32768:
        parser.error('batch >= 0, window in [1,32768] required')
    if args.hidden <= 0 or not 1 <= args.topk <= 32:
        parser.error('hidden > 0 and topk in [1,32] required')
    if args.experts < 0:
        parser.error('experts must be nonnegative')
    if args.short_second is not None and not 0 <= args.short_second <= args.batch:
        parser.error('short-second must be within [0,batch]')
    if args.timeout_seconds <= 0 or args.max_sweeps <= 0:
        parser.error('timeout-seconds and max-sweeps must be positive')
    if args.microbatches > 1 and args.batch > args.window and not args.round_step:
        parser.error('multiple slots with multiple rounds require --round-step')
    if args.num_vec_cores < 0:
        parser.error('num-vec-cores must be nonnegative')
    if args.a_budget_mib <= 0 or args.f_budget_mib <= 0 or args.reserve_mib <= 0:
        parser.error('budgets and reserve must be positive')
    if args.transaction_gmm and (not args.ffn_transaction or args.hidden != 128):
        parser.error('--transaction-gmm requires --ffn-transaction and H=128')
    if args.ffn_transaction and (args.backend_max_rows <= 0 or args.neg_qalign or
            args.transaction_alignment <= 0 or args.transaction_alignment & (args.transaction_alignment - 1)):
        parser.error('transaction mode requires a positive backend limit and power-of-two alignment')
    if args.transaction_cohort and (not args.ffn_transaction or args.batch > args.window or args.microbatches != 1):
        parser.error('--transaction-cohort requires one transaction-mode MB and a single communication round')
    if not args.ffn_transaction and args.transaction_layers != 1:
        parser.error('--transaction-layers=2 requires --ffn-transaction')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=args.timeout_seconds))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (2, 4, 8):
        raise ValueError('2 ranks = one A/one F; 4 = two A/two F; 8 = four A/four F')
    num_sources = world // 2
    sources, receivers = list(range(num_sources)), list(range(num_sources, world))
    if args.transaction_layers > num_sources:
        parser.error('need at least two A ranks to exercise two layers')
    dp_ids = ([src % args.transaction_layers for src in sources] if args.ffn_transaction else [0] * num_sources)
    layer_ids = list(range(args.transaction_layers))
    lane_specs = [[src, mb, dp_ids[src], src, 0, args.window]
                  for src in sources for mb in range(args.microbatches)]
    lengths = [args.batch] * num_sources
    if num_sources == 2 and args.short_second is not None:
        lengths[1] = args.short_second
    rounds = max(1, max((b+args.window-1)//args.window for b in lengths))
    # Second-layer split capacity q (max expanded rows = (token, local expert)
    # routes per scheduler slice). A balanced round puts at most
    # num_sources * window * topk / num_receivers rows on one F rank, so
    # ceil(1.2 * that) leaves headroom for imbalance and keeps essentially
    # every round single-slice. The tiling domain caps q at 32768; batches
    # above the formula cap fall back to slicing instead of failing.
    Q_MAX = args.backend_max_rows if args.ffn_transaction else 32768
    group_bound = len(lane_specs) * args.window
    if args.q == 'auto' and args.ffn_transaction:
        q_mode = 'whole_expert_bound'
        q_raw = q = group_bound
        q_clamped = 0
    elif args.q == 'auto':
        q_mode = 'auto'
        q_raw = -(-12 * num_sources * args.window * args.topk // (10 * len(receivers)))
        q = max(1, min(Q_MAX, q_raw))
        q_clamped = int(q_raw > Q_MAX)
    else:
        q_mode = 'explicit'
        try:
            q = int(args.q)
        except ValueError:
            parser.error('--q must be "auto" or a positive integer')
        q_clamped = 0
        if not 1 <= q <= Q_MAX:
            parser.error(f'explicit q must be in [1,{Q_MAX}]')
    alignment = args.transaction_alignment if args.ffn_transaction else 1
    qa = (q + alignment - 1) // alignment * alignment
    if args.ffn_transaction and (q < group_bound or qa > Q_MAX):
        parser.error(f'CAPACITY requires Q >= {group_bound} and Q_alloc={qa} <= backend={Q_MAX}')
    capacity, hidden, k = max(1, args.batch), args.hidden, args.topk
    experts = args.experts if args.experts else 2 * len(receivers)
    if experts > 512 or experts < k or experts % len(receivers):
        parser.error('experts in [topk,512] and divisible by the F rank count required')
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    def _align(n, a):
        return (n + a - 1) // a * a

    _role = 'A' if rank in sources else 'F'
    # One AIV count for the owner and every CAM attr: it sizes the (peer, AIV)
    # channel table, channel_stride and the per-core workspace, so overriding
    # only the op attr walks the table out of bounds (the 507035 family).
    _nvc = deep_ep._C.get_num_vec_cores() if args.num_vec_cores == 0 else args.num_vec_cores
    if _nvc <= 0:
        parser.error('--num-vec-cores resolved to 0')
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
        + _align(_num_lanes * _nffn * 64, 32) + _align(len(layer_ids) * 4, 32))
    _ctx = _ctx_base + 32 + 32 * len(sources) + 32 * _num_lanes
    _c = args.window
    _slots = args.microbatches
    _j = (_c + 31) // 32
    _ws_upper = (_align(4 * _c * k, 32) + _align(_c * _mstride, 32)
        + 2 * _align(4 * _j * (_nffn + experts), 32) + _align(4 * _c * k, 32)
        + 2 * _align(4 * experts, 32) + _align(4 * _nffn, 32) + 32 * _slots + 256)
    _ledger = 128 * _slots
    # ChannelHandle is uint64_t (hcomm_res_defs.h); world>2 keeps one channel
    # per (peer, AIV) and channel_stride == num_vec_cores (hccl.hpp grouped path).
    _ccap = (world - 1) * _nvc if world > 2 else max(_nvc, world - 1)
    _tables = 2 * _ccap * 8
    if _role == 'A':
        _prep = (256 + _align(32 * _nvc, 32)) + 32 + (96 * _slots if args.round_step else 0)
    else:
        _active = 64 + 64 * _num_lanes + 32 * _num_lanes * _nf
        _execution = 32 + _align(4 * _num_lanes, 32)
        _drecv = 32 * _nvc + 256 + 2 * _align(4 * min(_num_lanes * _nf, q), 32)
        if args.ffn_transaction:
            _active += 32 + 64 * _num_lanes + _align(4 * _num_lanes * _nf, 32)
            _active += _align(4 * len(layer_ids) * _nf, 32) + 16 * _num_lanes
        _add = _active + _execution + _drecv + qa * (2 * hidden + 16)
        _sched = 32 + 96 * _num_lanes + 2 * _align(4 * _num_lanes * _nf, 32)
        _compl = 32 + 64 * _num_lanes + _align(4 * _num_lanes * _nf, 32)
        _prep = _add + _sched + _compl + 8 * _nf + qa * hidden * 2 + (32 * _nvc + 256)
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
        _so = deep_ep._C.__file__
        _h = hashlib.sha256()
        with open(_so, 'rb') as _f:
            for _chunk in iter(lambda: _f.read(1 << 20), b''):
                _h.update(_chunk)
        _so_desc = {'path': _so, 'bytes': os.path.getsize(_so), 'sha256': _h.hexdigest()}
    except Exception as _exc:
        _so_desc = {'error': repr(_exc)[:200]}
    try:
        _cam_so = umdk_cam_op_lib.__file__
        _h = hashlib.sha256()
        with open(_cam_so, 'rb') as _f:
            for _chunk in iter(lambda: _f.read(1 << 20), b''):
                _h.update(_chunk)
        _cam_desc = {'path': _cam_so, 'bytes': os.path.getsize(_cam_so), 'sha256': _h.hexdigest()}
    except Exception as _exc:
        _cam_desc = {'error': repr(_exc)[:200]}
    print('RUNCFG ' + json.dumps({'rank': rank, 'argv': sys.argv, 'run_tag': args.run_tag,
        'B': args.batch,
        'C': args.window, 'rounds': rounds, 'Q': q, 'Q_mode': q_mode,
        'Q_raw': q_raw if q_mode == 'auto' else q, 'Q_clamped': q_clamped,
        'MB': args.microbatches,
        'weights': args.weights, 'routes': args.routes, 'so': _so_desc,
        'cam_so': _cam_desc}), flush=True)

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
        lane_specs, layer_ids,
        hidden, 0, k, experts, capacity, False, _nvc, 120, args.timeout_seconds,
        _budget, _reserve)
    _mem('constructed', owner)
    # Launch parameters come from the owner itself; only num_hidden is derived
    # (camConfig reports the bf16 byte width, CAM attrs take elements).
    _cfg = owner.camConfig()
    _cfg['num_hidden'] = _cfg['num_hidden_bytes'] // 2
    print('CAMCFG ' + json.dumps({'rank': rank, 'role': _role,
        'cfg': {key: int(value) for key, value in _cfg.items()}}), flush=True)
    _jetty = owner.jetty_ptrs
    print('CAMJETTY ' + json.dumps({'rank': rank, 'n': int(_jetty.numel()),
        'dtype': str(_jetty.dtype)}), flush=True)
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
        # prepare_combine_recv fixes round_step/return_progress_bytes; refresh so the
        # CAM attrs and the progress tensor follow the real values.
        _cfg.update({key: int(value) for key, value in owner.camConfig().items()})
        _cfg['num_hidden'] = _cfg['num_hidden_bytes'] // 2
        # CAM combine_recv needs a status tensor of the CAM dtype (int32); the
        # owner keeps the same storage behind a uint32 view.
        status_i32 = owner.recv_status.view(torch.int32)
        progress = (owner.combine_recv_progress if _cfg['round_step']
                    else torch.empty(0, dtype=torch.uint8, device=owner.context.device))
        # bf16 has no scale factors; dispatch_send still takes a tensor, which is
        # never read while num_sf_packs is zero.
        sf = torch.zeros(1, dtype=torch.int16, device=owner.context.device)
        source_tasks = []
        for mb in range(args.microbatches):
            # Different, exactly representable BF16 data detects cross-slot output confusion.
            x_cpu = ((torch.arange(capacity*hidden).reshape(capacity, hidden) + 3*mb + rank) % 17).to(torch.bfloat16)
            x = x_cpu.npu()
            # k adjacent ids per group: k=2 reproduces the validated identity pairs,
            # and cycling groups still spreads the batch over every F rank.
            groups = [tuple((g * k + j) % experts for j in range(k))
                      for g in range(max(1, experts // k))]
            if args.routes == 'balanced':
                sel = [groups[t % len(groups)] for t in range(capacity)]
            elif args.routes == 'hotspot':
                sel = [groups[0] if (len(groups) == 1 or t % 4 != 3) else groups[1]
                       for t in range(capacity)]
            else:
                sel = [groups[0]] * capacity
            idx = torch.tensor(sel, dtype=torch.int64).npu()
            weights = torch.tensor(weight_vector(args.weights, k), dtype=torch.float32)
            weights = weights.repeat(capacity, 1).npu()
            mask_cpu = torch.arange(capacity) % 5 != 0
            if args.all_masked:
                mask_cpu[:] = False
            mask = mask_cpu.npu()
            # Task ABI word must match AFD kAbiVersion (context.hpp; 2 with the channel_stride layout).
            task = torch.from_numpy(np.frombuffer(struct.pack('<16I',
                2, rank, mb, mb, dp_ids[rank], lengths[rank], rounds, *([0]*9)), dtype=np.uint8).copy()).npu()
            source_tasks.append(SourceTask(x_cpu, mask_cpu, x, idx, weights, mask, task, torch.full_like(x, -7)))
        send_stream, recv_stream = torch.npu.Stream(), torch.npu.Stream()
        send_complete = torch.npu.Event()
        send_results = [torch.empty_like(owner.send_status) for _ in source_tasks]
    else:
        if args.ffn_transaction:
            owner.prepare_recv(q, qa, alignment, ffn_transaction=True, backend_max_rows=args.backend_max_rows)
        else:
            owner.prepare_recv(q, qa, alignment)
        # prepare_recv fixes recv_q/q_alloc/alignment/tile; refresh so the CAM
        # attrs carry the real values instead of the pre-prepare zero defaults.
        _cfg.update({key: int(value) for key, value in owner.camConfig().items()})
        _cfg['num_hidden'] = _cfg['num_hidden_bytes'] // 2
        owner.prepare_ffn(args.precheck)
        # bf16: the CAM dispatch_recv output is an empty [rows, 0] tensor.
        recv_sf = torch.empty(_cfg['recv_q_alloc'], _cfg['num_sf_packs'],
                              dtype=torch.int16, device=owner.context.device)
    _mem('prepared', owner)
    if args.neg_qalign:
        # Negative path: q_alloc below one alignment has no padding headroom and
        # must be rejected by host tiling instead of stalling on device.
        _ok = 1
        dist.barrier()
        if rank not in sources:
            _bad_alloc, _bad_align = 64, 128
            _ok = 0
            try:
                CAM.cam_deepep_dispatch_recv(
                    owner.payload, owner.context, owner.active, owner.execution_state,
                    owner.recv_workspace, owner.control_buffer,
                    owner.recv_x, recv_sf, owner.recv_src_metadata, group_name,
                    _cfg['num_ranks'], _cfg['num_experts'], _cfg['num_topk'],
                    _cfg['num_source_lanes'], _cfg['num_layers'],
                    _bad_alloc, _bad_alloc, _bad_align, _cfg['recv_tile'],
                    _cfg['recv_rank'], 0, 0,
                    _cfg['payload_bytes'], _cfg['control_bytes'], _cfg['context_bytes'],
                    _cfg['num_hidden'], _cfg['num_sf_packs'], _cfg['num_vec_cores'])
            except RuntimeError as _exc:
                _ok = int('max_padded_tokens' in str(_exc))
                _msg = str(_exc)[:300]
            else:
                _msg = 'accepted'
            print('NEG_QALIGN ' + json.dumps({'rank': rank, 'ok': _ok,
                'q_alloc': _bad_alloc, 'alignment': _bad_align, 'err': _msg}), flush=True)
        dist.barrier()
        owner.destroy()
        dist.barrier()
        dist.destroy_process_group()
        if not _ok:
            raise SystemExit(9)
        return
    # Prewarm the AFD-side JIT kernels (scheduler/compute) serially in world rank
    # order. The four CAM operators are prebuilt in the run package and warm on
    # their first real launch, so they are not prewarmed here.
    for warm_rank in range(world):
        if rank == warm_rank:
            if rank in sources:
                pass
            else:
                owner.compileFFNSchedulerOnly()
                if args.ffn_transaction:
                    owner.compileFFNPollOnly()
                owner.compileComputeReadyOnly()
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
        send_status_i32 = owner.send_status.view(torch.int32)
        with torch.npu.stream(send_stream):
            for source_task, send_result in zip(source_tasks, send_results):
                send_dispatch(owner, _cfg, group_name, source_task, sf, _jetty, send_status_i32)
                send_result.copy_(owner.send_status)
            send_complete.record(send_stream)
        with torch.npu.stream(recv_stream):
            if args.round_step:
                # Initialization executes exactly once for this transaction set.
                for source_task in source_tasks:
                    receive(owner, _cfg, group_name, source_task, status_i32, progress, start=True)
                deadline = time.monotonic() + args.timeout_seconds
                while True:
                    # Host observation is queued on recv_stream, never behind the live send.
                    # This is deliberately bring-up code, not a graph/performance loop.
                    progress_bytes = bytes_cpu(owner.combine_recv_progress)
                    states = [struct.unpack_from('<4I', progress_bytes, 96*mb+64)
                              for mb in range(args.microbatches)]
                    if any(state == 4 for state, _, _, _ in states):
                        raise RuntimeError(f'combineRecv round-step failed: {states}')
                    if all(state == 3 for state, _, _, _ in states):
                        break
                    if time.monotonic() > deadline or sweeps >= args.max_sweeps:
                        raise TimeoutError(f'fair return sweep did not close all slots: {states}, sweeps={sweeps}')
                    for source_task in source_tasks:
                        receive(owner, _cfg, group_name, source_task, status_i32, progress)
                    sweeps += 1
            else:
                # The parser and device admission forbid unsafe multi-slot/multi-round use.
                for source_task in source_tasks:
                    receive(owner, _cfg, group_name, source_task, status_i32, progress)
                    status = owner.recv_status.cpu().tolist()
                    assert status[:2] == [2, 0], f'combineRecv failed: {status}'
            # The last recv-stream copy/event and Epub must both finish before Etxn reuse.
            recv_stream.synchronize()
        send_complete.synchronize()
        with torch.npu.stream(recv_stream):
            for mb, (source_task, result) in enumerate(zip(source_tasks, send_results)):
                assert result.cpu().tolist()[:2] == [2, 0], f'dispatchSend failed for microbatch {mb}'
                xf = (source_task.x_cpu * (dp_ids[rank] + 1)).to(torch.float32)
                acc = torch.zeros_like(xf)
                for wv in weight_vector(args.weights, k):
                    acc = acc + xf * wv  # FP32 fold in k order, one BF16 round at the end
                expected = acc.to(torch.bfloat16)
                expected[~source_task.mask_cpu] = 0
                b = lengths[rank]
                actual = source_task.output.cpu()
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
                        for kk in range(min(2, k)):
                            off=ret_off+(1*k+kk)*stride
                            vals=struct.unpack_from('<8H', pay, off)
                            print(f'DIAGRAW t=1 k={kk} off={off} first8={[hex(v) for v in vals]}', flush=True)
                        print(f'DIAGW weights_host={source_task.weights[0].tolist()}', flush=True)
                    except Exception as exc:
                        print('DIAGRAW failed', repr(exc)[:400], flush=True)
                assert torch.equal(actual[:b], expected[:b]), f'weighted return mismatch for microbatch {mb}'
                assert torch.all(actual[b:] == -7), f'capacity tail modified for microbatch {mb}'
    elif args.ffn_transaction:
        run_transaction_ffn(
            owner, _cfg, rank=rank, lane_specs=lane_specs, layer_ids=layer_ids,
            expected_rounds=rounds * args.microbatches * len(set(dp_ids)), timeout=args.timeout_seconds,
            recv=lambda: recv_dispatch(owner, _cfg, group_name, recv_sf),
            send=lambda: send_combine(owner, _cfg, group_name, _jetty),
            grouped_gemm=args.transaction_gmm, wait_first_cohort=args.transaction_cohort)
        torch.npu.synchronize()
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
            recv_dispatch(owner, _cfg, group_name, recv_sf)
            if os.environ.get('CAM_DIAG_ACTIVE') == '1':
                print('DIAGPOSTRECV exec=' + str(struct.unpack_from('<8I', bytes_cpu(owner.execution_state)))
                      + ' comp=' + str(struct.unpack_from('<8I', bytes_cpu(owner.completion))), flush=True)
            # Actual identity-expert result; no manual completion or control writes.
            if header[1] == 1:
                owner.combine_x.copy_(owner.recv_x)
            owner.computeReady()
            if os.environ.get('CAM_DIAG_ACTIVE') == '1':
                print('DIAGPRE active=' + str(struct.unpack_from('<10I3Q', bytes_cpu(owner.active)))
                      + ' exec=' + str(struct.unpack_from('<8I', bytes_cpu(owner.execution_state)))
                      + ' comp=' + str(struct.unpack_from('<8I', bytes_cpu(owner.completion))), flush=True)
            send_combine(owner, _cfg, group_name, _jetty)
            completed = bytes_cpu(owner.completion)
            state, error, entries = struct.unpack_from('<3I', completed)
            if not (state == 2 and error == 0):
                exec_words = struct.unpack_from('<8I', bytes_cpu(owner.execution_state))
                print(f'DIAGCOMBO state={state} error={error} entries={entries} exec={exec_words}', flush=True)
            assert state == 2 and error == 0, \
                f'combineSend did not commit: state={state} error={error} entries={entries}'
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
        print(json.dumps(dict(status='FUNCTION_PASS', scope='CAM four operators, identity experts, no graph',
            lengths=lengths, window=args.window, rounds=rounds, q=q, q_mode=q_mode, precheck=args.precheck,
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
        print(json.dumps(dict(status='RUN_PASS', scope='CAM four operators, identity experts, no graph',
            lengths=lengths, window=args.window, rounds=rounds, q=q, q_mode=q_mode, precheck=args.precheck,
            microbatches=args.microbatches, round_step=args.round_step, sweeps_on_rank0=sweeps)))
    stage('main_return')


if __name__ == '__main__':
    main()
