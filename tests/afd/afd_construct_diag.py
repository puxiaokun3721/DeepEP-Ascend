"""AFDBuffer-construction-only diagnostic for the jetty-layer0-default-sq tree.

Replicates the s8 small-shape constructor call exactly and stops there: one
AFDBuffer per rank (the constructor creates the payload context and then the
control context; up to two contexts per rank), no prepare_recv, no
dispatchSend, no communication kernels.

v2 (reviewer-directed error propagation fix): binding failures exit nonzero
immediately; any stage failure yields a nonzero final exit; a failed
owner.destroy() does not enter a further cross-rank barrier; owner destroy,
final barrier and process-group destroy are recorded explicitly, with
process-group destruction UNKNOWN unless it explicitly succeeded.
"""
import hashlib
import json
import os
import sys
import time
from datetime import timedelta


def emit(obj):
    print('CONSTRUCT_DIAG ' + json.dumps(obj, sort_keys=True), flush=True)


def main():
    out = {'rank_env': int(os.environ.get('RANK', -1)),
           'local_rank': int(os.environ.get('LOCAL_RANK', -1)),
           'pid': os.getpid()}
    # Stage 1: import binding. A failure here must stop before any device work.
    try:
        import torch
        import torch_npu  # noqa: F401  Registers the NPU backend.
        import torch.distributed as dist
        import deep_ep
        from deep_ep.utils.envs import get_hccl_group_name
        from gen_data import generate
        out['torch_npu'] = getattr(torch_npu, '__version__', None)
        out['deep_ep_file'] = deep_ep.__file__
        import deep_ep._C as _c
        so = _c.__file__
        h = hashlib.sha256()
        with open(so, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        out['so_file'] = so
        out['so_sha256'] = h.hexdigest()
    except BaseException as exc:
        out['stage'] = 'binding'
        out['ok'] = False
        out['error_type'] = type(exc).__name__
        out['error'] = str(exc)[:6000]
        emit(out)
        sys.exit(3)
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=180))
    out['world'] = dist.get_world_size()
    rank = dist.get_rank()
    out['rank'] = rank
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    out['group_name_len'] = len(group_name)
    # s8 small shape: --batch 33 defaults; payload [0], control [1]; one
    # payload region; owner rank list [0]; hidden 128; k 8; experts 8.
    hidden, k, experts, capacity = 128, 8, 8, 33
    fp8, cores = False, 1
    data = generate(batch=capacity, hidden=hidden, k=k, experts=experts,
                    ranks=1, fp8=fp8)
    sf_packs = data['sf_bytes'].shape[1] // 2
    out['sf_packs'] = sf_packs
    out['ctor_args'] = {'rank': rank, 'payload': [0], 'control': [1],
                        'regions': [[0, 0, 0, 0, 0, capacity]], 'owner': [0],
                        'hidden': hidden, 'sf_packs': sf_packs, 'k': k,
                        'experts': experts, 'capacity': capacity,
                        'fp8': fp8, 'cores': cores}
    owner = None
    t0 = time.monotonic()
    try:
        owner = deep_ep.AFDBuffer(rank, group_name, [0], [1],
            [[0, 0, 0, 0, 0, capacity]], [0], hidden, sf_packs, k, experts,
            capacity, fp8, cores, 60, 30, 8 * 1024 ** 3, 256 * 1024 ** 2)
        out['ok'] = True
        out['construct_ms'] = round((time.monotonic() - t0) * 1000.0, 1)
        emit(out)
    except BaseException as exc:
        out['stage'] = 'construct'
        out['ok'] = False
        out['construct_ms'] = round((time.monotonic() - t0) * 1000.0, 1)
        out['error_type'] = type(exc).__name__
        out['error'] = str(exc)[:6000]
        out['cleanup'] = {'owner_destroy': 'NOT_ATTEMPTED_INCOMPLETE_OBJECT',
                          'pg_barrier': 'NOT_ATTEMPTED', 'pg_destroy': 'UNKNOWN'}
        emit(out)
        sys.exit(1)
    try:
        dist.barrier()
        out['barrier_ok'] = True
    except BaseException as exc:
        out['stage'] = 'barrier'
        out['barrier_ok'] = False
        out['barrier_error'] = repr(exc)[:1500]
        out['cleanup'] = {'owner_destroy': 'NOT_ATTEMPTED', 'pg_barrier': 'FAILED',
                          'pg_destroy': 'UNKNOWN'}
        emit(out)
        sys.exit(2)
    emit(out)
    try:
        owner.destroy()
        out['destroy_ok'] = True
    except BaseException as exc:
        out['stage'] = 'owner_destroy'
        out['destroy_ok'] = False
        out['destroy_error'] = repr(exc)[:2000]
        out['cleanup'] = {'owner_destroy': 'FAILED', 'pg_barrier': 'NOT_ATTEMPTED',
                          'pg_destroy': 'UNKNOWN'}
        emit(out)
        sys.exit(4)
    out['stage'] = 'final_barrier'
    try:
        dist.barrier()
        out['final_barrier_ok'] = True
    except BaseException as exc:
        out['final_barrier_ok'] = False
        out['final_barrier_error'] = repr(exc)[:1500]
        out['cleanup'] = {'owner_destroy': 'OK', 'pg_barrier': 'FAILED',
                          'pg_destroy': 'UNKNOWN'}
        emit(out)
        sys.exit(5)
    out['stage'] = 'pg_destroy'
    try:
        dist.destroy_process_group()
        out['pg_destroy_ok'] = True
    except BaseException as exc:
        out['pg_destroy_ok'] = False
        out['pg_destroy_error'] = repr(exc)[:1500]
        out['cleanup'] = {'owner_destroy': 'OK', 'pg_barrier': 'OK',
                          'pg_destroy': 'FAILED'}
        emit(out)
        sys.exit(6)
    out['cleanup'] = {'owner_destroy': 'OK', 'pg_barrier': 'OK', 'pg_destroy': 'OK'}
    emit(out)
    print('CONSTRUCT_DIAG_DONE rank=%d' % rank, flush=True)


if __name__ == '__main__':
    main()
