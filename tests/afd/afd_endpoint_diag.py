"""AFDBuffer world-4 constructor-only endpoint diagnostic (reviewer-directed).

Runs the exact pipeline-shape constructor (2 sources + 2 receivers, batch 13,
window 16, q 7, one microbatch; identity experts) and stops at construction.
The instrumented hccl.hpp prints per-peer endpoint comparison scalars before
the original assertion. No prepare_*, no communication kernels. If the
construction unexpectedly succeeds, destroy only and exit.
"""
import hashlib
import json
import os
import sys
import time
from datetime import timedelta


def emit(obj):
    print('ENDPOINT_DIAG ' + json.dumps(obj, sort_keys=True), flush=True)


def main():
    out = {'rank_env': int(os.environ.get('RANK', -1)),
           'local_rank': int(os.environ.get('LOCAL_RANK', -1)),
           'pid': os.getpid()}
    try:
        import torch
        import torch_npu  # noqa: F401  Registers the NPU backend.
        import torch.distributed as dist
        import deep_ep
        from deep_ep.utils.envs import get_hccl_group_name
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
        out.update({'stage': 'binding', 'ok': False,
                    'error_type': type(exc).__name__, 'error': str(exc)[:6000]})
        emit(out)
        sys.exit(3)
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=300))
    out['world'] = dist.get_world_size()
    rank = dist.get_rank()
    out['rank'] = rank
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    world = out['world']
    if world != 4:
        out.update({'stage': 'world', 'ok': False, 'error': 'world=%d != 4' % world})
        emit(out)
        sys.exit(3)
    sources, receivers = [0, 1], [2, 3]
    hidden, sf_packs, k = 128, 0, 2
    experts, capacity, window = 4, 13, 16
    channel_specs = [[a, mb, 0, a, 0, window] for a in sources for mb in range(1)]
    out['ctor_args'] = {'rank': rank, 'sources': sources, 'receivers': receivers,
                        'channel_specs': channel_specs, 'owner': [0],
                        'hidden': hidden, 'sf_packs': sf_packs, 'k': k,
                        'experts': experts, 'capacity': capacity, 'fp8': False,
                        'cores': 1, 'timeout': 120, 'timeout_seconds': 300,
                        'hbm': 8 * 1024 ** 3, 'ub': 256 * 1024 ** 2}
    owner = None
    t0 = time.monotonic()
    try:
        owner = deep_ep.AFDBuffer(rank, group_name, sources, receivers,
            channel_specs, [0], hidden, sf_packs, k, experts, capacity, False, 1,
            120, 300, 8 * 1024 ** 3, 256 * 1024 ** 2)
        out['ok'] = True
        out['construct_ms'] = round((time.monotonic() - t0) * 1000.0, 1)
        emit(out)
    except BaseException as exc:
        out.update({'stage': 'construct', 'ok': False,
                    'construct_ms': round((time.monotonic() - t0) * 1000.0, 1),
                    'error_type': type(exc).__name__, 'error': str(exc)[:6000],
                    'cleanup': {'owner_destroy': 'NOT_ATTEMPTED_INCOMPLETE_OBJECT',
                                'pg_barrier': 'NOT_ATTEMPTED',
                                'pg_destroy': 'UNKNOWN'}})
        emit(out)
        sys.exit(1)
    try:
        dist.barrier()
        out['barrier_ok'] = True
    except BaseException as exc:
        out.update({'stage': 'barrier', 'barrier_ok': False,
                    'barrier_error': repr(exc)[:1500],
                    'cleanup': {'owner_destroy': 'NOT_ATTEMPTED',
                                'pg_barrier': 'FAILED', 'pg_destroy': 'UNKNOWN'}})
        emit(out)
        sys.exit(2)
    try:
        owner.destroy()
        out['destroy_ok'] = True
    except BaseException as exc:
        out.update({'stage': 'owner_destroy', 'destroy_ok': False,
                    'destroy_error': repr(exc)[:2000],
                    'cleanup': {'owner_destroy': 'FAILED',
                                'pg_barrier': 'NOT_ATTEMPTED', 'pg_destroy': 'UNKNOWN'}})
        emit(out)
        sys.exit(4)
    out['stage'] = 'final_barrier'
    try:
        dist.barrier()
        out['final_barrier_ok'] = True
    except BaseException as exc:
        out.update({'final_barrier_ok': False, 'final_barrier_error': repr(exc)[:1500],
                    'cleanup': {'owner_destroy': 'OK', 'pg_barrier': 'FAILED',
                                'pg_destroy': 'UNKNOWN'}})
        emit(out)
        sys.exit(5)
    try:
        dist.destroy_process_group()
        out['pg_destroy_ok'] = True
    except BaseException as exc:
        out.update({'pg_destroy_ok': False, 'pg_destroy_error': repr(exc)[:1500],
                    'cleanup': {'owner_destroy': 'OK', 'pg_barrier': 'OK',
                                'pg_destroy': 'FAILED'}})
        emit(out)
        sys.exit(6)
    out['cleanup'] = {'owner_destroy': 'OK', 'pg_barrier': 'OK', 'pg_destroy': 'OK'}
    emit(out)
    print('ENDPOINT_DIAG_DONE rank=%d' % rank, flush=True)


if __name__ == '__main__':
    main()
