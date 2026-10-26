"""Compile-only preparer for combineSend/combineRecv production TUs.

Builds one AFDBuffer per rank with the same s8 small shape (approved
construction-only), then rank0 compiles both production TUs through
compile_without_load (kernel.asc -> bisheng -> ld.lld): no driver load of the
combine kernels and no kernel launch. The two operators are independent: a
failure of one is recorded and the other is still attempted; first error keeps
its full diagnostics. Any failure yields a nonzero exit.
"""
import hashlib
import json
import os
import sys
import time
from datetime import timedelta


def emit(obj):
    print('COMBINE_COMPILE ' + json.dumps(obj, sort_keys=True), flush=True)


def main():
    out = {'rank_env': int(os.environ.get('RANK', -1)),
           'local_rank': int(os.environ.get('LOCAL_RANK', -1)),
           'pid': os.getpid()}
    try:
        import torch
        import torch_npu  # noqa: F401
        import torch.distributed as dist
        import deep_ep
        from deep_ep.utils.envs import get_hccl_group_name
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
        out['error'] = repr(exc)[:20000]
        emit(out)
        sys.exit(3)
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    out['rank'] = rank
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    owner = None
    t0 = time.monotonic()
    try:
        owner = deep_ep.AFDBuffer(rank, group_name, [0], [1],
            [[0, 0, 0, 0, 0, 33]], [0], 128, 0, 8, 8,
            33, False, 1, 60, 30, 8 * 1024 ** 3, 256 * 1024 ** 2)
        out['construct_ms'] = round((time.monotonic() - t0) * 1000.0, 1)
    except BaseException as exc:
        out['stage'] = 'construct'
        out['ok'] = False
        out['error'] = repr(exc)[:20000]
        emit(out)
        sys.exit(1)
    results = {}
    prep_error = None
    # Role-correct wiring (entry only, no production-source change): the A side
    # (rank0) owns combineRecv admission and prepare_combine_recv; the F side
    # (rank1) owns recv/ffn preparation feeding combineSend. Each rank compiles
    # a different production TU, so no duplicate same-TU compile.
    if rank == 0:
        try:
            owner.prepare_combine_recv(round_step=True, fair_round_robin=True)
            out['prepared'] = 'A-side combine_recv(round_step)'
            t0 = time.monotonic()
            try:
                cache_path = owner.compileCombineRecvOnly()
                results['combine_recv'] = {'ok': True,
                    'compile_ms': round((time.monotonic() - t0) * 1000.0, 1),
                    'cache_path': cache_path}
            except BaseException as exc:
                results['combine_recv'] = {'ok': False,
                    'compile_ms': round((time.monotonic() - t0) * 1000.0, 1),
                    'error': repr(exc)[:20000]}
            emit({'rank': rank, 'stage': 'compile', 'op': 'combine_recv',
                  'result': results['combine_recv']})
        except BaseException as exc:
            prep_error = repr(exc)[:20000]
    if rank == 1:
        try:
            owner.prepare_recv(56, 64, 4, False)
            owner.prepare_ffn(False)
            out['prepared'] = 'F-side recv(56,64,4)+ffn'
            t0 = time.monotonic()
            try:
                cache_path = owner.compileCombineSendOnly()
                results['combine_send'] = {'ok': True,
                    'compile_ms': round((time.monotonic() - t0) * 1000.0, 1),
                    'cache_path': cache_path}
            except BaseException as exc:
                results['combine_send'] = {'ok': False,
                    'compile_ms': round((time.monotonic() - t0) * 1000.0, 1),
                    'error': repr(exc)[:20000]}
            emit({'rank': rank, 'stage': 'compile', 'op': 'combine_send',
                  'result': results['combine_send']})
        except BaseException as exc:
            prep_error = repr(exc)[:20000]
    out['prepare_error'] = prep_error
    out['compile'] = results
    try:
        dist.barrier()
        out['barrier_ok'] = True
    except BaseException as exc:
        out['barrier_ok'] = False
        out['barrier_error'] = repr(exc)[:2000]
        emit(out)
        sys.exit(2)
    bad = bool(out.get('prepare_error')) or any(not r.get('ok') for r in results.values())
    out['ok'] = not bad
    out['cleanup'] = {'owner_destroy': 'PENDING', 'pg_barrier': 'OK', 'pg_destroy': 'UNKNOWN'}
    try:
        owner.destroy()
        out['destroy_ok'] = True
        out['cleanup']['owner_destroy'] = 'OK'
    except BaseException as exc:
        out['destroy_ok'] = False
        out['destroy_error'] = repr(exc)[:2000]
        out['cleanup']['owner_destroy'] = 'FAILED'
        emit(out)
        sys.exit(4)
    try:
        dist.barrier()
        out['final_barrier_ok'] = True
    except BaseException as exc:
        out['final_barrier_ok'] = False
        out['final_barrier_error'] = repr(exc)[:1000]
        emit(out)
        sys.exit(5)
    try:
        dist.destroy_process_group()
        out['pg_destroy_ok'] = True
        out['cleanup']['pg_destroy'] = 'OK'
    except BaseException as exc:
        out['pg_destroy_ok'] = False
        out['pg_destroy_error'] = repr(exc)[:1000]
        emit(out)
        sys.exit(6)
    emit(out)
    print('COMBINE_COMPILE_DONE rank=%d' % rank, flush=True)
    if bad:
        sys.exit(7)


if __name__ == '__main__':
    main()
