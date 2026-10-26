"""Compile-only JIT preparer for the queue-metadata probe candidate.

Builds one AFDBuffer per rank with the same s8 small shape (approved
construction-only), then calls compileQueueMetaProbeOnly(), which runs the
production DeepJIT pipeline (kernel.asc -> bisheng -> ld.lld) through
compile_without_load: no driver load of the probe kernel and no kernel
launch. Prints the cache path; any stage failure yields nonzero.
"""
import hashlib
import json
import os
import sys
import time
from datetime import timedelta


def emit(obj):
    print('COMPILE_ONLY ' + json.dumps(obj, sort_keys=True), flush=True)


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
    # In-process loaded-library identity: only this process's mapped objects,
    # distinguishing the LD search path from the file actually loaded.
    def loaded_libs():
        seen = {}
        try:
            with open('/proc/self/maps') as f:
                for ln in f:
                    if 'libhccl' not in ln and 'libhcomm' not in ln:
                        continue
                    parts = ln.split()
                    if len(parts) >= 6 and parts[-1].startswith('/'):
                        seen[parts[-1]] = True
        except BaseException as exc:
            return {'error': repr(exc)[:400]}
        entries = []
        for p in sorted(seen):
            rp = os.path.realpath(p)
            try:
                with open(rp, 'rb') as f:
                    h = hashlib.sha256(f.read()).hexdigest()
            except BaseException as exc:
                h = 'ERROR:' + repr(exc)[:200]
            entries.append({'mapped_path': p, 'realpath': rp, 'sha256': h,
                            'ld_path_differs_from_mapped': p != rp})
        return entries
    out['loaded_libs'] = loaded_libs()
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
    t0 = time.monotonic()
    try:
        cache_path = owner.compileQueueMetaProbeOnly()
        out['compile_ms'] = round((time.monotonic() - t0) * 1000.0, 1)
        out['cache_path'] = cache_path
        out['ok'] = True
        emit(out)
    except BaseException as exc:
        out['stage'] = 'compile'
        out['ok'] = False
        out['error'] = repr(exc)[:20000]
        emit(out)
        sys.exit(2)
    try:
        dist.barrier()
        out['barrier_ok'] = True
    except BaseException as exc:
        out['barrier_ok'] = False
        out['barrier_error'] = repr(exc)[:1000]
        emit(out)
        dist.destroy_process_group() if False else None
        sys.exit(4)
    try:
        owner.destroy()
        out['destroy_ok'] = True
    except BaseException as exc:
        out['destroy_ok'] = False
        out['destroy_error'] = repr(exc)[:1000]
        emit(out)
        sys.exit(5)
    try:
        dist.barrier()
        out['final_barrier_ok'] = True
    except BaseException as exc:
        out['final_barrier_ok'] = False
        out['final_barrier_error'] = repr(exc)[:1000]
        emit(out)
        sys.exit(6)
    try:
        dist.destroy_process_group()
        out['pg_destroy_ok'] = True
    except BaseException as exc:
        out['pg_destroy_ok'] = False
        out['pg_destroy_error'] = repr(exc)[:1000]
        emit(out)
        sys.exit(7)
    out['cleanup'] = {'owner_destroy': 'OK', 'pg_barrier': 'OK', 'pg_destroy': 'OK'}
    emit(out)
    print('COMPILE_ONLY_DONE rank=%d' % rank, flush=True)


if __name__ == '__main__':
    main()
