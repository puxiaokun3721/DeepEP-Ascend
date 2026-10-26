"""Read-only RankGraph endpoint probe at world 4 (reviewer-directed scalars only).

Initializes the HCCL process group only (no AFDBuffer, no endpoints, no
registration, no channels), calls the private read-only binding
deep_ep._C.probe_rank_graph_scalars(group_name, 0), prints its single-line
JSON, then destroys the process group and exits. No operator launch.
"""
import hashlib
import json
import os
import sys
from datetime import timedelta


def emit(obj):
    print('RANKGRAPH_DIAG ' + json.dumps(obj, sort_keys=True), flush=True)


def main():
    out = {'local_rank': int(os.environ.get('LOCAL_RANK', -1)), 'pid': os.getpid()}
    try:
        import torch
        import torch_npu  # noqa: F401  Registers the NPU backend.
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
        out['torch_npu'] = getattr(torch_npu, '__version__', None)
    except BaseException as exc:
        out.update({'stage': 'binding', 'ok': False, 'error': repr(exc)[:3000]})
        emit(out)
        sys.exit(3)
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=300))
    out['world'] = dist.get_world_size()
    rank = dist.get_rank()
    out['rank'] = rank
    if out['world'] != 4:
        out.update({'ok': False, 'error': 'world=%d != 4' % out['world']})
        emit(out)
        sys.exit(3)
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    try:
        line = _c.probe_rank_graph_scalars(group_name, 0)
        print('RANKGRAPH_PROBE ' + line.strip(), flush=True)
        out['probe_ok'] = True
    except BaseException as exc:
        out.update({'probe_ok': False, 'probe_error': repr(exc)[:3000]})
        emit(out)
        sys.exit(4)
    try:
        dist.barrier()
        out['barrier_ok'] = True
    except BaseException as exc:
        out.update({'barrier_ok': False, 'barrier_error': repr(exc)[:1500]})
        emit(out)
        sys.exit(5)
    try:
        dist.destroy_process_group()
        out['pg_destroy_ok'] = True
    except BaseException as exc:
        out.update({'pg_destroy_ok': False, 'pg_destroy_error': repr(exc)[:1500]})
        emit(out)
        sys.exit(6)
    out['ok'] = True
    emit(out)
    print('RANKGRAPH_DIAG_DONE rank=%d' % rank, flush=True)


if __name__ == '__main__':
    main()
