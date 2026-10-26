"""AFD queue-metadata read-only probe runner (review candidate).

Constructs one AFDBuffer per rank with the same s8 small shape, allocates an
int32 result tensor on the device, invokes the read-only metadata probe over
the payload/control jetty tables, synchronizes, prints the scalar blocks, and
walks the existing destroy path. Any stage failure yields a nonzero exit; no
dispatchSend/combine/queue-write kernel is invoked.
"""
import hashlib
import json
import os
import sys
import time
from datetime import timedelta


def emit(obj):
    print('QUEUE_META ' + json.dumps(obj, sort_keys=True), flush=True)


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
        if out['so_sha256'] != '513c0307ae85e773623c19d0ffdd75dfddb9e834b8e4e962e32bdbd9146f2c9e':
            raise RuntimeError('unexpected task host extension')
    except BaseException as exc:
        out['stage'] = 'binding'
        out['ok'] = False
        out['error'] = repr(exc)[:4000]
        emit(out)
        sys.exit(3)
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=180))
    out['world'] = dist.get_world_size()
    rank = dist.get_rank()
    out['rank'] = rank
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    hidden, k, experts, capacity = 128, 8, 8, 33
    fp8, cores = False, 1
    data = generate(batch=capacity, hidden=hidden, k=k, experts=experts,
                    ranks=1, fp8=fp8)
    sf_packs = data['sf_bytes'].shape[1] // 2
    owner = None
    t0 = time.monotonic()
    try:
        owner = deep_ep.AFDBuffer(rank, group_name, [0], [1],
            [[0, 0, 0, 0, 0, capacity]], [0], hidden, sf_packs, k, experts,
            capacity, fp8, cores, 60, 30, 8 * 1024 ** 3, 256 * 1024 ** 2)
        out['construct_ms'] = round((time.monotonic() - t0) * 1000.0, 1)
    except BaseException as exc:
        out['stage'] = 'construct'
        out['ok'] = False
        out['error'] = repr(exc)[:4000]
        out['cleanup'] = {'owner_destroy': 'NOT_ATTEMPTED_INCOMPLETE_OBJECT',
                          'pg_barrier': 'NOT_ATTEMPTED', 'pg_destroy': 'UNKNOWN'}
        emit(out)
        sys.exit(1)
    t0 = time.monotonic()
    try:
        # One rank builds the shared private cache; both then load that artifact.
        if rank == 0:
            owner.compileQueueMetaProbeOnly()
        dist.barrier()
        result = torch.zeros(32, dtype=torch.int32).npu()
        owner.probeQueueMeta(result)
        torch.npu.synchronize()
        out['probe_ms'] = round((time.monotonic() - t0) * 1000.0, 1)
        words = result.cpu().tolist()
        out['payload_block'] = words[0:16]
        out['control_block'] = words[16:32]
        out['layout'] = ['valid', 'engine', 'protocol', 'sqNum', 'cqNum',
                         'ubJfs.wqeSize', 'ubJfs.sqDepth', 'ubJfc.cqeSize',
                         'ubJfc.cqDepth', 'defaultQpIdx', 'error',
                          'sqType', 'cqType', 'reserved...']
        out['metadata_valid'] = all(
            block[0] == 1 and block[10] == 0
            for block in (words[0:16], words[16:32]))
        out['ok'] = out['metadata_valid']
        emit(out)
    except BaseException as exc:
        out['stage'] = 'probe'
        out['ok'] = False
        out['error'] = repr(exc)[:4000]
        out['cleanup'] = {'owner_destroy': 'NOT_ATTEMPTED', 'pg_barrier': 'NOT_ATTEMPTED',
                          'pg_destroy': 'UNKNOWN'}
        emit(out)
        sys.exit(2)
    try:
        dist.barrier()
        out['barrier_ok'] = True
    except BaseException as exc:
        out['stage'] = 'barrier'
        out['barrier_ok'] = False
        out['barrier_error'] = repr(exc)[:1000]
        emit(out)
        sys.exit(4)
    try:
        owner.destroy()
        out['destroy_ok'] = True
    except BaseException as exc:
        out['stage'] = 'owner_destroy'
        out['destroy_ok'] = False
        out['destroy_error'] = repr(exc)[:1000]
        out['cleanup'] = {'owner_destroy': 'FAILED', 'pg_barrier': 'NOT_ATTEMPTED',
                          'pg_destroy': 'UNKNOWN'}
        emit(out)
        sys.exit(5)
    try:
        dist.barrier()
        out['final_barrier_ok'] = True
    except BaseException as exc:
        out['final_barrier_ok'] = False
        out['final_barrier_error'] = repr(exc)[:1000]
        out['cleanup'] = {'owner_destroy': 'OK', 'pg_barrier': 'FAILED', 'pg_destroy': 'UNKNOWN'}
        emit(out)
        sys.exit(6)
    try:
        dist.destroy_process_group()
        out['pg_destroy_ok'] = True
    except BaseException as exc:
        out['pg_destroy_ok'] = False
        out['pg_destroy_error'] = repr(exc)[:1000]
        out['cleanup'] = {'owner_destroy': 'OK', 'pg_barrier': 'OK', 'pg_destroy': 'FAILED'}
        emit(out)
        sys.exit(7)
    out['cleanup'] = {'owner_destroy': 'OK', 'pg_barrier': 'OK', 'pg_destroy': 'OK'}
    emit(out)
    print('QUEUE_META_DONE rank=%d' % rank, flush=True)
    if not out['metadata_valid']:
        sys.exit(8)


if __name__ == '__main__':
    main()
