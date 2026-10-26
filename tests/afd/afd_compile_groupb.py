"""World-2 construction + serial compile-only check for the branch B diff.

Constructs one AFDBuffer per rank with the pipeline shape (1 A + 1 F, batch 13,
window 16, q 7, one microbatch), calls the existing prepare entries, then
compiles every real specialization TU once, serially in world rank order with a
barrier between ranks (true single compile job). No communication kernel runs.
"""
import hashlib
import json
import os
import sys
from datetime import timedelta


def emit(obj):
    print('COMPILE_GRPB ' + json.dumps(obj, sort_keys=True), flush=True)


def main():
    out = {'local_rank': int(os.environ.get('LOCAL_RANK', -1)), 'pid': os.getpid()}
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
    if out['world'] != 2:
        out.update({'ok': False, 'error': 'world=%d != 2' % out['world']})
        emit(out)
        sys.exit(3)
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    sources, receivers = [0], [1]
    owner = None
    try:
        owner = deep_ep.AFDBuffer(rank, group_name, sources, receivers,
            [[0, 0, 0, 0, 0, 16]], [0], 128, 0, 2, 2, 13, False, 1, 120, 300,
            8 * 1024 ** 3, 256 * 1024 ** 2)
        out['construct_ok'] = True
    except BaseException as exc:
        out.update({'stage': 'construct', 'ok': False, 'error': repr(exc)[:4000]})
        emit(out)
        sys.exit(1)
    try:
        if rank in sources:
            owner.prepare_combine_recv(round_step=False, fair_round_robin=False)
        else:
            owner.prepare_recv(7, 7, 1)
            owner.prepare_ffn(False)
        out['prepare_ok'] = True
    except BaseException as exc:
        out.update({'stage': 'prepare', 'ok': False, 'error': repr(exc)[:4000]})
        emit(out)
        sys.exit(2)
    results = {}
    failed = []

    def run_tu(name, fn):
        try:
            r = fn()
            results[name] = {'ok': True, 'result_len': len(str(r)) if r is not None else 0}
            emit({'rank': rank, 'tu': name, 'ok': True})
        except BaseException as exc:
            results[name] = {'ok': False, 'error': repr(exc)[:1500]}
            failed.append(name)
            emit({'rank': rank, 'tu': name, 'ok': False, 'error_type': type(exc).__name__})

    for warm_rank in range(out['world']):
        if rank == warm_rank:
            if rank in sources:
                run_tu('dispatch_send', owner.compileDispatchSendOnly)
                run_tu('combine_recv', owner.compileCombineRecvOnly)
            else:
                run_tu('ffn_scheduler', owner.compileFFNSchedulerOnly)
                run_tu('dispatch_recv', owner.compileDispatchRecvOnly)
                run_tu('compute_ready', owner.compileComputeReadyOnly)
                run_tu('combine_send', owner.compileCombineSendOnly)
        dist.barrier()
    out['tus'] = results
    out['failed'] = failed
    try:
        torch.npu.synchronize()
        dist.barrier()
        out['barrier_ok'] = True
    except BaseException as exc:
        out.update({'barrier_ok': False, 'barrier_error': repr(exc)[:1200]})
        emit(out)
        sys.exit(4)
    try:
        owner.destroy()
        out['destroy_ok'] = True
    except BaseException as exc:
        out.update({'destroy_ok': False, 'destroy_error': repr(exc)[:2000]})
        emit(out)
        sys.exit(5)
    try:
        dist.destroy_process_group()
        out['pg_destroy_ok'] = True
    except BaseException as exc:
        out.update({'pg_destroy_ok': False, 'pg_destroy_error': repr(exc)[:1200]})
        emit(out)
        sys.exit(6)
    out['ok'] = not failed
    emit(out)
    print('COMPILE_GRPB_DONE rank=%d failed=%d' % (rank, len(failed)), flush=True)
    if failed:
        sys.exit(7)


if __name__ == '__main__':
    main()
