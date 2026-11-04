"""Repeat the full CAM AFD transaction inside one process.

Each cycle re-runs run_pipeline_cam.main(): build the AFDBuffer, run the real
dispatch/combine transaction, destroy the owner. The process group stays alive
across cycles because HCCL does not support destroy -> re-init in one process
(CYCMIN2 cycle 1 failed in hcclCommInitRootInfoConfig, error 4 / EI0015), so
the driver covers owner teardown after real traffic plus rebuild on a live
communicator, which the construct/destroy reuse test does not touch.
"""
import argparse
import importlib.util
import os
import sys
import time


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location('run_pipeline_cam', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--pipeline-script', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'run_pipeline_cam.py'))
    known, rest = parser.parse_known_args()
    if known.cycles < 1:
        parser.error('cycles must be positive')
    if '--profile-trace' in rest:
        parser.error('profile-trace is single-cycle only')
    module = load_pipeline(known.pipeline_script)
    base_tag = None
    for i, token in enumerate(rest):
        if token == '--run-tag' and i + 1 < len(rest):
            base_tag = rest[i + 1]
    for cycle in range(known.cycles):
        argv = list(rest)
        tag = '{}-c{}'.format(base_tag or 'CYC', cycle)
        if base_tag is None:
            argv += ['--run-tag', tag]
        else:
            argv[argv.index('--run-tag') + 1] = tag
        if '--keep-pg' not in argv:
            argv.append('--keep-pg')
        sys.argv = [sys.argv[0]] + argv
        print('CYCLE_BEGIN cycle={} tag={} t={:.3f}'.format(
            cycle, tag, time.monotonic()), flush=True)
        module.main()
        print('CYCLE_END cycle={} t={:.3f}'.format(
            cycle, time.monotonic()), flush=True)
    import torch.distributed as dist
    if dist.is_initialized():
        stage = 'after last cycle'
        print('CYCLE_PG_DESTROY ' + stage, flush=True)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
