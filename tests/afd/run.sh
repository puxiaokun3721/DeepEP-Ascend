#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -B -m unittest discover -s tests/afd -p test_cpu.py -v
printf '%s\n' 'CPU contracts only; no Ascend kernel, NPU communication or performance validation performed.'
