#!/usr/bin/env bash
set -euo pipefail

GPU="$1"
shift
cd /data2/yb/paper/GCNet_TPAMI_missing_m3_target_ple
for seed in "$@"; do
    echo "START Text-Core seed=$seed GPU=$GPU"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. \
      /data2/yb/reproduction_envs/s0/bin/python3.10 -u \
      experiments/osram_text_core_20260918/run.py --seed "$seed"
done
