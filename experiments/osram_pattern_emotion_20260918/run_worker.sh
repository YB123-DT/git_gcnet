#!/usr/bin/env bash
set -euo pipefail

GPU="$1"
shift
cd /data2/yb/paper/GCNet_TPAMI_missing_m3_target_ple
LOG_DIR=/data2/yb/remote_experiments/osram_pattern_emotion_20260918/logs
mkdir -p "$LOG_DIR"

for spec in "$@"; do
    read -r variant seed <<< "$spec"
    echo "START variant=$variant seed=$seed gpu=$GPU"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. \
        /data2/yb/reproduction_envs/s0/bin/python3.10 -u \
        experiments/osram_pattern_emotion_20260918/run.py \
        --variant "$variant" --seed "$seed"
done
