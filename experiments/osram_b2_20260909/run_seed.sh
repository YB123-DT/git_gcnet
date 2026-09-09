#!/usr/bin/env bash
# One approved MOSI seed: fixed-space pretraining, audit, then B2 joint training.
set -euo pipefail
seed=${1:?seed required}
gpu=${2:?GPU required}
case "$seed" in 66|67|68|69|70) ;; *) exit 2 ;; esac
case "$gpu" in 0|1) ;; *) exit 2 ;; esac
cd /data2/yb/paper/GCNet_TPAMI_missing_m3_target_ple
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
python=/data2/yb/reproduction_envs/s0/bin/python3.10
output=/data2/yb/remote_experiments/osram_b2_20260909/formal/seed_${seed}
base=/data2/yb/remote_experiments/osram_write_step_train_20260909/mosi/seed_${seed}/best.pt
features=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features
test -f "$base"
# Atomic directory creation rejects duplicate launch and never overwrites artifacts.
mkdir "$output"
trap 'echo "FAILED seed=$seed line=$LINENO" >&2' ERR
echo "START seed=$seed gpu=$gpu utc=$(date -u +%FT%TZ)"
sha256sum gcnet_missing_m3/{b2.py,b2_training.py,train_b2.py,model.py,osram.py,train_gcnet.py,loss.py,mixed_rate.py}
"$python" -u -m gcnet_missing_m3.train_b2 --stage stage1 \
    --base-checkpoint "$base" --feature-root "$features" \
    --output-dir "$output/stage1" --epochs 100 --device cuda
echo "STAGE1_COMPLETE seed=$seed utc=$(date -u +%FT%TZ)"
"$python" -u -m gcnet_missing_m3.train_b2 --stage stage2 \
    --base-checkpoint "$base" --feature-root "$features" \
    --pretrain-checkpoint "$output/stage1/source_only_completion_pretrain.pt" \
    --output-dir "$output/stage2" --epochs 100 --device cuda
echo "COMPLETE seed=$seed utc=$(date -u +%FT%TZ)"
