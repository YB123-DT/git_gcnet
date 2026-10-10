#!/usr/bin/env bash

set -euo pipefail

REPOSITORY="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
WORKSPACE="${PAPER_WORKSPACE:-/data2/yb/paper}"
RUN_DIR="${WORKSPACE}/05_reproduction/runs/ComP_causal/smoke/CMUMOSI"
WRAPPER="${REPOSITORY}/experiments/comp_causal_seed66/run_comp_causal.py"
GPU="${1:?host GPU index is required}"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
env PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  COMP_OUTPUT_ROOT="${RUN_DIR}/saved" \
  /home/yangbin/miniconda3/bin/conda run --no-capture-output -n comp-repro \
  python "${WRAPPER}" \
    --dataset=CMUMOSI \
    --audio-feature=wav2vec-large-c-UTT \
    --text-feature=deberta-large-4-UTT \
    --video-feature=manet_UTT \
    --seed=66 --batch-size=32 --epochs=2 --lr=0.0001 \
    --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 \
    --attn_drop_rate=0.0 --stage_epoch=1 --gpu=0 \
    --mask_rate=0.7 --lbd=0.3 \
    > run.log 2>&1
