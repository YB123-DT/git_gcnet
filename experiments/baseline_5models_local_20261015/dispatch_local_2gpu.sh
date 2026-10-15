#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
exec "${BASELINE_PYTHON:-/home/yangbin/miniconda3/envs/multimodalerc310/bin/python}" "$BASE/dispatch.py" "$@"
