#!/usr/bin/env bash
set -euo pipefail

# Same verified worker/source/environment as the prior feature-only comparison.
run_root=/home/qinjinghui/remote_experiments/osram_meld_mmgcn_features_20261018
python_env=/data2/yb/conda_envs/cmerc-repro/bin/python
worker=$run_root/code/66fc911607d06ebc98b7e7ab407aaf786185586eebcd8cb2ac4ab4e6b46a48d1/worker.py
source_root=/home/qinjinghui/remote_experiments/osram_meld_b3_code_737fdb1
gpu_uuid=GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62

export CUDA_VISIBLE_DEVICES=$gpu_uuid
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>"$run_root/.pipeline.lock"
flock -n 9

common=(
  --data "$run_root/data/mmgcn_canonical_meld.pkl"
  --data-sha256 d98f81538c1a2932ec3b959df6d02cf54930cf61063353174d284c9c1b8facd0
  --gpu-uuid "$gpu_uuid"
  --source-root "$source_root"
  --source-config "$run_root/CONFIG.json"
)

"$python_env" -u "$worker" "${common[@]}" --output "$run_root/smoke/Flat_MMGCN_seed66" --smoke
"$python_env" -u "$worker" "${common[@]}" --output "$run_root/runs/Flat_MMGCN_seed66"
