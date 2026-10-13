#!/usr/bin/env bash

set -euo pipefail

REPOSITORY="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PYTHON_BIN="${OSRAM_PYTHON:-/data2/yb/reproduction_workspace/envs/s0/bin/python}"
SOURCE_ROOT="${MELD_FLAT_SOURCE_ROOT:-/data2/yb/paper/remote_runs_20261012/results/osram_meld_gcnet_legacy_fixed_50e_3seed_20261013/runs}"
DATASET_ROOT="${GCNET_DATASET_ROOT:-/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset}"
MELD_PICKLE="${GCNET_MELD_PICKLE:-/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl}"
OUTPUT_ROOT="${MELD_FLAT_COMBINATION_OUTPUT:-/data2/yb/paper/remote_runs_20261012/results/osram_meld_flat_source_combinations_20261013}"
GPU_UUID="${MELD_FLAT_COMBINATION_GPU:-GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4}"

mkdir -p "${OUTPUT_ROOT}/logs"
pids=()
for seed in 66 67 68; do
  output="${OUTPUT_ROOT}/seed_${seed}"
  if [[ -f "${output}/SUMMARY.json" ]] && grep -q '"status": "complete"' "${output}/SUMMARY.json"; then
    echo "[skip] complete seed=${seed}"
    continue
  fi
  [[ ! -e "${output}" ]] || mv "${output}" "${output}.partial-$(date -u +%Y%m%dT%H%M%SZ)"
  env CUDA_VISIBLE_DEVICES="${GPU_UUID}" GCNET_CPU_THREADS=2 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    "${PYTHON_BIN}" -u -m experiments.osram_meld_flat_source_combinations_20261013.evaluate \
      --source "${SOURCE_ROOT}/flat_seed${seed}" \
      --dataset-root "${DATASET_ROOT}" \
      --meld-pickle "${MELD_PICKLE}" \
      --output "${output}" --device cuda \
      > "${OUTPUT_ROOT}/logs/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
exit "${failed}"
