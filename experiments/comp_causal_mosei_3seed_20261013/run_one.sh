#!/usr/bin/env bash

set -euo pipefail

EPOCHS=100
STAGE_EPOCH=50
CPU_THREADS=4

if [[ "${1:-}" == "--print-budget" ]]; then
  if [[ "${2:-}" != "CMUMOSEI" ]]; then
    echo "Unsupported MOSEI dataset: ${2:-}" >&2
    exit 2
  fi
  printf 'epochs=%s stage_epoch=%s\n' "${EPOCHS}" "${STAGE_EPOCH}"
  exit 0
fi

if [[ "${1:-}" == "--print-runtime" ]]; then
  printf 'cpu_threads=%s\n' "${CPU_THREADS}"
  exit 0
fi

REPOSITORY="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
WRAPPER="${REPOSITORY}/experiments/comp_causal_seed66/run_comp_causal.py"
PYTHON_BIN="${COMP_PYTHON:-/home/qinjinghui/anaconda3/envs/emo/bin/python}"
DATA_ROOT="${COMP_DATA_ROOT:-/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset}"
OUTPUT_ROOT="${COMP_MOSEI_OUTPUT_ROOT:-/data2/yb/paper/remote_runs_20261012/results/comp_causal_mosei_3seed_100e_cpu4_20261013}"

DATASET="${1:?dataset is required}"
SEED="${2:?seed is required}"
MASK_RATE="${3:?mask rate is required}"
GPU_UUID="${4:?physical GPU UUID is required}"
[[ "${DATASET}" == "CMUMOSEI" ]] || { echo "Unsupported MOSEI dataset: ${DATASET}" >&2; exit 2; }

RUN_DIR="${OUTPUT_ROOT}/${DATASET}/seed${SEED}/mr_${MASK_RATE}"
RESULT_FILE="${RUN_DIR}/${DATASET}-${MASK_RATE}.txt"
MANIFEST="${RUN_DIR}/run_manifest.txt"
LOCK_FILE="${RUN_DIR}/.launch.lock"
REPOSITORY_COMMIT="$(git -C "${REPOSITORY}" rev-parse HEAD)"

mkdir -p "${RUN_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[skip] locked ${DATASET} seed=${SEED} rate=${MASK_RATE}"
  exit 0
fi
if [[ -f "${RESULT_FILE}" ]] && grep -q "Folder avg:" "${RESULT_FILE}"; then
  echo "[skip] complete ${DATASET} seed=${SEED} rate=${MASK_RATE}"
  exit 0
fi
if [[ -s "${RESULT_FILE}" ]]; then
  mv "${RESULT_FILE}" "${RESULT_FILE}.partial-$(date -u +%Y%m%dT%H%M%SZ)"
fi

printf '%s\n' \
  "repository_commit=${REPOSITORY_COMMIT}" \
  "temporal_mode=causal" \
  "dataset=${DATASET}" \
  "seed=${SEED}" \
  "mask_rate=${MASK_RATE}" \
  "epochs=${EPOCHS}" \
  "stage_epoch=${STAGE_EPOCH}" \
  "budget_evidence=ComP paper Figure 5(a)" \
  "cpu_threads=${CPU_THREADS}" \
  "batch_size=32" \
  "learning_rate=0.0001" \
  "hidden=256" \
  "depth=4" \
  "num_heads=2" \
  "drop_rate=0.5" \
  "lbd=0.3" \
  "data_root=${DATA_ROOT}" \
  "host=$(hostname)" \
  "physical_gpu_uuid=${GPU_UUID}" \
  "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${MANIFEST}"

echo "[start] ${DATASET} seed=${SEED} rate=${MASK_RATE} gpu=${GPU_UUID} epochs=${EPOCHS} stage=${STAGE_EPOCH}"
(
  cd "${RUN_DIR}"
  env PYTHONDONTWRITEBYTECODE=1 \
    COMP_CPU_THREADS="${CPU_THREADS}" \
    OMP_NUM_THREADS="${CPU_THREADS}" \
    MKL_NUM_THREADS="${CPU_THREADS}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
    NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
    VECLIB_MAXIMUM_THREADS="${CPU_THREADS}" \
    OMP_WAIT_POLICY=PASSIVE \
    CUDA_VISIBLE_DEVICES="${GPU_UUID}" \
    COMP_DATA_ROOT="${DATA_ROOT}" \
    COMP_OUTPUT_ROOT="${RUN_DIR}/saved" \
    "${PYTHON_BIN}" -u "${WRAPPER}" \
      --dataset="${DATASET}" \
      --audio-feature=wav2vec-large-c-UTT \
      --text-feature=deberta-large-4-UTT \
      --video-feature=manet_UTT \
      --temporal-mode=causal \
      --seed="${SEED}" --batch-size=32 --epochs="${EPOCHS}" --lr=0.0001 \
      --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 \
      --attn_drop_rate=0.0 --stage_epoch="${STAGE_EPOCH}" --gpu=0 \
      --mask_rate="${MASK_RATE}" --lbd=0.3 \
      > run.log 2>&1
)

if ! grep -q "Folder avg:" "${RESULT_FILE}"; then
  echo "[error] result did not reach Folder avg: ${RESULT_FILE}" >&2
  exit 5
fi
printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${MANIFEST}"
echo "[done] ${DATASET} seed=${SEED} rate=${MASK_RATE}"
