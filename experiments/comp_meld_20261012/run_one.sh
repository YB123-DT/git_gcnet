#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
WORKSPACE="${PAPER_WORKSPACE:-/data2/yb/paper}"
MODE="${1:?causal or noncausal required}"
MASK_RATE="${2:?mask rate required}"
HOST_GPU="${3:?host GPU required}"
SEED="${COMP_SEED:-66}"
RUN_ROOT="${COMP_MELD_RUN_ROOT:-${WORKSPACE}/05_reproduction/runs/ComP_MELD_20261012}"
RUN_DIR="${RUN_ROOT}/${MODE}/seed_${SEED}/mr_${MASK_RATE}"
WRAPPER="${REPOSITORY}/experiments/comp_meld_20261012/run_comp_meld.py"
PYTHON_BIN="${COMP_PYTHON:-/home/yangbin/miniconda3/envs/comp-repro/bin/python}"
CPU_THREADS="${COMP_CPU_THREADS:-4}"
EPOCHS="${COMP_EPOCHS:-300}"
STAGE_EPOCH="${COMP_STAGE_EPOCH:-150}"
KEEP_CHECKPOINTS="${COMP_KEEP_CHECKPOINTS:-1}"
MASK_PROTOCOL="${COMP_MASK_PROTOCOL:-legacy-random}"

if [[ "${MODE}" != "causal" && "${MODE}" != "noncausal" ]]; then
  echo "invalid temporal mode: ${MODE}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"
if [[ -f "${RUN_DIR}/MELD-${MASK_RATE}.txt" ]] && grep -q 'Folder avg:' "${RUN_DIR}/MELD-${MASK_RATE}.txt"; then
  echo "[skip] completed ${MODE} MELD seed=${SEED} miss=${MASK_RATE}"
  exit 0
fi

printf '%s\n' \
  "repository_commit=$(git -C "${REPOSITORY}" rev-parse HEAD)" \
  "upstream_comp_commit=28192d3a5683543d7383e40898f9a98d1f114a08" \
  "dataset=MELD" "feature_source=ECERC" "temporal_mode=${MODE}" \
  "seed=${SEED}" "mask_rate=${MASK_RATE}" "host_gpu=${HOST_GPU}" \
  "epochs=${EPOCHS}" "stage_epoch=${STAGE_EPOCH}" "batch_size=32" "hidden=256" \
  "cpu_threads=${CPU_THREADS}" "keep_checkpoints=${KEEP_CHECKPOINTS}" \
  "mask_protocol=${MASK_PROTOCOL}" \
  > "${RUN_DIR}/run_manifest.txt"

cd "${RUN_DIR}"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="${HOST_GPU}" \
  COMP_CPU_THREADS="${CPU_THREADS}" OMP_NUM_THREADS="${CPU_THREADS}" \
  MKL_NUM_THREADS="${CPU_THREADS}" OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
  NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
  COMP_OUTPUT_ROOT="${RUN_DIR}/saved" \
  "${PYTHON_BIN}" "${WRAPPER}" --dataset=MELD --temporal-mode="${MODE}" \
    --seed="${SEED}" --batch-size=32 --epochs="${EPOCHS}" --stage_epoch="${STAGE_EPOCH}" \
    --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 \
    --attn_drop_rate=0.0 --gpu=0 --mask_rate="${MASK_RATE}" --lbd=0.3 \
    --mask-protocol="${MASK_PROTOCOL}" \
  > run.log 2>&1

if [[ "${KEEP_CHECKPOINTS}" == "0" ]]; then
  checkpoint_count="$(find "${RUN_DIR}/saved" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.pkl' \) 2>/dev/null | wc -l)"
  checkpoint_bytes="$(find "${RUN_DIR}/saved" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.pkl' \) -printf '%s\n' 2>/dev/null | awk '{total += $1} END {print total + 0}')"
  find "${RUN_DIR}/saved" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.pkl' \) -delete 2>/dev/null || true
  printf '%s\n' \
    "discarded_checkpoint_count=${checkpoint_count}" \
    "discarded_checkpoint_bytes=${checkpoint_bytes}" \
    >> "${RUN_DIR}/run_manifest.txt"
fi
