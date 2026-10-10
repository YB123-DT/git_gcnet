#!/usr/bin/env bash

set -euo pipefail

REPOSITORY="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
WORKSPACE="${PAPER_WORKSPACE:-/data2/yb/paper}"
WRAPPER="${REPOSITORY}/experiments/comp_causal_seed66/run_comp_causal.py"
CONDA_BIN="/home/yangbin/miniconda3/bin/conda"
ENV_NAME="comp-repro"
GPU="${1:?host GPU index is required}"
shift

if [[ "$#" -eq 0 ]]; then
  echo "At least one mask rate is required" >&2
  exit 2
fi

OUTPUT_BASE="${WORKSPACE}/05_reproduction/runs/ComP_causal/seed66/CMUMOSI"
REPOSITORY_COMMIT="$(git -C "${REPOSITORY}" rev-parse HEAD)"

if [[ -n "$(git -C "${REPOSITORY}" status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing formal launch from a dirty repository: ${REPOSITORY}" >&2
  exit 3
fi

manifest_matches() {
  local manifest="$1"
  local mask_rate="$2"
  [[ -f "${manifest}" ]] \
    && rg -Fxq "repository_commit=${REPOSITORY_COMMIT}" "${manifest}" \
    && rg -Fxq "upstream_comp_commit=28192d3a5683543d7383e40898f9a98d1f114a08" "${manifest}" \
    && rg -Fxq "seed=66" "${manifest}" \
    && rg -Fxq "mask_rate=${mask_rate}" "${manifest}" \
    && rg -Fxq "epochs=300" "${manifest}" \
    && rg -Fxq "stage_epoch=150" "${manifest}" \
    && rg -Fxq "batch_size=32" "${manifest}" \
    && rg -Fxq "hidden=256" "${manifest}" \
    && rg -Fxq "features=wav2vec-large-c-UTT,deberta-large-4-UTT,manet_UTT" "${manifest}"
}

run_one() {
  local mask_rate="$1"
  local run_dir="${OUTPUT_BASE}/mr_${mask_rate}"
  local result_file="${run_dir}/CMUMOSI-${mask_rate}.txt"
  local manifest="${run_dir}/run_manifest.txt"

  mkdir -p "${run_dir}"
  if [[ -f "${result_file}" ]] && rg -q "Folder avg:" "${result_file}"; then
    if manifest_matches "${manifest}" "${mask_rate}"; then
      echo "[skip] CMUMOSI seed=66 mask_rate=${mask_rate} already complete with matching provenance"
      return 0
    fi
    echo "Refusing to reuse completed result with mismatched provenance: ${run_dir}" >&2
    return 4
  fi
  if [[ -f "${result_file}" ]]; then
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    mv "${result_file}" "${run_dir}/CMUMOSI-${mask_rate}.partial-${stamp}.txt"
  fi

  printf '%s\n' \
    "repository_commit=${REPOSITORY_COMMIT}" \
    "upstream_comp_commit=28192d3a5683543d7383e40898f9a98d1f114a08" \
    "hostname=$(hostname)" \
    "host_gpu=${GPU}" \
    "seed=66" \
    "mask_rate=${mask_rate}" \
    "epochs=300" \
    "stage_epoch=150" \
    "batch_size=32" \
    "hidden=256" \
    "features=wav2vec-large-c-UTT,deberta-large-4-UTT,manet_UTT" \
    > "${manifest}"

  echo "[start] CMUMOSI seed=66 mask_rate=${mask_rate} host_gpu=${GPU}"
  (
    cd "${run_dir}"
    env PYTHONDONTWRITEBYTECODE=1 \
      CUDA_VISIBLE_DEVICES="${GPU}" \
      COMP_OUTPUT_ROOT="${run_dir}/saved" \
      "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
      python "${WRAPPER}" \
        --dataset=CMUMOSI \
        --audio-feature=wav2vec-large-c-UTT \
        --text-feature=deberta-large-4-UTT \
        --video-feature=manet_UTT \
        --seed=66 --batch-size=32 --epochs=300 --lr=0.0001 \
        --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 \
        --attn_drop_rate=0.0 --stage_epoch=150 --gpu=0 \
        --mask_rate="${mask_rate}" --lbd=0.3 \
        > run.log 2>&1
  )
  echo "[done] CMUMOSI seed=66 mask_rate=${mask_rate}"
}

for mask_rate in "$@"; do
  run_one "${mask_rate}"
done

echo "[complete] CMUMOSI seed=66 assigned mask rates: $*"
