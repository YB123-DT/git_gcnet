#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUN_ROOT="${1:?new output root required}"
RUN_ONE="${REPOSITORY}/experiments/comp_meld_20261012/run_one.sh"
RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7)
SEEDS=(66 67 68)
GPU_UUIDS=(
  GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45
  GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049
  GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4
)

if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}/logs"

failures=0
run_seed() {
  local seed="$1" gpu_uuid="$2"
  local rate
  for rate in "${RATES[@]}"; do
    COMP_SEED="${seed}" \
    COMP_EPOCHS=50 \
    COMP_STAGE_EPOCH=25 \
    COMP_KEEP_CHECKPOINTS=0 \
    COMP_MASK_PROTOCOL=gcnet-legacy-fixed-v1 \
    COMP_MELD_RUN_ROOT="${RUN_ROOT}/runs" \
      "${RUN_ONE}" causal "${rate}" "${gpu_uuid}"
  done
}

printf '%s\n' \
  "repository_commit=$(git -C "${REPOSITORY}" rev-parse HEAD)" \
  "mask_protocol=gcnet-legacy-fixed-v1" \
  "selection=per-rate Test-oracle; internal diagnostic" \
  "epochs=50" "stage_epoch=25" \
  > "${RUN_ROOT}/DISPATCH.txt"

pids=()
for index in 0 1 2; do
  run_seed "${SEEDS[$index]}" "${GPU_UUIDS[$index]}" \
    > "${RUN_ROOT}/logs/seed${SEEDS[$index]}.log" 2>&1 &
  pids+=("$!")
  printf 'seed=%s gpu_uuid=%s pid=%s\n' \
    "${SEEDS[$index]}" "${GPU_UUIDS[$index]}" "${pids[-1]}" \
    >> "${RUN_ROOT}/DISPATCH.txt"
done

for pid in "${pids[@]}"; do
  wait "${pid}" || failures=$((failures + 1))
done
printf 'failures=%s\n' "${failures}" >> "${RUN_ROOT}/DISPATCH.txt"
exit "${failures}"
