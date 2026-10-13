#!/usr/bin/env bash

set -euo pipefail

REPOSITORY="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUN_ONE="${REPOSITORY}/experiments/comp_causal_mosei_3seed_20261013/run_one.sh"
OUTPUT_ROOT="${COMP_MOSEI_OUTPUT_ROOT:-/data2/yb/paper/remote_runs_20261012/results/comp_causal_mosei_3seed_100e_20261013}"
GPU_UUID="${COMP_GPU_UUID:-GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a}"
CONTROL_FILE="${OUTPUT_ROOT}/MAX_PARALLEL"
STATUS_FILE="${OUTPUT_ROOT}/dispatcher.status"
mkdir -p "${OUTPUT_ROOT}"
[[ -f "${CONTROL_FILE}" ]] || printf '6\n' > "${CONTROL_FILE}"

declare -a TASKS=()
for seed in 66 67 68; do
  for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
    TASKS+=("CMUMOSEI ${seed} ${rate}")
  done
done

running=0; completed=0; failed=0; next=0
declare -A PIDS=()
write_status() {
  printf 'updated_at=%s\nqueued=%d\nrunning=%d\ncompleted=%d\nfailed=%d\nmax_parallel=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(( ${#TASKS[@]} - next ))" \
    "${running}" "${completed}" "${failed}" "$(<"${CONTROL_FILE}")" > "${STATUS_FILE}"
}
while (( next < ${#TASKS[@]} || running > 0 )); do
  max_parallel="$(<"${CONTROL_FILE}")"
  [[ "${max_parallel}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid MAX_PARALLEL" >&2; exit 3; }
  while (( next < ${#TASKS[@]} && running < max_parallel )); do
    read -r dataset seed rate <<< "${TASKS[next]}"
    "${RUN_ONE}" "${dataset}" "${seed}" "${rate}" "${GPU_UUID}" \
      > "${OUTPUT_ROOT}/dispatch_${dataset}_seed${seed}_mr${rate}.log" 2>&1 &
    pid=$!; PIDS["${pid}"]="${dataset} seed=${seed} rate=${rate}"
    ((next += 1)); ((running += 1)); write_status; sleep 3
  done
  reaped=0
  for pid in "${!PIDS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then ((completed += 1)); else ((failed += 1)); fi
      unset 'PIDS['"${pid}"']'; ((running -= 1)); reaped=1
    fi
  done
  write_status
  (( reaped == 1 )) || sleep 20
done
write_status
(( failed == 0 ))
