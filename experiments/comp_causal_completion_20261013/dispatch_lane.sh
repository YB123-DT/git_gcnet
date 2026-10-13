#!/usr/bin/env bash

set -euo pipefail

GPU_UUID="${1:?physical GPU UUID is required}"
LANE="${2:?lane index is required}"
LANES="${3:-2}"
MAX_PARALLEL="${COMP_MAX_PARALLEL:-6}"

[[ "${LANE}" =~ ^[0-9]+$ && "${LANES}" =~ ^[1-9][0-9]*$ ]] || exit 2
(( LANE < LANES )) || { echo "lane must be smaller than lanes" >&2; exit 2; }
[[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]] || exit 2

REPOSITORY="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
IEMOCAP_RUN="${REPOSITORY}/experiments/comp_causal_iemocap_3seed_20261013/run_one.sh"
MOSEI_RUN="${REPOSITORY}/experiments/comp_causal_mosei_3seed_20261013/run_one.sh"
MOSI_RUN="${REPOSITORY}/experiments/comp_causal_mosi_seed68_20261013/run_one.sh"
STATE_ROOT="${COMP_COMPLETION_STATE_ROOT:-/data2/yb/paper/remote_runs_20261012/results/comp_causal_completion_20261013}"
mkdir -p "${STATE_ROOT}"
STATUS_FILE="${STATE_ROOT}/lane${LANE}.status"

declare -a TASKS=()
index=0
add_task() {
  if (( index % LANES == LANE )); then TASKS+=("$*"); fi
  ((index += 1))
}

for dataset in IEMOCAPFour IEMOCAPSix; do
  for seed in 66 67 68; do
    for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
      add_task iemocap "${dataset}" "${seed}" "${rate}"
    done
  done
done
for seed in 66 67 68; do
  for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
    add_task mosei CMUMOSEI "${seed}" "${rate}"
  done
done
for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
  add_task mosi CMUMOSI 68 "${rate}"
done

running=0; completed=0; failed=0; next=0
declare -A PIDS=()
write_status() {
  printf 'updated_at=%s\nqueued=%d\nrunning=%d\ncompleted=%d\nfailed=%d\nmax_parallel=%d\ngpu_uuid=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(( ${#TASKS[@]} - next ))" \
    "${running}" "${completed}" "${failed}" "${MAX_PARALLEL}" "${GPU_UUID}" \
    > "${STATUS_FILE}"
}

while (( next < ${#TASKS[@]} || running > 0 )); do
  while (( next < ${#TASKS[@]} && running < MAX_PARALLEL )); do
    read -r family dataset seed rate <<< "${TASKS[next]}"
    case "${family}" in
      iemocap) command=("${IEMOCAP_RUN}" "${dataset}" "${seed}" "${rate}" "${GPU_UUID}") ;;
      mosei) command=("${MOSEI_RUN}" "${dataset}" "${seed}" "${rate}" "${GPU_UUID}") ;;
      mosi) command=("${MOSI_RUN}" "${seed}" "${rate}" "${GPU_UUID}") ;;
      *) exit 3 ;;
    esac
    "${command[@]}" > "${STATE_ROOT}/lane${LANE}_${dataset}_seed${seed}_mr${rate}.log" 2>&1 &
    pid=$!; PIDS["${pid}"]="${dataset} seed=${seed} rate=${rate}"
    ((next += 1)); ((running += 1)); write_status; sleep 2
  done
  reaped=0
  for pid in "${!PIDS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then ((completed += 1)); else
        echo "[failed] ${PIDS[${pid}]}" >&2; ((failed += 1))
      fi
      unset 'PIDS['"${pid}"']'; ((running -= 1)); reaped=1
    fi
  done
  write_status
  (( reaped == 1 )) || sleep 20
done

write_status
(( failed == 0 ))
