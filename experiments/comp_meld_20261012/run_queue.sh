#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_GPU="${1:-3}"
MAX_PARALLEL="${COMP_MAX_PARALLEL:-3}"
RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7)
failures=0

run_task() {
  local mode="$1" rate="$2"
  if ! "${SCRIPT_DIR}/run_one.sh" "${mode}" "${rate}" "${HOST_GPU}"; then
    echo "[failed] mode=${mode} rate=${rate}" >&2
    return 1
  fi
}

for rate in "${RATES[@]}"; do
  for mode in causal noncausal; do
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
      sleep 10
    done
    run_task "${mode}" "${rate}" &
  done
done

for pid in $(jobs -rp); do
  wait "${pid}" || failures=$((failures + 1))
done
echo "[queue complete] failures=${failures}"
exit "${failures}"
