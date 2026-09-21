#!/usr/bin/env bash
set -u

ROOT=/data2/yb/remote_experiments/osram_mosi_base_gap_delta_20260921
REPO=/data2/yb/paper/GCNet_TPAMI_missing_m3_target_ple
PYTHON=/data2/yb/reproduction_envs/s0/bin/python3.10

while true; do
  ready=1
  for gpu in 0 1 2 4 5; do
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
      ready=0
      break
    fi
  done
  if [ "$ready" -eq 1 ]; then
    break
  fi
  echo "CUDA unavailable; retrying in 60s" >&2
  sleep 60
done

rm -rf "$ROOT"
mkdir -p "$ROOT"
cd "$REPO"
exec env PYTHONPATH=. "$PYTHON" -u experiments/osram_mosi_base_gap_delta_20260921/run.py --launch
