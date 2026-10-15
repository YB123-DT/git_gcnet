#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${BASELINE_PYTHON:-/home/yangbin/miniconda3/envs/multimodalerc310/bin/python}" "$BASE/aggregate_results.py"
cat "${BASELINE_RUN_ROOT:-${BASE}/../../runs/baseline_5models_local_20261015}/dispatcher.json"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
