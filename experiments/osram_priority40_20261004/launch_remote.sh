#!/usr/bin/env bash
set -euo pipefail
# Run on biggpu; the immutable source intentionally remains at this code commit.
task_root=/data2/yb/remote_experiments/osram_priority40_20261004
task_snapshot="$task_root/source_9adf59f"
cd "$task_snapshot"
exec env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  "$task_snapshot/experiments/osram_priority40_20261004/dispatch.py" \
  --root "$task_root" --snapshot "$task_snapshot" \
  --controller "$task_root/control/occupied_lane.py" \
  --manifest "$task_snapshot/experiments/osram_priority40_20261004/launch_plan/PART1.json" \
  --manifest "$task_snapshot/experiments/osram_priority40_20261004/launch_plan/PART2.json" \
  --reference-root /data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full \
  --baseline-audit "$task_snapshot/experiments/osram_meaningful20_20261003/BASELINE_AUDIT.json" \
  --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json \
  --cpu-log "$task_snapshot/experiments/osram_priority40_20261004/launch_cpu_evidence.log" \
  --cpu-command 'OMP_NUM_THREADS=1 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_priority40_integration -v' \
  --existing-root /data2/yb/remote_experiments/osram_new40_gpu0123_20261004/attempt2 \
  --existing-root /data2/yb/remote_experiments/osram_new40_gpu6_eleven_20261004 \
  --existing-root /data2/yb/remote_experiments/osram_meaningful20_round2_20261004 \
  --estimated-peak-mib 2000 --poll-seconds 30
