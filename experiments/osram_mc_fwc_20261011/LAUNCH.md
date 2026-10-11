# Formal MOSI launch

Status: running, not a completed performance result.

- Server: `ssh biggpu` (`user22`).
- Code commit: `e9a7949`, baseline `c838e14`.
- Immutable deployed source: `/home/qinjinghui/yb_gcnet_experiments/osram_mc_fwc_20261011/source_e9a7949`.
- Results: `/home/qinjinghui/yb_gcnet_experiments/osram_mc_fwc_20261011/formal_e9a7949`.
- Controller PID: `3423806`; launched `2026-10-11T10:39:15Z`.
- Controller log: `/home/qinjinghui/yb_gcnet_experiments/osram_mc_fwc_20261011/formal_e9a7949.controller.log`.
- Finite queue: E0–E4 × seeds66/67/68 ×150epochs =15runs, not a sweep.
- GPUs1/2, two concurrent jobs/card; UUIDs recorded in remote `LAUNCH.json`.
- GPU4 prohibited; GPU7 receives no new jobs. Existing other jobs untouched.
- No GitHub push or merge performed.

Controller is detached from SSH and automatically admits pending jobs as capacity
is released. It records every command/PID/output and will not automatically rerun
failures. Completion requires each run's COMPLETED.json and controller SUMMARY.json;
a started process or smoke report is not a completed experiment.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  /home/qinjinghui/yb_gcnet_experiments/osram_mc_fwc_20261011/source_e9a7949/experiments/osram_mc_fwc_20261011/run_suite.py \
  --root /home/qinjinghui/yb_gcnet_experiments/osram_mc_fwc_20261011/formal_e9a7949 \
  --source /home/qinjinghui/yb_gcnet_experiments/osram_mc_fwc_20261011/source_e9a7949 \
  --dataset-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
  --commit e9a7949 --gpus 1 2 --per-gpu 2
```

Do not resubmit this command: that suite already exists. Recovery of a failed run
must inspect its last.pt and log first, then explicitly use the same snapshot and
configuration with `train_mc_fwc --resume`. Do not alter the deployed source.
