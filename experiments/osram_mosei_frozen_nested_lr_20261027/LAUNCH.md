# Actual launch record — 2026-10-27

**Running, not finished.** User approved the two-priority-LR screen only.
No new seed or structure search was started. Actual immutable training source:
`bfcdbaff89ce33ec2276161c9ff307b8fae19cee` (already pushed to feature/ecerc-features).

- Server: ssh biggpu, `/data2/yb/conda_envs/cmerc-repro/bin/python`.
- Cwd: `/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c`.
- Overlay: `/data2/yb/remote_experiments/osram_mosei_frozen_nested_lr_20261027/source_bfcdbaf`.
- Output: `/data2/yb/remote_experiments/osram_mosei_frozen_nested_lr_20261027/execution_v1`.
- Controller log: `/data2/yb/remote_experiments/osram_mosei_frozen_nested_lr_20261027/controller_v1.log`.
- tmux: `mosei_nested_lr_20261027`.
- GPU0/1/2 UUIDs and actual process IDs/commands/logs: LAUNCH.json snapshot;
  the remote QUEUE.json is the live authority. Maximum4/card and12global.
- Registered new formal jobs:16; at launch verification12running/4pending.
  All12 registered optimizer LRs were the requested1e-4 or1e-3, not3e-4.
- PhysicalGPU4 and other tasks were not used or modified.

```sh
cd /data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c
PYTHONPATH=/data2/yb/remote_experiments/osram_mosei_frozen_nested_lr_20261027/source_bfcdbaf:$PWD \
 /data2/yb/conda_envs/cmerc-repro/bin/python -u \
 -m experiments.osram_mosei_frozen_nested_lr_20261027.scheduler \
 --root /data2/yb/remote_experiments/osram_mosei_frozen_nested_lr_20261027/execution_v1 \
 --source-commit bfcdbaff89ce33ec2276161c9ff307b8fae19cee --per-gpu 4
```

## Completed launch checks

15 scoped local tests passed;7pure protocol/scheduler/report tests also passed
inside the remote cmerc environment. Worker plus reused stage2/cross regression
tests passed separately. Compileall and diff checks passed. Independent review
caught a Flat-row metadata issue before deployment; its failing regression was
fixed before the immutable training snapshot was created.

Both real GPU smokes passed:

| Branch LR | Budget | Wall seconds | Peak allocated MiB | Status |
|---|---|---:|---:|---|
|1e-4|2 single-batch updates|42.795|449.713|smoke_passed|
|1e-3|2 single-batch updates|43.191|449.713|smoke_passed|

Verified original selected parent predictions/masks, bit-exact original Nested
initial state,159235trainable parameters, actual all-group LR, connected finite
gradients/parameter updates, frozen parent unchanged, one live CHM scan and exact
selected checkpoint reload. Smoke validates the rate/step prefix only, not a
full training-epoch mask trace. Formal workers verify full71-update epoch traces
against the protected completed3e-4 Nested reference as training proceeds.

At launch, GPU0/1/2 were healthy/idle, over476GiB RAM and304GiB disk were available.
The original same-architecture formal reference peaked near1GiB VRAM/4GiB RSS;
controller uses conservative2GiB/8GiB floors plus reserves and actual free space.
Initial12workers used about6.2–6.6GiB/card at99%GPUutilization. No batch reduction.

## Pending

60epochs per formal job, selected same-mask W-F1/ACC results, full-state/frozen
checks and final independent16-job aggregation are **not yet complete**.
The controller automatically starts pending jobs and writes final RESULT files
when all16 finish; no claim of performance improvement is made by the smokes.
Old Flat/defaultNested results and all checkpoints remain intact.
