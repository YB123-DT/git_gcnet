# Actual launch — 2026-10-27

**Running, not completed.** Authorized three structures only,24 formal jobs.
No extra seed, model combination, parent training or new LR setting was started.

- Training source: `1b74bd2db763971a7c6b8aff5a77d335521e3ace`, pushed to feature/ecerc-features.
- Server/runtime: ssh biggpu, `/data2/yb/conda_envs/cmerc-repro/bin/python`.
- Numerical cwd: `/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c`.
- Immutable overlay: `/data2/yb/remote_experiments/osram_mosei_frozen_nested_capacity_20261027/source_1b74bd2`.
- Output: `/data2/yb/remote_experiments/osram_mosei_frozen_nested_capacity_20261027/execution_v1`.
- Controller log: `/data2/yb/remote_experiments/osram_mosei_frozen_nested_capacity_20261027/controller_v1.log`.
- tmux session: `mosei_nested_capacity_20261027`.
- At verification:12formal running,12pending; GPU0/1/2 each maximum4.
  Exact PIDs,commands,UUIDs,cwd,logs are in LAUNCH.json; remote QUEUE.json is live.

```sh
cd /data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c
PYTHONPATH=/data2/yb/remote_experiments/osram_mosei_frozen_nested_capacity_20261027/source_1b74bd2:$PWD \
 /data2/yb/conda_envs/cmerc-repro/bin/python -u \
 -m experiments.osram_mosei_frozen_nested_capacity_20261027.scheduler \
 --root /data2/yb/remote_experiments/osram_mosei_frozen_nested_capacity_20261027/execution_v1 \
 --source-commit 1b74bd2db763971a7c6b8aff5a77d335521e3ace --per-gpu 4
```

## Completed checks

19scoped CPU tests passed locally;6pure protocol/queue/report tests passed in the
remote cmerc environment. Model+worker+original-wrapper/LR regressions passed
separately. Compileall/diff checks and independent integration review passed.
No core implementation or protected old result was edited.

| Variant | Actual params | Real GPU check | Wall seconds, including preparation | Reported torch peak MiB |
|---|---:|---|---:|---:|
|dim32, explicit MLP64|80163|smoke_passed|42.450|411.387|
|MLP128|216899|smoke_passed|43.353|470.784|
|depth2|146690|smoke_passed|37.776|423.155|

Each GPU check performs2single-batch updates, exact protected parent and selected
mask replay, zero-initialized Flat prediction equality, configured-seed tensor
reproducibility, actualarchitecture/parameter/LR checks, finite only-block
gradients and updates, original parent unchanged, one live CHM scan, strict best
reload and exact saved prediction replay. This is correctness, not a performance
experiment. Full-epoch mask equality is not claimed from two smoke batches.

Formal training has started; ten workers had already completed their first
71-update full epoch at verification, with original sample/mask/update traces
matching the protected default Nested. Other workers remained in their first
epoch; no failure was observed. Exact counts are a dated launch observation.

GPU0/1/2 were healthy and idle before launch; about459GiB RAM/301GiB disk available.
GPU4 was excluded by actual UUID checks. Existing work on GPU5/6 was untouched.
Resource admission retains conservative2GiB VRAM/8GiB RAM per-worker floors plus
reserves and uses actual free space; it never changes batch32 to fit.

## Pending

All24complete60epoch outcomes, final variant ranking and independent selected
artifact aggregation are still pending. The persistent controller automatically
fills released slots and generates RESULT.md/JSON/CSV when all24finish. It does
not automatically add combined structures or more seeds. Existing Flat/default
Nested/LR screen checkpoints, exact masks and results remain protected.
