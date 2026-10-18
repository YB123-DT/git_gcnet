# MELD fixed LR1e-5: finite weight-decay check

INTERNAL TEST-ORACLE SCREEN. No added model block or further adaptive search.

Only Adam weight_decay changes, from the completed2e-4 same-seed reference.
New runs: seeds67/68 × WD `[0,1e-5,1e-4,5e-4]` =8 jobs. Reuse completed WD2e-4
seeds67/68, rather than retraining them. Historical WD1e-3 exists for seed66
only; do not mix it into a seed67/68 or three-seed mean.

| New setting | LR | WD | Seeds |
|---|---:|---:|---|
|WD0|1e-5|0|67,68|
|WD1|1e-5|1e-5|67,68|
|WD2|1e-5|1e-4|67,68|
|WD3|1e-5|5e-4|67,68|

Everything else stays original: MELD ECERC features and official splits,
Flat1600 with Local Skip, cfg84 causal CHM (joint A/T/V write, write_step.6),
no-JEPA emotion-only CE, batch32,50epochs, constant learning rate,
gcnet-random-mask-v1 host/guest padded-rectangle sampling, cyclic .0–.7.
No class reweighting, auxiliary loss, Gate, PCA or Nested module.

Per-rate Test-best **ACC** selection is unchanged. UA is seven-class Macro
Recall from the same selected epoch/mask/predictions, not UA-selected or a fresh
random post-training evaluation. All8 rates, high-missing means and class recall
are preserved. Completion produces all runs and failures, not only a winner.

## Reuse existing implementation

The numeric trainer/model remains the untouched b4a3138 checkout, using original
cmerc-repro Python (Torch1.13.1+cu117 / NumPy1.26.4 / sklearn1.3.2).
ECERC SHA256:
`a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`.

The extension reuses the existing optimizer-screen worker in a fresh process,
adapting only its orchestration configuration, allowed GPU UUIDs, manifest and
same-seed control validation. It does not edit that worker's source or any
numerical function. Each run verifies initialization hash against its own
completed seed67/68 WD2e-4 control. Every epoch verifies the actual optimizer
LR/WD and complete training/eight-rate test mask streams against that same seed.
Cross-seed masks are intentionally different; no cross-seed equality claim.

Controls:
`/home/qinjinghui/remote_experiments/osram_meld_small_lr_3seed_20261018/runs_v1/seed_{67,68}`.

Both shared historical helper hashes and new extension source hashes must match
the immutable snapshot before running. All runs preserve8 BEST checkpoints,
their exact selected masks/logits/labels, and full model/Adam/RNG/progress state.
No protected existing checkpoints or data are overwritten/deleted.

## Resources and launch

biggpu healthy GPU0 andGPU2, max4 jobs each, bound by actual UUID. GPU4 is
forbidden. Prior identical-model runs measured about0.6GiB peak torch allocation
per worker; reserve4GiB per job for admission, without changing batch size.
No redundant smoke/control retraining. Other users' tasks are not changed.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_fixed_lr_wd_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_meld_fixed_lr_wd_20261018/runs_v1 \
  --driver-commit SOURCE_COMMIT
```

Run from the immutable deployed helper snapshot, including the original shared
root SOURCE_MANIFEST.json. Actual source SHA/commands/UUIDs/PIDs/logs are recorded
in LAUNCH.md and the remote queue. Use explicit --resume only after checking the
existing controller and preserving source/configuration. OOM retries may reduce
concurrency once, never change hyperparameters. No automatic extra seeds or WD
search follows these8jobs. Two-seed means are not labeled three-seed results.
