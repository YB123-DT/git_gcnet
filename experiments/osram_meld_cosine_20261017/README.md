# Single MELD Flat warmup/cosine control

INTERNAL TEST-ORACLE SCREEN. One authorized from-scratch seed66 run, not HPO.

## Scope and paired control

The completed constant control is biggpu
`/data2/yb/remote_experiments/osram_meld_write_tuning_20261017/runs_v1/W0`.
Its ACC-selected mean ACC/UA is 60.838123/35.632827 percent.
Accepted historical three-seed baseline checkpoints are untouched.

The immutable numerical source is commit
`b4a313841f098487a6482d2712bdac94aa8e319e`, already used for W0.
The new driver reuses the five-run driver's exact capture and complete-state
recovery utilities; it does not edit either source directory or running jobs.
Effective config is copied from W0 with **only** `lr_schedule="cosine"` changed.
MELD ECERC data SHA256 remains
`a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`.

Unchanged: 50 epochs, batch32, Adam peak LR .001, weight decay1e-5,
dropout .5, seed66, cyclic0.0–0.7, gcnet-random-mask-v1,
emotion-only/no-JEPA, Flat1600 with original Local Skip,
eight heads/KV64, causal read-before-write, joint Block Delta Write,
write_step .6, write_ridge .001. No additional block, loss or data augmentation.

## Exact schedule

Use the existing `_apply_epoch_learning_rate`, not a new scheduler formula.
Existing dormant `warmup_ratio=.05` now takes effect:
`max(1, round(.05*50))=2` warmup epochs (Python rounding).
Epoch1 LR .0005; epoch2 .001; epoch3–50 cosine decline to0.
This tests the **warmup + cosine package**, not isolated cosine. The last
epoch has LR0 under the existing convention; it is not silently changed.
All actual optimizer group rates are checked and saved every committed epoch.
Scheduling uses config/epoch plus saved optimizer base rates; no external
scheduler object or additional optimizer step is introduced.

## Metrics and recovery

Primary selection remains per-rate **test ACC-best**. Independently retain
per-rate **test UA-best**, including the corresponding ACC/WF1/CE and recalls.
Never combine maximum ACC and maximum UA from different checkpoints.
Both selectors use the same original evaluation forward; no extra inference,
sampler draw or test-mask regeneration is added. Ties keep the first epoch.

Capture exact masks, IDs, logits, labels, selected epoch and hashes.
Each epoch checks actual train and eight test selection-mask hashes against
the same epoch of completed W0; initialization is checked against W0 too.
Do not equate distinct rates' selected checkpoints or selected-epoch masks.

`best_miss_*.pt` is ACC-best. `ua_best/best_miss_*.pt` is UA-best.
`EVALUATION_HISTORY.json` includes all eight metrics for every epoch;
`LEARNING_RATES.json` contains all actual group rates.
`last_training.pt` retains optimizer/RNG and additional UA selection references
and histories atomically. Recovery repairs UA-best canonical files from
committed hashed versions, never from an uncommitted sidecar. Unreferenced
version cache pruning is restricted to this new run's own directories.

W0 did not retain every epoch's UA or UA-best weights. Consequently C0 UA-best
versus W0 ACC-best is not a selection-matched performance comparison. The
clean schedule comparison is C0 ACC-best versus W0 ACC-best.

## Execution

Server biggpu; environment `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Only physical GPU5, UUID `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`;
process-visible cuda:0. Never GPU4. One additional training process only.
Queue reuses measured Flat peak with conservative4GiB GPU admission and
3GiB output budget plus2GiB disk margin. No batch or architecture reduction.

From the immutable driver root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_cosine_20261017/run.py \
  --model-source /data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138 \
  --control /data2/yb/remote_experiments/osram_meld_write_tuning_20261017/runs_v1/W0 \
  --output /data2/yb/remote_experiments/osram_meld_cosine_20261017/runs_v1
```

Use `--resume` only with the same immutable driver and complete-state queue.
The controller stops after this single job and writes RESULTS.json, including
the constant reference. A tmux session makes it survive SSH disconnection.
Status/launch evidence is separate from final training results.

## Verification

Tests first observed five failures for the absent implementation, then checked
schedule-only config, all50 LR values against the existing function,
same-forward UA artifacts and weights, first-tie handling, RNG invariance,
observer recovery, rejection of incomplete evaluation sets and GPU5-only queue.
Regression tests cover the reused original capture/config paths. Real GPU
training's first committed epoch is the end-to-end optimizer/save check;
not an additional smoke model and not evidence of improved performance.
