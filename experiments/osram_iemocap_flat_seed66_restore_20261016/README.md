# Original Flat seed66 restoration

This finite queue retrains only original Flat on IEMOCAPFour and IEMOCAPSix,
seed66, five folds each: ten fold jobs at 100 epochs. Fold order is 5,1,2,3,4,
interleaving the two datasets. No B6, other seed, Nested, Gate or JEPA job is
created. Discarded historical weights are not loaded. The new worker may
resume this experiment's own compatible complete `last_training.pt` after
interruption.

Every job reads its own corrected-mask original config from:

```
/data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1/iemocap{4,6}/seed_66/fold_{1,2,3,4,5}/config.json
```

The launcher preserves original fields and validates Flat/Local Skip,
GCNet random missing masks, cyclic rates0.0–0.7, emotion-only training,
batch32, Adam, lr.001, weight decay1e-5, dropout.5 and per-rate Test-best ACC.
Only these explicit bookkeeping/constructor values are set:

| Field | Effective value |
|---|---|
| `preserve_selection_artifacts` | `true` |
| `checkpoint_selection_metric` | `accuracy` |
| `device` | `cuda:0`, after UUID pinning |
| `osram_rpca_lr_multiplier` | `1.0`, neutral with no B6 block |

`osram_meaningful_block` stays `none`. B6 rank/steps/threshold are not added.
No old checkpoint is used to initialize training.

## Deployment and preparation

Run from the new complete immutable implementation checkout, containing this
launcher, the existing transfer helpers, and the updated worker. Numerical
model/trainer code comes from the separate unchanged checkout:

```
/home/qinjinghui/remote_experiments/osram_iemocap_b6_source_785a012
785a01242ef1a82b97d5337acbb33ba963df5537
```

Pass the updated worker file explicitly; every submitted command appends
`--flat-only`. Existing deployed B6 workers are not modified by this queue.

```bash
python -m experiments.osram_iemocap_flat_seed66_restore_20261016.run \
  --worker /absolute/new-implementation/experiments/osram_iemocap_b6_transfer_20261015/worker.py \
  --smoke-summary /absolute/existing-B6-smoke/SUMMARY.json \
  --data-provenance /absolute/DATA_PROVENANCE.json \
  --output /home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016 \
  --check-only
```

`--check-only` prepares an immutable manifest and all ten effective config
snapshots without creating a GPU queue or starting workers. Remove that flag
to run the already-authorized queue using the project's persistent process
mechanism. Record the controller PID, command and log path. Optional
`--baseline-root` and `--source-root` locate the recorded inputs.

No new GPU smoke is run. The existing completed longest IEMOCAPSix B6 batch
measured3333.87MiB allocated /3590MiB reserved and bounds the simpler Flat
path conservatively. Admission uses the larger recorded value plus1024MiB
headroom and reserves complete peaks for active children. The original smoke
summary path/hash/identity are retained in the manifest.

Only physical GPU1 and GPU3 are eligible, with at most four and two jobs
belonging to this queue respectively; at most six total. Actual free memory,
disk and UUID checks can reduce concurrency. The inherited GPU1 total-process
safety ceiling also accounts for external processes. GPUs4/5 are excluded;
external processes are never signalled. Each worker uses a unique output and
explicit UUID pinning.

## Acceptance and retention

A completed result must have exact dataset/seed/fold/config identity,
`method=Original_Flat`, complete finite per-rate ACC/UA/W-F1/CE and a valid
`FIRST_EPOCH_RETENTION_CHECK.json` with:

```
status=complete
best_checkpoints=8
selected_artifacts=8
cuda_rng_states=1
```

The worker checks actual first-epoch retention before continuing. Queue
locking, PID birth identity, collision checks, complete-state RNG recovery,
completed-output reuse and memory admission reuse the existing scheduler.
Failed or invalid jobs never enter aggregates. Keep all80 per-rate BEST
weights, their exact-epoch selection artifacts, and ten full last-training
states. Neither the launcher nor its summary removes weights or artifacts.

`MANIFEST.json` records the source and implementation Git versions, original
config paths/hashes, original and effective settings, exact diffs, worker and
helper hashes, numerical-code hashes, data provenance and environment. Config
snapshots live under `configs/`; scheduler state, provenance and logs live
under `queue/`. Restart rejects changed provenance/configs in the same output.

`SUMMARY.json`, `SUMMARY.csv` and `RESULT.md` refresh after completed workers.
Each dataset gets a fivefold average only after all five valid folds finish;
rates and the high0.5/0.6/0.7 summary are retained. Fold averages are equal
weight. This is seed66 only, without a multi-seed uncertainty estimate.
Selection is internal per-rate test-oracle ACC; UA uses the same selected
logits and masks. Existing B6 seed66 results can be compared separately as
references; this launcher never submits B6 work.

## CPU verification

```bash
python -m pytest -q tests/test_iemocap_flat_seed66_restore.py
python -m py_compile experiments/osram_iemocap_flat_seed66_restore_20261016/run.py
```

Tests cover the ten-job scope and exact config diffs, neutral B6 multiplier,
Flat-only commands, retention-gated acceptance, GPU capacity/memory admission,
fivefold reporting, ten real CPU children, restart reuse, and preparation
without GPU scheduling. They do not start GPU experiments.
