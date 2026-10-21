# V3 finite overnight transfer

INTERNAL TEST-ORACLE SCREEN

Only the already implemented V3 is enabled. No new block, hyperparameter search,
new seeds, Router, Nested, JEPA or dual views. Existing MOSEI seed66 continues in
its original queue and s0 environment; this controller never resubmits it.

New work: MELD seed66, and IEMOCAP-4/6 seed66 each with all five LOSO folds
(11 training jobs). Original classification runtime cmerc-repro, CE, batch32,
Adam, WD1e-5 and cyclic GCNet random-mask-v1 remain unchanged. MELD uses50epochs
and LR1e-5; IEMOCAP4/6 use100epochs and LR1e-3. Per-rate TEST ACC selects the
checkpoint; UA/WF1/CE are from the same saved prediction and exact mask.

MELD replays accepted baseline selected masks (mask arrays only, no baseline
weights). IEMOCAP-6 replays the corrected Flat fixed evaluation banks. IEMOCAP-4
uses a regenerated legal seed66 epoch0 bank and records that distinction;
historical per-epoch random-selected results are not strict paired controls.
The original mask generator is unchanged; evaluation-bank replay is explicit.

Physical GPU0 cap4; GPU2 cap8. Caps are not guarantees of fitting: measured
GPU peaks plus headroom, available RAM and disk gate each submission. GPU4 is
forbidden. Other tasks and all old checkpoints remain untouched. Three short
real-data smokes precede training; previous V3 mathematical tests are reused.

Run in a committed immutable source snapshot on biggpu:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_v3_classification_20261021.run \
  --source-commit SOURCE_SHA \
  --output /data2/yb/remote_experiments/osram_v3_classification_20261021/execution_v1
```

Use tmux for persistence. QUEUE.json records pending/running/completed/failed
jobs and commands. REPORT.json and RESULT.md update automatically. A failed
worker stops further admissions, without killing already running peers. Each
worker retains eight matching BEST artifacts and a full last checkpoint.
Source snapshots remain unchanged while runs execute. No performance claim is
made by implementation, smoke success or queue startup.
