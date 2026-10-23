# MOSI66 TRAIN-exemplar retrieval transfer

INTERNAL TEST-ORACLE SCREEN. No performance claim follows from implementation or smoke.

Bounded to two new one-stage runs: `feature-only` and `label-aware`. The accepted
Flat seed66 reference is reused, not retrained. No extra seeds, HPO or datasets.
TabR-style retrieval uses other TRAIN conversations, never labels from evaluation.
It requires the TRAIN support feature/label bank at inference, unlike a pure
parameter-only Block. CHM still scans the query conversation exactly once.

## Locked protocol

- MOSI official split, seed66; existing `s0` environment, original data hashes.
- 150 epochs, first100 LR `3e-4`, last50 `3e-5`; batch32, Adam, WD `1e-5`.
- Original MSE, original dropout, one jointly trained optimizer group.
- Original `gcnet-random-mask-v1`, cyclic0.0–0.7; no altered Missing or extra loss.
- Exact original test mask replay; per-rate TEST-best W-F1, earliest ties.
- Scores retain same-call predictions, labels, availability, checkpoint and mask hash.
- Training masks/order/LR/update count must match the accepted Flat trace, but task
  loss is expected to change and is not required to match.
- Default model path remains unchanged. New retrieval uses a standalone wrapper.
- Entire same TRAIN conversation excluded from the support candidates.

Original accepted reference:

`/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003`

## Entrypoints

Run from an immutable source snapshot, bind healthy physical GPU UUIDs. Host GPU4
on biggpu and its known failed UUID are rejected by the existing health gate.
The runner owns exactly two jobs, admits concurrency using measured smoke peaks,
and does not signal any other process.

```bash
PYTHON=/data2/yb/reproduction_workspace/envs/s0/bin/python
"$PYTHON" -m experiments.osram_exemplar_mosi_20261023.run \
  --output /data2/yb/remote_experiments/osram_exemplar_mosi_20261023/execution_v1 \
  --source-commit SOURCE_SHA --gpu-uuids GPU_HEALTHY_UUID --per-gpu 2
```

The persistent controller serializes two real-data smokes before admitting the
two formal runs. Start the controller inside tmux; it writes `QUEUE.json` with
PIDs, process start ticks, commands, logs, GPU UUIDs and completion status.

Explicit recovery reattaches live worker PIDs or restores compatible full-state
training checkpoints; it cannot silently repeat completed runs:

```bash
"$PYTHON" -m experiments.osram_exemplar_mosi_20261023.run \
  --output SAME_OUTPUT --source-commit SAME_SHA \
  --gpu-uuids SAME_UUID --per-gpu 2 --resume
```

`last.pt` stores model, Adam, epoch/history, RNG and exact identity, including the
TRAIN bank manifest. A resumed run rebuilds the bank from TRAIN and verifies the
manifest before restoring. Eval caches must invalidate on training/state reload.
`best_miss_*.pt` and matching `predictions_miss_*.npz` preserve original selection.
The compatibility key `router` in metric JSON refers to the single task prediction;
there is no added prediction-mixture Router or extra classifier.

Reports are refreshed without new inference:

```bash
"$PYTHON" -m experiments.osram_exemplar_mosi_20261023.report --output OUTPUT
```

Report `RESULT.md`, `REPORT.json`, `METRICS.csv`; only completed runs enter ranking.
No training result exists merely because source files or a smoke have completed.
