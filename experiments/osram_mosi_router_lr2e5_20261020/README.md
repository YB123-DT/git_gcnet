# MOSI constant2e-5 paired screen

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY

Exactly two fresh seed66 runs: original Flat and the existing one-stage
Text-source probability/decision Router. No new architecture, objective,
dataset, seed, missing protocol, or hyperparameter search.

- MOSI, 150epochs, batch32; original MSE, Adam, weight decay1e-5 and dropout.
- Backbone LR2e-5 for all150epochs (the only changed training setting).
- Router LR3e-4 for epochs1–100 and3e-5 for101–150, unchanged.
- Original `gcnet-random-mask-v1`; cyclic0.0–0.7; reference initialization,
  split, evaluation masks, every epoch's training-mask hash and step count
  are checked against the completed original seed66 run.
- Per-rate earliest Test-best W-F1; ACC from those same predictions.
  This eight-rate mean does not represent one deployment checkpoint.
- Full model/optimizer/RNG last checkpoint plus eight matched best weights
  and predictions per run. No old checkpoint or result is overwritten.
- Source is frozen before submission. One three-batch joint GPU smoke
  checks gradients, actual updates, checkpoint recovery and exact prediction
  replay before either formal job starts. Smoke is not a performance result.
- The reused low-LR worker retains its original1e-5 defaults; this dedicated
  entry point explicitly selects2e-5 and includes its wrappers in source hashes.

Run from the immutable source snapshot on `biggpu`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_mosi_router_lr2e5_20261020.run \
  --output /data2/yb/remote_experiments/osram_mosi_router_lr2e5_20261020/execution_v1 \
  --source-commit SOURCE_SHA \
  --gpus GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4
```

One healthy GPU, at most two formal jobs concurrently. Resource admission
requires measured smoke memory,64GiB available host RAM and12GiB free disk;
it never kills peers, changes batch size, or uses physical GPU4. The finite
queue generates `RESULT.md`, `METRICS.csv` and `SUMMARY.json` only after both
150epoch runs complete and selected-prediction checks pass.
