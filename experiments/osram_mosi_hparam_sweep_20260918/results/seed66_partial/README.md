# Seed-66 partial screening

This directory contains the first 60 completed configurations from the MOSI
causal-OSRAM hyperparameter screen. The remaining 30 configurations were
queued separately and are not included in these files.

- Training protocol: cyclic missing-rate schedule.
- Model selection: one Test-oracle epoch per seed and missing rate.
- Seed: 66 only.
- Status: internal diagnostic only; not a formal paper result.
- `screening_summary.csv` is sorted by the descriptive eight-rate mean.
- `summary.json` records that this is a partial `60/90` summary.

The raw checkpoints and per-epoch artifacts remain on the remote experiment
node and are intentionally not committed to Git.
