# Approved B2 MOSI launch — 2026-09-09

User approved starting after implementation acceptance (`启动`).

- Seeds: 66,67,68 on GPU0; 69,70 on GPU1 (biggpu). GPU4 excluded.
- Each seed: Stage1 100 epochs → fixed-final latent audit → Stage2 100 epochs.
- Base: corresponding seed of `osram_write_step_train_20260909/mosi`, native write_step=.6.
- P0 inherited, not retrained. No new ablations or other datasets.
- Stage2 uses the same cyclic eight-rate training and **one 8-rate-mean Test-oracle checkpoint per seed** as P0. Internal diagnostic, not formal paper evidence.
- Model implementation: commit d7d02d1. Code SHA256 recorded in each launch log.
- Stage1 projectors/teacher frozen; Stage2 jointly fine-tunes B2 with unchanged loss/LR configuration.
- Output root: `/data2/yb/remote_experiments/osram_b2_20260909/formal/seed_{66..70}`.
- Logs: `/data2/yb/remote_experiments/osram_b2_20260909/formal/seed_N.log`.
- Each seed starts Stage2 only after Stage1 training and audit exit successfully. A failed process stops its own pipeline. Other seeds run independently.
- Directory creation rejects duplicate runs. No checkpoint overwrite/resume.

This authorizes the previously pending full-training step; earlier smoke records remain smoke-only evidence.
