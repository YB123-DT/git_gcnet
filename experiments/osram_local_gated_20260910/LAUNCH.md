# Local-gated MOSI launch — 2026-09-10

User authorized training after implementation verification. Model implementation: `3379513`.

Five fresh runs, 100 epochs, cyclic missing rates 0.0–0.7, forward-only OSRAM,
write step 0.6. Each seed inherits its existing Flat configuration, changing only
`osram_readout_fusion=local-gated` and `checkpoint_selection=test-oracle-per-rate`.
No baseline retraining, checkpoint warm start, B2, loss or learning-rate change.

| Seed | GPU on biggpu | PID |
|---|---|---|
| 66 | 2 | 2280099 |
| 67 | 2 | 2280100 |
| 68 | 2 | 2280101 |
| 69 | 3 | 2280102 |
| 70 | 3 | 2280103 |

Detached supervisor PID: 2279948. Three jobs on GPU 2, two on GPU 3.
Output root: `/data2/yb/remote_experiments/osram_local_gated_20260910`.
Logs: `seed66.log` through `seed70.log`; process manifest: `QUEUE.json`.
Run artifacts: `mosi/seed_N/`. Every run stores configuration and source/reference hashes.

Flat reference: `/data2/yb/remote_experiments/osram_write_step_train_20260909/mosi/seed_N`.
Extract Flat per-rate maxima from the existing complete 100-epoch histories, **not**
the former eight-rate-mean checkpoint. Match test mask hashes after completion.

Selection: **each seed × rate independently selects its best Test W-F1 epoch**;
ties keep the earliest epoch. Each rate has its own saved checkpoint. Logged eight-rate
means are descriptive only and never select a checkpoint in these new runs.

**INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.**

Pre-launch evidence: 258 tests and a real one-batch smoke already passed (see
VERIFICATION.md); these were not rerun. Launcher validates all five inherited configs
before starting processes and refuses existing run directories/queue manifests.
All five processes loaded MOSI features; seeds 69/70 had completed epoch 1 at initial
launch inspection. Final accuracy/convergence is not yet known.
