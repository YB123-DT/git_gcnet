# MOSEI Flat128 / Nested finite overnight screen

INTERNAL TEST-ORACLE SCREEN

Approved scope:24 result slots,23 fresh runs. Reuse completed Flat128 seed66;
train Flat128 seeds67/68 and21 existing Nested configurations at seed66.
The exact structure table and motivation are in
`../osram_mosei_flat128_nested_plan_20261024/PLAN.md`.

All runs: cmerc-repro, MOSEI,60epochs, constant1e-4, Adam, batch32, WD1e-5,
MSE, global clipping1, original split and GCNet random-mask-v1 cyclic0–.7.
Flat hidden128/output1600; original Local Skip and8x64 CHM/write_step.6.
Only existing Nested configuration registrations are added; no algorithm changes.
Graph groups2/4/8 are not CHM Memory head counts.

Per-rate TEST-best W-F1 (earliest tie), with ACC/MSE from the identical selected
predictions/masks. This is exploratory, not a single deployable checkpoint or
independent test claim. Compare each Nested66 to Flat128 seed66; report Flat
three-seed statistics separately, never as a paired control for one Nested seed.

## Execution

Copy the accepted immutable capacity snapshot, overlay this committed runner,
its tests and the21 configuration-only registry entries. Regenerate its complete
SOURCE_SNAPSHOT.json and retain the original manifest. The queue checks that
all numerical model/sampler files except those exact registrations are unchanged.

Run the bounded resource/correctness probe first:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_mosei_flat128_nested_20261024.probe \
  --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
  --source-commit SOURCE_SHA --output ROOT/probe_v1
```

It profiles N01/N20/N21 and the fresh Flat67 initialization path using real
maximum-padding/maximum-valid batches, checks zero-init equivalence, finite
task gradients through the Nested core after decoder startup, one Memory scan
per update and exact full-model/Adam/RNG recovery. These are not model scores.

Then run persistently under tmux from the immutable source directory:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_mosei_flat128_nested_20261024.run \
  --output ROOT/execution_v1 --source-commit SOURCE_SHA \
  --resource-proof ROOT/probe_v1/RESOURCE_PROOF.json
```

GPU0/1 UUID-bound, at most6 own workers per card/12 globally, including any live
old capacity workers. Measured reservations, foreign occupancy, RAM and disk
can reduce concurrency. Never use biggpu GPU4, signal others' jobs, change batch
to fit, silently retry failures or overwrite existing outputs. Resume is explicit
`--resume` with complete last-model/Adam/CPU+CUDA RNG and identical source/config.

QUEUE.json records each PID/command/UUID/log/state. Each run retains eight BEST
weights, exact matching predictions/masks, history/config/identity and full last
state. REPORT.json, RESULT.md and PER_RATE.csv update automatically; only complete,
integrity-checked results enter rankings. Final RESULT.json requires all24 slots.
Ranking: eight-rate W-F1, then high-missing W-F1, then lower parameter count.
No automatic extra seed promotion, new methods or HPO.

Wall average seconds/epoch includes setup/train/evaluation/saving. The reused
training loop does not provide training-only per-epoch timing; do not invent it.
All protected historical and new-missing checkpoints remain untouched.
