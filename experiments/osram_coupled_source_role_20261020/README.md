# Coupled Source Role: Base–Gap Partner Read

INTERNAL TEST-ORACLE SCREEN — candidate method, not proven improvement.

User design: keep original CHM, Local/Skip, queries, joint write, read-before-
write, Flat1600, emotion norm and scalar task head. The existing tracker gives
four read roles × three additive joint Value-slot sources. It does NOT produce
three independent pure-modality memories or semantic causal attributions.

For each source s, Base's partner is the mean of the currently active Gap
source reads; each Gap's partner is the same-source Base read. A shared
215→64→GELU→1 MLP sees64-d Local/read/partner projections, role/source
embeddings8 each, current availability3 and strict-past trajectory4. Last
layer zero-init; gamma=.25*tanh(score). Add sum_s(gamma*source_read) to the
corresponding original forward Base/Gap slot. Never modify the backward zero
halves or original Local Skip input. Safe masking occurs before partner
pooling/projection and again at features/gamma/output.

Opt-in `--osram-source-role` / `TrainConfig.osram_source_role=True`, default
False. New module initialization uses fork_rng. Existing strict checkpoints
load unchanged with flagFalse. True adds96,487 parameters (default256/512/64).
The existing joint source capture supplies differentiable Base AND Gap to
joint training. The shared Memory remains the original authoritative state.
`osram_tasr` and other incompatible blocks are rejected, not stacked.

## Finite experiment

| Setting | MOSI | MOSEI |
|---|---|---|
| Seed / epochs |66 /150|66 /60|
| Backbone AND new-module LR |1e-3 epochs1–100;1e-4 epochs101–150|constant1e-3|
| Loss |original MSE only|original MSE only|
| Optimizer / batch / wd |Adam /32 /1e-5|Adam /32 /1e-5|
| Missing |original gcnet-random-mask-v1 cyclic0.0–0.7|same|
| Selection |per-rate earliest Test-best W-F1|same|
| Initialization |original seed66 fold5 initialization quirk retained|original seed66 fold1|

One stage, all original trainable parameters and the new module train together.
No freezing, source-neutral control training, persistent view, Router,
InfoNCE, JEPA, completion or auxiliary loss. No automatic seeds or search.
The separate2e-5 experiment is not the backbone or comparator for this run.

Existing original Flat controls are reused read-only:

```
/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_66/flat
/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat
```

Worker guards same original-parameter initialization hash, effective config
(only source-role switch differs), splits, evaluation masks and seed protocol.
Every full epoch must match the original Flat's actual training-mask hash,
optimizer steps and learning rate; task losses are NOT required to match a
different architecture. Old core source hashes change for opt-in integration,
so numerical baseline regressions and actual zero-init prediction equality
are checked rather than falsely claiming identical old core files.

Runtime stays `/data2/yb/reproduction_workspace/envs/s0/bin/python` on biggpu:
Torch2.2.2+cu121,NumPy1.24.3,sklearn1.3.2. Original data manifests and files
are SHA-checked by reused dataset-specific setup. No new split/holdout.

## Run

Two independent resource-admitted controllers, one per approved dataset,
each perform one short real dataset-specific GPU smoke before their ONLY
formal seed66 job. At most two formal jobs in total; never GPU4 or peer kills.
Use a frozen source snapshot and persistent tmux sessions; replace SOURCE_SHA
and select the verified healthy physical GPU UUID:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_coupled_source_role_20261020.run --dataset CMUMOSI \
 --output /data2/yb/remote_experiments/osram_coupled_source_role_20261020/execution_v1/CMUMOSI \
 --source-commit SOURCE_SHA --gpus GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e

/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_coupled_source_role_20261020.run --dataset CMUMOSEI \
 --output /data2/yb/remote_experiments/osram_coupled_source_role_20261020/execution_v1/CMUMOSEI \
 --source-commit SOURCE_SHA --gpus GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
```

Only explicit `--resume` permits exact full-state recovery, with unchanged
source/config/mask identity. No automatic failure retry or protocol edits.
Each run saves eight BEST checkpoints, same-call selected labels/predictions/
masks and full model+Adam+RNG last.pt. Runtime masks do not resample on report.
Queue mode`joint` is a compatibility label; scalar metric key`router` is the
legacy alias for the PRIMARY prediction. This model instantiates NO Router,
and saved candidate copies are not alternative predictions or extra heads.

Final paired ACC/W-F1 per-rate,mean andhigh tables are generated only after
full epochs and selected-prediction integrity checks. Eight-rate means can
combine eight checkpoints, not one deployable model. New-module gradients,
gamma/delta norms and source-decomposition errors are logged explicitly as
the LAST TRAINING BATCH of each epoch, not mislabeled as epoch-global means.
Smoke verifies finite/nonzero new-parameter updates, same baseline initial
predictions, source errors, checkpoint recovery and exact replay; it is not
evidence that the method improves sentiment prediction.
