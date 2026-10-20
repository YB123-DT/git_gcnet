# Coupled Source Role: approved design and implementation plan

Goal: implement the user-provided shared Base–Gap Partner Read calibration,
then screen MOSI seed66/150epochs and MOSEI seed66/60epochs, one stage.
PyTorch, existing source tracker and existing scalar MSE/test-best runners only.
The detailed user equations/code are the approved design; do not reopen idea
selection or wait for another approval for reversible implementation steps.

## Fixed design

Source input `[T,B,4 roles,3 sources,forward_dim]` contains Base/A/T/V Gap
Value-slot contributions from the existing joint write. A Base source's
partner is the active-Gap mean of that same source; every Gap source's
partner is the same source's Base read. Use width64 local/read/partner
projections, role/source embeddings8 each, count/coverage/age/ever_seen
metadata, and shared LN→Linear64→GELU→Linear1. Score output zero-initialized;
gamma=.25*tanh(score). Sum gamma*source per role, add to original forward
Base/Gap before unchanged Flat. Local, Local Skip and backward zero halves
are unchanged. No Router, TASR, new head, query, loss or second scan.

Safety adjustment to the supplied core: torch.where-mask invalid inputs BEFORE
projections/partner pooling and mask features/output again. Never infer an
active source from a biased MLP output; strict-past ever_seen is required.
Protect runtime readout ablations from reintroducing disabled evidence.

## Implementation lanes and checks

- [x] Standalone `gcnet_missing_m3/source_role.py` and
  `tests/test_source_role.py`: first test missing implementation; then test
  zero delta, partner formulas, inactive/NaN slots, first turn, padding,
  bounded coefficients, gradients and parameter count.
- [x] Add default-off `osram_source_role` through model, TrainConfig, CLI,
  builder and backbone. Use fork_rng for module initialization; enable
  differentiable existing source Base capture without changing shared memory.
  Add `tests/test_source_role_integration.py`, run it red before integration,
  then green: strict old checkpoint, zero-init logits and RNG parity,
  causal/full/persistent masks, source trace/gradient and runtime ablations.
- [x] Independent experiment runner in this directory reuses existing MOSI
  and MOSEI setup, exact evaluation masks, scalar optimizer/train/evaluate
  utilities and eight-BEST/full-last persistence. No copy of CHM or sampler.
  Compare initial backbone excluding only new parameters; compare actual
  training-mask hash/steps per epoch to completed Flat reference. New config
  differs ONLY by source-role switch; Adam same LR schedule for all weights.
- [x] Two dataset-specific short real GPU smokes, including nonzero new
  parameter updates, exact zero-init baseline, source decomposition error,
  optimizer/RNG checkpoint recovery, and selected prediction replay.
- [x] Tests, compileall and diff checks; bounded independent review for spec
  then quality. Commit source before training; push only own github remote.
  52focused tests pass, including final-checkpoint resume metadata recovery.
  Both independent reviews pass; real GPU checks remain below.
- [x] Frozen snapshot on biggpu; healthy GPU UUID admission (never physical4),
  at most two formal jobs, fresh output directories. Preserve running2e-5
  jobs and all protected weights. Actual formal PIDs/config verified, MOSI
  through9epochs; MOSEI first epoch in progress at launch snapshot.
- [ ] Queue writes per-rate/mean/high ACC/W-F1 and diagnostics only after full
  completion; launch status is not performance. Commit and push launch proof.

## Protocol

MOSI original GCNet seed66 source initialization and dataset split,150epochs,
backbone LR1e-3 first100/1e-4 last50. MOSEI original GCNet seed66 official
split,60epochs,constantLR1e-3. Original Adam/batch32/wd1e-5/dropout/MSE.
New parameters follow the same LR as the backbone, not prior Router LR.
Per-rate Test-best W-F1, same-call ACC/masks/predictions, explicitly
INTERNAL TEST-ORACLE SCREEN; not a single deployment checkpoint.
Existing completed original Flat controls are reused, with exact paired
initialization and mask checks; do not silently use a Router or low-LR control.
No automatic seeds, sweeps, extra datasets or persistent training views.
