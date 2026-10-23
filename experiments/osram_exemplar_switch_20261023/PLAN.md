# Frozen TRAIN-exemplar switch audit

INTERNAL TEST-ORACLE DIAGNOSTIC. No training or new model.

MOSI66, both completed feature-only and label-aware runs, all eight retained
per-rate BESTs and their exact test masks. ON/OFF share checkpoint, inputs,
Local, original Base/Gap, Flat, normalization and classifier parameters.
Capture original skip/adapter outputs during ON inference; replay OFF from
their sum before the retrieval addition. Primary candidates share ONE CHM
scan. Separately verify OFF against the existing disabled wrapper path.

Report eight-rate/high-missing and per-rate ACC/W-F1/MSE, OFF-to-ON corrections
and harms, prediction changes and residual norm, and compare both to original
same-protocol Flat. OFF minus Flat is a trained-main-path comparison, not a
causal decomposition of the training process. No OFF-specific epoch selection.

- [x] Small red-first metric/alignment/readout tests.
- [x] Reuse training setup, protected TRAIN bank and GCNet evaluation masks.
- [x] Freeze weights; strict selected-checkpoint/data/source/mask integrity.
- [x] Healthy biggpu UUID7, original s0 environment, persistent inference run.
- [x] ON reproduces saved predictions; OFF reproduces explicit disabled path;
      full valid-target coverage, no parameter/RNG updates, originals untouched.
- [x] Complete result/stop decision, commit and push small records only.

Decision: prioritize label-aware. If ON does not improve OFF, stop this retrieval
implementation. If it does but ON does not improve original Flat, only the
direct branch benefit is supported; joint training changes may offset it.
No automatic HPO, seeds, retraining or architecture changes under either result.
