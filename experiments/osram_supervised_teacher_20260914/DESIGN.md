# Supervised frozen modality Teacher: two-stage strategy comparison

User-approved scope: retain original ContextualM3Predictor, source/target heads,
MMoE routing, source averaging, SmoothL1/InfoNCE and their weighting. No latent
feedback or memory changes. Student starts from the same random initialization.

1. Stage 1: causal OSRAM .6 mean/Flat, complete inputs (fixed rate 0), emotion-only
   training. Select by complete-view validation W-F1 (existing task metric),
   evaluate_test=False. Record actual epochs, optimizer steps and elapsed time.
2. Export only `observed_set.projectors.*` from selected `best.pt`, never its
   dormant `teacher.*`. Validate source objective/protocol, exact keys/shapes,
   finite tensors and hashes. Preserve provenance and validation selection.
3. Stage 2: `teacher_mode=pretrained-frozen`, joint original JEPA, same cyclic
   rates and per-seed/per-rate Test-oracle rule. Load Teacher only after all
   Student/MMoE initialization. Explicitly skip EMA, exclude Teacher from Adam,
   force eval, assert parameter/buffer hash unchanged before/after training.
4. Keep default `teacher_mode=ema` behavior, keys and RNG unchanged. New mode
   conflicts with Complete/Future/WSC/B2/classification completion and extra
   backbone/readout interventions. No new model parameters or dependencies.

Implementation ownership: model/loader/tests/runner in main lane; trainer,
CLI and lifecycle tests in separate bounded lane. TDD red before implementation.
Verification: wrong-prefix trap, bad keys/shapes/metadata, frozen EMA no-op,
default RNG/state compatibility, gradient isolation and real MOSI smoke.

Only implementation, tests and bounded smoke are authorized in this turn.
No full two-stage/five-seed run. Compare Teacher strategies, not solely the
effect of labels: initial target quality and update policy both change.
