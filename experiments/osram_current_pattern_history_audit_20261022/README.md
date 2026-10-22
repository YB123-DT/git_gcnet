# Fixed-history current-pattern and history-content audit

INTERNAL DIAGNOSTIC ONLY. No training, optimizer, virtual update or new model.
The interrupted Full-History Teacher implementation remains unlaunched and is
not used by this task. This supersedes that turn's immediate execution plan.

## Scope and paired controls

MOSI seed66, original LargeFlat/CHM, protected miss0.3 BEST checkpoint (epoch50).
All seven current patterns use the SAME checkpoint. Keep original data, s0,
GCNet random-mask-v1 host/guest batch32 sampling and causal joint block writes.
History is a single predeclared rate0.3 trajectory, not eight different models
or Persistent conditions. Test history masks are the exact accepted saved bank.

1. TRAIN: one target per eligible conversation, deterministic ID-only selection,
   with at least one past utterance. Sample historical masks in original padded
   batch rectangles BEFORE isolating target prefixes. Force only the current
   target pattern A/T/V/AT/AV/TV/ATV; available raw features may be revealed at
   that target as an explicit diagnostic intervention, not a sampler change.
   Original Local and Query adapt normally to that current pattern. Past
   features/masks/speakers, pre-write state and past predictions stay identical.
   Inspect target MSE gradients only w.r.t. original K/V writer projectors.
   Detach the current Key output to exclude its direct Gap-addressing derivative;
   past writes keep their true BPTT graph. Current Value write cannot influence
   current prediction because reading precedes writing. No parameter .grad,
   optimizer or weight update is produced. Report loss-gradient and prediction-
   Jacobian cosines: MSE scalar residual sign alone can reverse loss gradients.
   Also report geometry of aggregate mean-task gradients, not only the mean of
   per-target cosines. A negative cosine is local gradient competition, not proof
   of long-term negative transfer or a defective Memory.

2. TEST: evaluate all valid utterances in seven current patterns under factual
   history, empty history, and content-replaced history. Cache the original
   already-decayed target state BEFORE its own write. Reuse original current
   encoder/Local/Query/addressing/Flat/head; only substitute the cached state.
   Test each target as an independent current-only intervention: this is NOT
   Persistent Missing, and later factual states never include intervened writes.
   Donors are deterministic, label-independent other TRAIN conversations with
   enough actual utterances for the complete target history. Preserve recipient
   history availability, speaker IDs and length; replace only raw history content.
   Donor identity/split/length/hash are recorded. If no compatible donor exists,
   mark unavailable; do not substitute zero or change the trajectory.
   Report all versus history-present cohorts, MSE and polarity ACC/W-F1,
   corrections/harms and correct-history advantage over both controls.

## Validation and execution

Tests-first for masking, strict past, writer-only graph, fixed-state replay,
donor validity, metric filtering and null/zero-gradient handling. Check factual
predictions against saved checkpoint output, state/Local pairing, model digest
and original RNG preservation. Empty-history rows must agree across conditions.
No test-label gradient calculations or updates. Stored labels only score outputs.

Independent immutable source snapshot and output on biggpu; never physicalGPU4,
never interrupt peer tasks. Full prediction/gradient-pair records stay remote;
small reports, configuration and checksums enter Git. The old checkpoint is
Test-oracle selected, so no independent generalization claim follows.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_current_pattern_history_audit_20261022.run \
 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
 --source-commit SOURCE_SHA --output OUTPUT_DIRECTORY
```

Add `--smoke` for only two gradient anchors and one test batch. Smoke is
correctness evidence only; it is excluded from final complete-cohort summaries.
