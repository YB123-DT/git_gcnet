# Frozen Memory Residual Learning

INTERNAL DIAGNOSTIC ONLY

## Fixed question and scope

Does a small task residual learner extract useful information from real frozen
OSRAM reads beyond the original prediction and equal-capacity local/donor controls?
This is a learning experiment, not another sensitivity-only diagnostic.

- MOSI, original large cfg84 no-JEPA Flat backbone seed66, eight existing per-rate
  Test-oracle checkpoints. No retraining, feature extraction or Memory updates.
- Reuse the completed `osram_frozen_memory_audit_20261009` train/test feature caches,
  preprocessing, and preceding-three-utterance-mean probes. Source code before
  this experiment: `05fb98ea6dc0814c12acd13af0cc4613ed1004aa`.
- Residual initialization seeds66/67/68 are NOT three independently trained backbones.
- Seven residual conditions per rate/seed: 168 fits total; no parameter sweep.

## Models

Every prediction is `y0 + Linear32to1(GELU(LinearInputTo32(features)))`.
The final linear weight and bias are zero-initialized. `y0` is the stored original
Flat prediction; the original task path is never replaced or updated.

A input: `[y0, standardized Local256, availability3, z256]` (516 dimensions).
All three heads have 16,577 parameters:

1. Local control: z is exactly zero.
2. Donor Memory: projected train-only donor reads, matching current availability
   and history existence, from a different conversation.
3. Real Memory: projected own Base and active Gap forward reads.

Use the existing training-only active-slot normalizer and fixed shared
512-to64 random projection, giving four fixed 64-d slots. Inactive slots and
first-turn Memory are zero. Donor assignment is fixed across residual seeds,
label-independent, and recorded. All comparisons use identical eligible rows.

B input: `[y0, standardized Local256, availability3, has_history, z1]`
(262 dimensions), 8,449 parameters per head:

1. Local-history: existing history Probe A on the current Local.
2. Donor-history: existing history Probe B on current Local and donor reads.
3. Real-history: the SAME Probe B on current Local and own reads.
4. Gold-history: mean of up to three preceding true sentiment labels.

Use fixed historical probe initialization seed66, with frozen weights and
original preprocessing. Do not substitute Probe C for the donor condition:
that would confound the input replacement with a different decoder. Use one
shared scalar normalization fitted on real-history TRAIN outputs, then re-zero
no-history rows. Gold history is only an offline privileged-information reference,
not deployable and not a guaranteed numerical upper bound. True current labels
are targets, never input features. Historical labels never cross conversations.

## Training and evaluation

- Adam, lr=0.001, weight_decay=0.00001, batch128, 100 epochs, taken from the
  completed audit's probe training configuration.
- Primary checkpoint: fixed LAST epoch100, not best test epoch. Save full last
  state, train/test curves and predictions. No validation or test-label gradients.
- Original checkpoints and historical probes were previously test-selected;
  consequently results remain internal diagnostics, not independent test evidence.
- The old history probes predict on their own training rows when producing
  residual training features. No cross-fitting is added in this experiment;
  acknowledge train/test decoder generalization differences as a limitation.
- Original task MSE is retained; primary comparisons are nonzero-label W-F1 and
  ACC with `prediction > 0`, per rate then equal-rate mean, high mean .5/.6/.7.
- Report all residual seeds, mean/std, and corrections/harms vs original. Do not
  pool utterances across rates to calculate the eight-rate summary.
- A must beat Original, Local and Donor; B must beat Original, Local-history and
  Donor-history. A gain over Original alone does not demonstrate extra Memory value.

## Execution and verification

Run on biggpu using cached features, CPU with two threads per process, split into
two independent rate groups. This tiny-head experiment needs no GPU or OSRAM
forward. Do not touch unrelated GPU jobs; physical GPU4 remains prohibited.

Before deployment check zero-init parity, equal capacity within each family,
donor/current-Local preservation, first-turn and inactive-slot masks, finite
gradients and train-only updates. Verify existing probe predictions reproduce
saved values and hash all input artifacts. No raw features or weights enter Git.

Artifacts: sealed code snapshot, launch record, per-fit last checkpoint and
prediction arrays (remote only), complete small summary tables and RESULT.md (Git).
