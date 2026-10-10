# Frozen Nested residual on/off for current Text-missing samples

INTERNAL DIAGNOSTIC ONLY. Evaluation-only intervention, no retraining.
User confirms dert means Nested Local/Base/Gap residual correction, NOT OSRAM
delta write. Fix original Nested seed66 per-rate BEST from completed original100
monitored run; not the ongoing lowLR/warmup/random-initialization experiment.
All eight rates 0.0–0.7, all test utterances; no cherry-picking. Primary cohort:
current T unavailable, split A/V/AV. Rate0.0 has no such samples (N/A).
Macro-average only nonempty rates; high-missing means .5/.6/.7.

Compute encoder/causal OSRAM exactly once per batch. Capture meaningful_block
input (L,B,G,a,umask) and output (L+deltaL,B+deltaB,G+deltaG). Full prediction
comes from normal model forward. Replay only shared Flat/head twice for parity
and off: off uses uncorrected captured L/B/G, with SAME original Local Skip,
availability, effective Gap masking, norm/head/weights. Base and Gap remain.
No query/write modification, new schedule, memory state or local-only comparison.

Report no-Text group and A/V/AV, W-F1(on/off), on-minus-off pp, corrections/harms
(OFF→ON), flips, mean absolute prediction shift, N/nonneutralN. Save sample IDs,
labels/masks and predictions; nonzero labels and >0 polarity threshold unchanged.
Optional history-only subgroup excludes first utterances, not main cohort.

Verify frozen checkpoint/state unchanged, replayON matches normal output≤1e-5,
all full-test ON predictions/labels match original prediction archive≤1e-5,
full-test W-F1 matches each original selected score, inactiveGap/padding safely zero.
One healthyGPU7 inference job on biggpu; no new training or Flat rerun.
Isolated tracked-code snapshot; strict state loading and cached prediction parity
guard against code/protocol mismatch. Record commit/checkpoint/data hashes.
