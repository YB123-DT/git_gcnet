# Frozen MELD sensitivity, preregistered before inference

INTERNAL DIAGNOSTIC ONLY. No training, no optimizer, no parameter selection.

Accepted ECERC Flat1600 with Local Skip, seed66; source b4a3138; exact
ACC-selected masks from osram_meld_testbest_ua_20261014/exact/source_0. Feature
SHA256 a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80.
Do not use final-evaluation masks in the checkpoint directory. Three predefined
rates 0.0/0.4/0.7 cover full, mixed and single-observed regimes; these are NOT an
8-rate mean and must not be compared as such with the accepted baseline mean.

Conditions (no sweep or winner selection):

- Original learned per-head alpha and per-head/modality beta.
- Fixed beta 0.05/0.50/0.95, retaining learned alpha and embedded-beta mode.
- Fixed alpha 0.950/0.980/0.995, retaining learned beta.

Constants deliberately bracket a broad beta range and short/near-original/long
retention range. They are diagnostic interventions, not proposed optima. Each
trajectory starts at zero independently. Current Local, Query, K/V and masks
must match the original exactly. Read-before-write and single joint Block Write
are unchanged. Original logits/classes must match saved exact predictions.

Compare pre/post-write states, Base/active Gap read differences and same-row task
CE/ACC/UA/W-F1. Also apply beta overrides to the *same original pre-write state*,
K/V and mask to separate immediate update sensitivity from propagated trajectory
sensitivity. Spectra exclude padding and inactive columns; compare actual
weighted K-transpose-K eigenvalues with ridge, and record ridge/beta diagonals.

For alpha report strict-past-history-count bins first/1-3/4-7/8-15/16+; this is
not the lag of an individual source observation and cannot establish survival of
one association through intervening writes. Separately fork from each *same
original post-write state*, change alpha only from that point onward, and replay
unchanged future joint writes to distances 1/2/4/8. Compare original target reads
and task outputs with these lagged forks; these source-target cohorts can repeat
a target and are not full-testset scores or a single-association attribution.
First verify this helper with original learned alpha reproduces normal states.
Canonical saved masks must retain at least one modality per valid utterance,
so availability.any(-1) identifies the active rows in these write snapshots.
For heads, keep the original scan
and mask one 64-dimensional forward read slice, separately in Base and all
active Gaps (16 readout-only interventions). Preserve Local, backward zero slots
and classifier. Report paired corrections/harms and class recall, not only norms.

All model weights/RNG must remain unchanged after restoring temporary logits.
Artifacts have generated batch-major conversation/utterance IDs; original NPZs
have no IDs, so alignment evidence is loader order plus labels/classes/mask
counts and hashes, not a nonexistent archived ID equality check. No new mask is
generated. Preserve all protected weights. No best-parameter claim, no new block
or multi-seed training is authorized by this diagnostic.
