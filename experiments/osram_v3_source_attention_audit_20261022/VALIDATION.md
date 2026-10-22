# Completed frozen attention audit

Source220b878, biggpu physicalGPU7, UUIDGPU-c38d9fe1-0b58-158f-a289-32d21e96df2e.
Actual four replay processes returned0; controllercomplete; all96 formal BEST
replays (MOSI8,MOSEI8,IEMOCAP4/6 each40) completed. No optimizer/backward/training,
no new checkpoints and no modification/deletion of existing weights or masks.

## Tests

Tests-first:2 capture tests and2 replay-interface tests failed for missing
implementation,9 statistics tests failed before implementation, then passed.
Leader's final related suite:30passed in5.99s,including original V3 regression.
Extra capture regression checks no-history/ATV empty rows and exact one-source
weight1 without treating it as learned confidence. Compile/diff checks passed.

## Actual complete-cohort replay

| Dataset | Checkpoints | Max abs prediction error | Classes/IDs/labels/speakers/masks |
|---|---:|---:|---|
| MOSI |8|4.76837158203125e-7|exact|
| MOSEI |8|2.384185791015625e-7|exact|
| IEMOCAP4 |40|0|exact|
| IEMOCAP6 |40|0|exact|

The earlier short update calling all MOSI outputs bitwise identical was too
strong: full verification finds the tiny errors above. Predetermined comparison
uses rtol=atol=1e-6 and exact classes; it was NOT relaxed after seeing outcomes.
IEMOCAP is bitwise identical; scalar numerical closeness is not bitwise parity.

All active head weight sums differ from1 by at most1.1920928955078125e-7.
Inactive-role and unseen-source attention weights are exactly0. Parameter-state
digests before and after every replay agree; selected checkpoint/prediction SHA,
original saved test bank and full test-order metadata are verified before export.
Every test batch executes exactly one original CHM scan. Query-swap computes only
an extra attention call with original keys/values and discards its output.

Original per-dataset Python/Torch/NumPy/sklearn versions were enforced, not changed:
MOSI/MOSEI s0 Torch2.2.2/NumPy1.24.3; IEMOCAP cmerc-repro1.13.1/1.26.4. Exact
paths,feature checks,environment and run durations appear in COMPLETED and remote
STATUS files. Final queries use no gold labels; labels only define offline groups.

## Interpretation

MOSEI focus source weights are almost uniform (entropy>0.9999998,including head
means); swapping final Local Query changes average source weights very little.
MOSI has weaker but nonzero role bias; IEMOCAP4/6 show more heterogeneity. IEMOCAP4
fold2 Base favorsV(~0.511),fold4 Base favorsT(~0.521),so pooled means cannot imply
a fixed universal source preference. Group means and per-head means cannot
determine which source causally helps classification after Layer2 source mixing.

All aggregates here pool test slots over different rate-selected weights; they
are not independent samples or single-checkpoint deployment behavior. Onlyseed66
is audited. No performance improvement or mechanism success follows from attention
selectivity, and weak selectivity alone does not locate a Query/Key/training cause.
