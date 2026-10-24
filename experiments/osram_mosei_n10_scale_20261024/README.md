# Frozen MOSEI N10 input-correction strength

INTERNAL DIAGNOSTIC ONLY. User-approved finite replay, not new training.

Fixed completed N10 seed66 (Flat128, dim64/depth3/groups2/MLP128), its eight
original per-rate Test-best checkpoints and exact accepted GCNet test masks.
Evaluate only gamma={0,.5,1}, shared across all five input correction slots:
`X_gamma = X_raw + gamma*(X_nested-X_raw)`.

LocalSkip uses the original Local. CHM, queries, write trajectory, original
Local/Base/Gap evidence, classifier parameters and masks remain unchanged.
Only valid forward input corrections are affected; the existing replay helper
retains inactive-gap/padding masks. Gamma0 is the Nested-trained checkpoint
with its input corrections bypassed, not separately-trained Flat128.

One original forward per batch captures raw and corrected inputs. Three Flat
readouts reuse them without a second Memory scan. Gamma1 must reproduce both
the actual forward and archived original predictions (float tolerance1e-6 and
exact polarity). Model states, source files and checkpoint hashes are checked.
No optimizer is constructed or updated. No active source snapshot is changed.

Outputs: SUMMARY.json, RESULT.md, PER_RATE.csv and separate exact-mask prediction
arrays for every gamma/rate. Metrics:ACC/W-F1/MSE, equal eight-rate mean and
high-missing mean. Correct/wrong counts versus gamma1 and original Flat66
include all samples and A-only positive/negative and near-Flat-boundary cohorts.
Gold-zero utterances are excluded only for polarity metrics/counts; MSE retains
the original full target set. Test-based coefficient ranking is exploratory.

Runtime: biggpu physicalGPU1 UUID-bound, cmerc-repro; requires>=6GiB free before
starting, never uses faultyGPU4 or alters others' tasks. Batch32 unchanged.
Copy the immutable N10 source snapshot and overlay only this runner/tests,
then regenerate its manifest. Persist under tmux, e.g.:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_mosei_n10_scale_20261024.run \
  --source-commit SOURCE_SHA --output ROOT/replay_v1
```

Five focused tests cover exact endpoint identity, half-strength arithmetic,
read-only inputs, finite authorized grid/shapes, nonzero-gold polarity counts
and complete equal-rate aggregation. The real replay itself performs the
checkpoint/mask/forward parity checks; no separate training smoke is added.
