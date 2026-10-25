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

The initial replay_v1 stopped before inference: checkpoint config retains
`train_missing_rates` as a tuple, whereas SUMMARY.json stores the same values
as a list. Canonical JSON identity comparison fixes this serialization-only
false mismatch; epoch and all actual config values remain strictly checked.
A regression test accepts tuple/list equivalence but rejects changed epochs
or write_step. Failed output/source_v1 are preserved; retry uses source_v2 and
replay_v2, with the same protected checkpoints and three coefficients.

## One slot at a time (2026-10-25 follow-up)

User explicitly requests separate, not joint, strengths. Add `--per-slot` to
the same entry. Independently scale Local/Base/Gap-A/Gap-T/Gap-V by0/.5/1 while
the other four corrections stay exactly1. There are11 unique readouts per
scan:one common full1 plus ten single-slot interventions. LocalSkip remains
the original Local even in the Local intervention. No combination search,
new training, other coefficients or new seeds are included.

Source_v3 and slot_replay_v1 are separate from all original sources and the
completed joint replay. The default entry remains joint; old artifacts are
not modified. New unit cases cover all15 slot/strength mappings, unchanged
other inputs, no mutation, backward zeros and inactive gap zeros.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_mosei_n10_scale_20261024.run \
  --per-slot --source-commit SOURCE_SHA --output ROOT/slot_replay_v1
```

## Fine single-slot strengths (2026-10-25 follow-up)

The next explicit request refines correction strength to0.1/0.2, not missing
rate. Add `--fine-slots --per-slot`:one common full-strength reference plus
Local/Base/Gap-A/Gap-T/Gap-V at0.1 and0.2, all other corrections fixed at1.
The old default and `--per-slot` grids are unchanged. No combined settings,
retraining or new checkpoint selection. Use a separate immutable source_v4
and output fine_slot_replay_v1; old results remain untouched.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_mosei_n10_scale_20261024.run \
  --per-slot --fine-slots --source-commit SOURCE_SHA --output ROOT/fine_slot_replay_v1
```
