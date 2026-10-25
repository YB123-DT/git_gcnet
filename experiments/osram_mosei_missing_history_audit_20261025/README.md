# Missing-target history exposure and Base/Gap readout audit

INTERNAL DIAGNOSTIC ONLY. Frozen MOSEI seed66 Flat128 and completed N10,
cmerc-repro, all eight original per-rate Test-best checkpoints and exact GCNet
masks. No new training, loss, mask sampling, Memory intervention or coefficient
selection. Existing model, sampler and experiment outputs remain unchanged.

For each currently missing target A/T/V, distinguish strictly-past seen,
never-seen (including first turns), never-seen with other history, and first
turn with no history at all. Availability is updated only after recording the
current cohort. Dialogues never share state. Missing rate0 has no missing
targets and is N/A for those cohorts, not zero performance.

Each batch performs one original causal scan, followed by13 original
Flat/head replays retaining the same task-side Local and original LocalSkip:
Full, Local, Local+Base, Local+all Gap, plus for each target removal of Gap-m,
removal of Base and Gap-m, and Local+Gap-m only. Other Gap slots remain active
in the target-removal conditions. These are frozen-input interventions, not
separately trained Local/Memory predictors. They remove entire original task
inputs, unlike the preceding Nested correction-only strength audit.

For N10, retained task-side Local already contains its learned Nested Local
correction. Thus the `Local` row means zero Base/Gap classification slots with
that Local held fixed, NOT a history-free N10 predictor. Likewise, removing
Gap-m at Flat input does not erase its earlier influence on Nested-corrected
Local/Base. Flat128 has no such extra mixing; its `Local` row uses original
Local. This distinction is essential when interpreting pathway sensitivity.
Recorded N10 Base/Gap norms are also the post-Nested task-input norms, not
raw CHM reads before the correction; Flat128 norms are the original reads.

Save all predictions, labels/IDs/indices/availability and forward read norms.
Report per-rate ACC/W-F1/MSE, label supports, positive/negative Recall,
corrections/harms relative to Full and Local, and Base/each Gap norm. Compare
models on identical cohorts/masks but retain their own original fixed
per-rate weights. Between-cohort differences alone are not causal effects of
history exposure; class balance/current modalities/history length can differ.
No source-specific semantic claim is justified by zeroing whole Base/Gap.

Checks:original full prediction/metrics, strictly-past grouping, no padding,
exact masks/metadata, causal backward/inactive Gap zeros, first-turn zero
history reads/equal readouts, one `_scan` per batch, frozen state/checkpoint
hashes and no CPU/CUDA RNG changes during readout replays. Empty cohorts are
N/A; polarity metrics exclude neutral gold, MSE does not.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_mosei_missing_history_audit_20261025.evaluate \
  --source-commit SOURCE_SHA --output ROOT/replay_v1
```

Server:biggpu physicalGPU1 (`GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`),
with>=6GiB free admission. Never use faultyGPU4 or change other jobs. Copy a
verified numerical source snapshot and add only the standalone evaluator and
tests; regenerate its immutable source manifest before execution.
