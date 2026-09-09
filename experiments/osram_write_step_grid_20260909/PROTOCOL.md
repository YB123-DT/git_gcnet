# Locked under-relaxation grid: frozen first, conditional retraining second

Fixed write strengths: **1.0 / 0.95 / 0.90 / 0.80**. No intermediate values,
adaptive gates, new modules, loss changes, upstream changes or grid expansion.
Reference implements 1.0 by returning the original block-write result exactly;
others use `M_read + eta * (original_post - M_read)` at every write, including
complete and no-history steps. The existing solve, alpha and beta are unchanged.

## Stage 1: frozen checkpoint evaluation

Run IEMOCAPFour first, then CMUMOSI; seeds 66–70 for each. Use the inherited
five missing rates **0 / .1 / .3 / .5 / .7**, not an eight-rate reporting mean.
Use one existing `best.pt` per seed, selected historically by eight-rate-mean
Test-oracle, with no new checkpoint selection. Never substitute the old
independent per-rate Test-oracle score tables for these checkpoint results.

IEMOCAP4 eta1/.9 reuse
`../osram_write_intervention_20260909/full5/iemocap4_seed*/metadata.json`.
Only evaluate eta.95/.8 anew, then verify equal checkpoint/config/mask hashes
against inherited cells. MOSI evaluates all four strengths. Thus the full grid
contains 200 cells, of which 50 are inherited and 150 newly evaluated.

Record W-F1, macro-F1, accuracy (MOSI nonzero-label binary convention unchanged),
and inherited MOSI MAE/correlation without optimizing them. `E_old` is the
historical missing-slot probe **err_decay**, the actual pre-write read state;
NO_HISTORY is excluded, not zero-imputed. `E_new` is current observed-slot
**err_after**, after the applied write. Preserve err_before, err_original_after,
fit_gain and signed historical decay/write damage as supporting diagnostics.
Summarize within each run and then equally across seeds. Missing patterns/heads
and rates are not independent experimental repeats.

Analyze the four-point curve and seed consistency, not a densely tuned best
hyperparameter. Increasing performance down to .8 is not evidence that .9 is
an optimum; different dataset maxima do not establish a universal step size.

## Predeclared Stage 2 decision

Proceed to fixed-eta=.9 retraining only if BOTH datasets show:

1. five-rate mean at .9 above 1.0, with at least 3/5 paired seeds positive;
2. five-rate mean at .8 below .9 (an interior region near .9 rather than an
   improving lower boundary).

This is an operational screening rule, not a significance test or proof of an
optimum. If it fails, report the shape and do not automatically train another
eta or introduce a new module. If it passes, train causal OSRAM with the sole
change `Delta *= .9`, preserving original initial seeds, all backbone/head/loss
components, cyclic schedule, masks, optimizer/LR/batch/epochs and eight-rate-mean
checkpoint-selection protocol. Compare matching checkpoint-based controls, not
the independent per-rate maxima. Inference-only gains cannot establish training
benefits because the original weights were trained with eta1.

## Artifacts

`iemocap4/seed*/metadata.json`, `mosi/seed*/metadata.json`, generated CSV and
Markdown summaries, source inheritance paths and raw SHA256 manifests.
Large per-head JSONL gzip files remain locally and on biggpu; Git holds compact
results/provenance/manifests. No raw data are deleted.

INTERNAL TEST-ORACLE CHECKPOINT DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.
