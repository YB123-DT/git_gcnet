# Utility-Supervised Text Retrieval: Internal Diagnostic Status

**This experiment is incomplete and the U1-U3 numbers below are not valid final results.**

## Why stopped

The frozen PAM-E candidate utility cache was generated successfully:

- candidate bank coverage: 93.25%
- mean candidate count: 8.64
- U1/U2/U3 retrievers trained for all 5 seeds x 7 nonzero rates.

However, a validation check showed that the manual per-step candidate scorer used
to generate the cache does **not** reproduce the original frozen PAM-E Reader
output for candidate 0:

- cache `scores[:, 0]` vs PAM-E `predictions_miss_*` at the same T-missing rows:
  mean absolute difference ~0.61 on seed 66/rate 0.5.

Consequently:

- Oracle-Best is not even guaranteed to fall back to the original PAM-E output.
- U1/U2/U3 learned retrieval scores are therefore not yet comparable to the
  internal baselines.
- U4 (utility-joint) and U5 (utility-global) were not run because the shared
  utility cache is not validated.

## Current partial observations (do not cite)

- Retrieval metrics from U1/U2/U3 cache:
  - U1 NDCG@1 0.272, top1 oracle hit 0.308
  - U2 NDCG@1 0.213, top1 oracle hit 0.273
  - U3 NDCG@1 0.278, top1 oracle hit 0.321
- These metrics use the unvalidated candidate score cache above.

## Required fix before a real run

1. Make candidate 0 exactly equal the frozen PAM-E Reader output at the same
   utterance (use a real frozen Reader forward for candidate 0, not the manual
   per-step readout).
2. Recompute/revalidate candidate 1..n scores against a full Reader forward
   substituted into the current T-missing slot.
3. Rebuild the utility cache.
4. Then train U1/U2/U3, implement U4/U5, and run the full matrix.

No U6 has been designed.
