# Frozen V3 source reading attention audit

User asks whether role-specific Local Queries select different historical source
tokens. This is no-training, no-weight-change inference on existing standardV3
seed66 only: MOSI/MOSEI8rates and IEMOCAP4/6 all5folds x8rates,96BEST replays.
Do not substitute the ongoing MOSI direct variant or capacity/HPO candidates.

Core files and existing results are untouched. `capture.py` requests per-head
weights from the existing output CrossAttention and returns their same default
mean to V3. Numerical outputs/RNG are tested identical. Actual predictions must
reproduce stored BEST predictions on exactly matching IDs/labels/speakers/masks
with tolerance1e-6 and exactly identical predicted classes. One original CHM scan
per batch; no optimizer, loss training, mask regeneration or checkpoint creation.

Strict original environments: scalars use s0 (Torch2.2.2/NumPy1.24.3); IEMOCAP
uses cmerc-repro (Torch1.13.1/NumPy1.26.4). Data roots/splits/configs come from
the actual protected source run; checkpoint/prediction/mask hashes are recorded.
All scalar feature hashes and IEMOCAP label hashes are verified. IEMOCAP feature
roots are reused and full-cohort prediction parity is required, rather than
claiming old missing feature-file hashes exist when they do not.

Reports contain role/source means,std,head-specific means/std,normalized entropy,
source counts0/1/2/3,current availability,Text history seen/coverage/age and
correct/error strata. Base and Gap-T focus uses SAME samples with currentText
missing and all3sources previously seen. Every head's valid source weights sum1.
ATV/no-history/inactive rows have no V3 attention, not uniform fabricated weights.

A label-free matched Query-swap sensitivity check keeps original keys/values and
uses another conversation's Query from same role/availability/source-set group
within the batch. Extra attention-only output is discarded; actual predictions
remain original. This tests dependence on Local Query, not semantic correctness.
Unmatched/one-source cases are excluded from this sensitivity statistic.

`stats.py` validates arrays strictly; it does not renormalize masks or hide NaNs.
All per-fold/rate NPZ and detailedJSON stay remote. Compact CSV/RESULT go to Git.
Pooled rate/fold counts are TEST SLOTS, not independent samples or seeds. Attention
is over source-mixed Layer2 tokens and cannot establish original Value-source
causal contribution or usefulness. Gold labels are offline strata only.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
 -m experiments.osram_v3_source_attention_audit_20261022.scheduler \
 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
 --source-commit SOURCE_SHA \
 --output /data2/yb/remote_experiments/osram_v3_source_attention_audit_20261022/execution_v1
```

Persistent controller launches at most2 frozen replays, within measured free
memory, with each dataset's original Python. Existing training is never stopped,
reconfigured or moved. GPU4 is forbidden. No automatic seed or method expansion.

## Completed evidence and compact exports

Immutable replay source:`220b878c11d2d93697bfe4193c5c411b1ce09312`.
Actual output:`/data2/yb/remote_experiments/osram_v3_source_attention_audit_20261022/execution_v1`.
All4 jobs/96 checkpoint replays completed successfully; no new training steps.

- `RESULT.md`: pooled focus table and all8 rates, with limitations.
- `FOCUS.csv`:64 role/rate rows, IEMOCAP test folds descriptively pooled.
- `HEADS.csv`:256 individual-head focus rows, same populations as FOCUS.
- `FOLDS.csv`:24 role/fold rows pooled over rates; exposes IEMOCAP fold differences.
- `ROLES.csv`:768 role/rate/source-count/correctness rows including population std.
- `STRATA.csv`:2304 additional role/rate rows for availability,historyText seen,
  coverage and age. Derived from full remoteREPORT `per_rate[rate].rows`; column
  meanings match ROLES. Null values are N/A, never missing-score zeros.
- `SUMMARY.json`:compact pooled diagnostics, query sensitivity and validation.
- `COMPLETED.json`:commands,source hashes,environment,completion and raw-report SHA.

Full9MB `REPORT.json` and96 per-fold/rateJSON/NPZ remain remote (not committed).
CSV exports retain reported numeric values; newline normalization is immaterial
and bytewise CSV equivalence is not claimed. Fold weights/entropy in FOLDS are
weighted by focus-row support across the8 saved per-rate summaries. Raw model,
mask and prediction artifacts are untouched. Query-swap tests ONLY the final
Local Query path: original Local-conditioned PairMLP/keys/values are retained.
