# All-Gap versus T-only Gate: current no-Text predictions

INTERNAL DIAGNOSTIC ONLY. No training or inference was added.

Reuse each model's eight per-rate Test-oracle BEST prediction files for seeds
66/67/68. Compare current A/V/AV rows under random missing, NOT whole-conversation
persistent missing. Nonzero labels only; positive iff prediction > 0. Report
equal means across nonempty seed/rate groups, not pooled W-F1. Rate0 has no
no-Text samples. NPZ files have no explicit sample IDs; existing saved row order,
labels, availability and evaluation mask hashes match exactly across versions.
Source file hashes and per-seed/per-rate results are in MISSING_TEXT_SUMMARY.json.

| Current condition | All-Gap Gate W-F1 % | T-only Gate W-F1 % | Delta pp |
|---|---:|---:|---:|
| A | 67.382 | 66.841 | -0.541 |
| V | 65.454 | 65.702 | +0.248 |
| AV | 65.940 | 65.295 | -0.645 |
| All current no-Text rows | 66.999 | 66.431 | -0.567 |
| All current no-Text, high rates .5/.6/.7 | 65.415 | 64.265 | -1.149 |

The combined no-Text statistic is computed within each seed/rate over A/V/AV
samples together, then averaged; it is not the mean of the three pattern scores.

| Seed | All-Gap no-Text | T-only no-Text | Delta pp |
|---|---:|---:|---:|
| 66 | 68.793 | 67.369 | -1.424 |
| 67 | 67.152 | 67.305 | +0.153 |
| 68 | 65.051 | 64.620 | -0.431 |

Across 5067 nonneutral seed/rate sample occurrences, T-only versus all-Gap has
410 wrong-to-right changes and 443 right-to-wrong changes. These are repeated
sample occurrences across seeds/rates, not 5067 unique utterances. Macro W-F1
differences need not have the same sign as pooled correction-minus-harm counts.

Both versions are independently jointly trained, so this comparison does not
isolate the Gate-T branch's direct causal contribution. It shows no overall
no-Text advantage from restricting gating to Gap-T.

Reproduce on biggpu:

```bash
python compare_missing_text.py \
  --all-gap-root /data2/yb/remote_experiments/osram_nested_gap_residual_gate_20261010/runs \
  --t-only-root /data1/yb/remote_experiments/osram_nested_gap_t_residual_gate_20261010/runs \
  --output /data1/yb/remote_experiments/osram_nested_gap_t_residual_gate_20261010/missing_text_comparison
```
