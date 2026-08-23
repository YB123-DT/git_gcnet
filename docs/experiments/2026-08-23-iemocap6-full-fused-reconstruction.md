# IEMOCAPSix GCNet Full-Fused Reconstruction

## Question

Continue from the completed GCNet missing-only reconstruction baseline and test
one controlled change: when an utterance contains any missing modality, use the
existing `linear_rec` head to reconstruct the complete Audio/Text/Visual target
instead of selecting only the missing modality dimensions.

No DFGCN, JEPA predictor, EMA target, extra encoder, or reconstructed-feature
classification path is used. The GCNet architecture and parameter count remain
unchanged.

## Protocol

- Dataset: IEMOCAPSix, official fold 5.
- Features: `wav2vec-large-c-UTT` (512D), `deberta-large-4-UTT` (1024D),
  `manet_UTT` (1024D).
- Missing rates: 0.0 through 0.7.
- Seeds: 66 through 75.
- Training: 100 epochs, batch size 32, hidden size 200, learning rate 0.001.
- Environment: Python 3.8.20, PyTorch 1.8.0, CUDA 10.2, PyG 2.0.1.
- Total model parameters: 37,249,214 in both conditions.
- Existing baseline was reused read-only; only 80 Full-Fused jobs were trained.
- Source commit: `47e5f2c` plus the previously committed model implementation.

Artifacts:

- Existing GCNet baseline: `/data2/yb/experiments/gcnet_official_4dataset_10seed_20260820/IEMOCAPSix`
- Full-Fused results: `/data2/yb/experiments/gcnet_iemocap6_full_fused_10seed_20260823`
- Smoke results: `/data2/yb/experiments/gcnet_iemocap6_full_fused_smoke_20260823`

Completion evidence: 80/80 jobs completed, zero worker errors, and 80/80 paired
manifest audits passed.

## Weighted-F1 Results

Values are ten-seed mean ± sample standard deviation. Delta is Full-Fused minus
the existing GCNet baseline, in percentage points.

| Missing | GCNet baseline | Full-Fused | Delta | Wins/10 | Wilcoxon p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 63.04 ± 1.63 | 62.78 ± 1.22 | -0.26 | 4 | 0.625 | 1.000 |
| 0.1 | 62.46 ± 1.05 | 63.57 ± 1.35 | +1.11 | 7 | 0.084 | 0.588 |
| 0.2 | 63.76 ± 1.88 | 62.41 ± 1.72 | -1.35 | 2 | 0.232 | 1.000 |
| 0.3 | 63.03 ± 1.37 | 63.11 ± 1.04 | +0.08 | 5 | 0.770 | 1.000 |
| 0.4 | 61.31 ± 1.87 | 62.68 ± 1.10 | +1.37 | 7 | 0.049 | 0.391 |
| 0.5 | 62.60 ± 2.02 | 62.16 ± 1.98 | -0.44 | 6 | 0.625 | 1.000 |
| 0.6 | 60.80 ± 1.38 | 61.26 ± 1.99 | +0.46 | 7 | 0.695 | 1.000 |
| 0.7 | 61.06 ± 1.54 | 61.96 ± 2.11 | +0.89 | 6 | 0.322 | 1.000 |

Across all 80 paired runs, the mean delta is +0.23 W-F1 points, 44/80 pairs
improve, and the paired Wilcoxon p-value is 0.340. None of the eight per-rate
comparisons remains significant after Holm correction.

## Collapse Audit

The pre-registered collapse rule from the earlier GCNet stability study is:

```text
median(group) - score >= 0.10
and
0.67448975 * (score - median(group)) / MAD(group) <= -3.5
```

Applied separately to each `(method, missing-rate)` ten-seed group, this rule
flags zero baseline runs and zero Full-Fused runs. Therefore, the primary table
is unchanged after strict collapse removal.

As a sensitivity analysis, the absolute 0.10 gap requirement was removed and
only the robust MAD rule was retained. This looser rule flags one run:
GCNet baseline at missing 0.3, seed 68 (W-F1 60.41, robust z = -3.63). No
Full-Fused run is flagged. Strictly removing the corresponding pair changes the
missing-0.3 delta from +0.08 to -0.13 points. Across all rates, the filtered
79-pair mean delta is +0.21 points with paired Wilcoxon p = 0.392, versus +0.23
points and p = 0.340 before filtering.

Thus, the positive effects at missing 0.1 and 0.4 and the negative effect at
missing 0.2 are not explained by rare collapsed seeds. They reflect broader
within-rate behavior and ordinary optimization variance.

## Interpretation

Full-Fused reconstruction produces promising local gains at missing rates 0.1,
0.4, and 0.7, with the largest mean gain at 0.4 (+1.37 points). However, it
degrades 0.2 by 1.35 points and does not yield a statistically reliable overall
improvement. The evidence therefore does not support using full-state
reconstruction as the final method without a mechanism that suppresses
unpredictable modality-private information.

At missing rate 0.0, both primary reconstruction losses are exactly zero and
the paired initialization and mask hashes match, yet the ten-seed means differ
by -0.26 points. This is consistent with the previously observed GPU/GNN
nondeterministic optimization variance: paired seeds and hashes control the
setup but do not force bit-identical training trajectories. Consequently,
sub-one-point changes should be treated cautiously.

The most defensible next experiment is not another broad sweep. It is a narrow
test at missing rates 0.2 and 0.4 that asks why complete-state supervision helps
one rate and hurts the other, for example by separately recording visible- and
missing-modality reconstruction gradients. The current result is best treated
as a diagnostic: complete reconstruction can help under some missing regimes,
but its supervision is not consistently aligned with ERC.
