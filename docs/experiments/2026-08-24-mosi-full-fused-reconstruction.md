# CMU-MOSI GCNet Full-Fused Reconstruction

## Question

Does reconstructing the complete multimodal utterance state provide a more
useful training signal than reconstructing only missing modality dimensions for
GCNet under incomplete multimodal input?

Only the reconstruction target changes. The model remains GCNet, reconstructed
features do not enter the regressor, and no JEPA predictor, EMA target, extra
encoder, or DFGCN component is introduced.

## Protocol and evidence

- Dataset: CMUMOSI, official fold 1.
- Missing rates: 0.0 through 0.7.
- Seeds: 66 through 75.
- Epochs: 100.
- Features: `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, `manet_UTT`.
- Environment: Python 3.8.20, PyTorch 1.8.0, CUDA 10.2, PyG 2.0.1.
- Parameters: 37,126,709 in both conditions.
- Existing baseline was reused read-only; only 80 Full-Fused jobs were trained.
- Completion: 80/80 jobs, zero worker errors, 80/80 paired audits.
- Source implementation commit: `bdca776`.

Artifacts:

- Baseline: `/data2/yb/experiments/gcnet_official_4dataset_10seed_20260820/CMUMOSI`
- Full-Fused: `/data2/yb/experiments/gcnet_mosi_full_fused_10seed_20260824`
- Smoke: `/data2/yb/experiments/gcnet_mosi_full_fused_smoke_20260824`

## Primary ten-seed results

Values are Weighted-F1 mean ± sample standard deviation. Delta is Full-Fused
minus the existing GCNet baseline in percentage points. Every seed is retained.

| Missing | GCNet baseline | Full-Fused | Raw delta | Wins/10 |
|---:|---:|---:|---:|---:|
| 0.0 | 80.42 ± 9.40 | 83.86 ± 1.22 | +3.44 | 8 |
| 0.1 | 81.85 ± 1.28 | 82.38 ± 1.47 | +0.54 | 6 |
| 0.2 | 79.84 ± 1.13 | 80.65 ± 1.54 | +0.81 | 6 |
| 0.3 | 77.75 ± 1.95 | 77.99 ± 2.69 | +0.24 | 5 |
| 0.4 | 75.26 ± 2.45 | 75.31 ± 1.15 | +0.05 | 5 |
| 0.5 | 73.66 ± 1.87 | 74.07 ± 1.55 | +0.41 | 7 |
| 0.6 | 70.68 ± 2.74 | 71.12 ± 2.91 | +0.44 | 4 |
| 0.7 | 69.31 ± 2.76 | 70.56 ± 1.74 | +1.26 | 7 |

All eight missing-rate means are positive. For the scientifically active
missing rates 0.1--0.7, the mean paired gain across 70 runs is +0.54 points,
with 40/70 seed-level wins and paired Wilcoxon p = 0.119. The seven rate-level
mean effects are all positive; Wilcoxon and exact sign tests over those seven
means both give p = 0.0156. This latter result is evidence of directional
consistency, not an independent-replication test, because rates share the same
dataset and seeds.

No individual missing rate survives multiple-comparison correction. The
smallest strict-filtered per-rate p-value is 0.074 at missing 0.0; its Holm value
is 0.594, and all other Holm values are 1.0.

## Collapse audit

The registered strict rule is:

```text
median(group) - score >= 0.10
and robust MAD z <= -3.5
```

It flags only GCNet baseline missing 0.0 seed 68 (W-F1 53.82). Removing the
corresponding pair changes missing-0.0 delta from +3.44 to +0.75 points
(83.38 to 84.13, nine pairs, p = 0.074).

This missing-0.0 difference cannot be attributed to Full-Fused reconstruction:
no modality is missing, both primary reconstruction losses are zero, and the
paired setup hashes match. It is another realization of GCNet/MOSI optimization
nondeterminism.

Across all rates:

| Analysis | Pairs | Mean delta | Wins | Paired Wilcoxon p |
|---|---:|---:|---:|---:|
| Raw | 80 | +0.90 | 48 | 0.0378 |
| Registered strict filter | 79 | +0.56 | 47 | 0.0548 |
| MAD-only sensitivity | 77 | +0.58 | 46 | 0.0468 |

The MAD-only sensitivity additionally flags baseline missing 0.0 seed 74 and
Full-Fused missing 0.1 seed 66. These do not satisfy the registered absolute
ten-point gap and are not labeled clear task collapses.

## Interpretation

Unlike IEMOCAPSix, MOSI shows a positive mean direction at every missing rate.
The effect is modest: the meaningful missing-only average is approximately half
a Weighted-F1 point, and only 40/70 individual pairs improve. Therefore the
result supports a cautious claim of cross-rate consistency, not a claim of a
large or uniformly seed-stable gain.

The strongest active-rate gains occur at missing 0.7 (+1.26 points) and 0.2
(+0.81 points). Missing 0.4 is effectively neutral (+0.05). The evidence is
compatible with complete-state supervision becoming useful in some incomplete
regimes, but it does not yet establish which modality information supplies the
benefit.

The next experiment should be narrow: measure visible-modality versus
missing-modality reconstruction gradients at missing 0.2, 0.4, and 0.7. This
directly tests why the same objective is helpful at 0.2/0.7 but neutral at 0.4,
without launching another broad architecture sweep.
