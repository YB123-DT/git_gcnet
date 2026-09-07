# OSRAM JEPA coupling diagnostic

**Internal diagnostic only; not a formal paper result.**

Joint is inherited Full-4E OSRAM with classification plus JEPA. The
Emotion-only condition changes only `training_objective` and removes
the JEPA gradient from the training objective.

| Condition | 8-rate mean | High-missing mean (0.5/0.6/0.7) |
|---|---:|---:|
| Joint | 80.781% | 76.586% |
| Emotion-only | 80.545% | 76.432% |

## Strict one-checkpoint cross-rate check

| Condition | mean | selected epochs (66,67,68,69,70) |
|---|---:|---|
| Joint | 80.267% | [61, 39, 46, 38, 38] |
| Emotion-only | 79.870% | [59, 57, 42, 47, 48] |
| Δ | -0.396 percentage points | — |

| Rate | Joint | Emotion-only | Δ | positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.639% ± 0.669 | 87.231% ± 0.584 | -0.408 | 1/5 |
| 0.1 | 85.365% ± 0.357 | 85.225% ± 0.494 | -0.140 | 1/5 |
| 0.2 | 82.693% ± 1.090 | 82.279% ± 0.942 | -0.414 | 1/5 |
| 0.3 | 81.439% ± 0.651 | 81.203% ± 0.663 | -0.235 | 2/5 |
| 0.4 | 79.350% ± 1.521 | 79.123% ± 2.096 | -0.227 | 2/5 |
| 0.5 | 77.300% ± 0.825 | 76.976% ± 1.053 | -0.325 | 2/5 |
| 0.6 | 76.804% ± 0.535 | 76.753% ± 1.320 | -0.050 | 2/5 |
| 0.7 | 75.654% ± 1.940 | 75.568% ± 2.106 | -0.087 | 3/5 |

The result distinguishes objective coupling from predictor architecture.
A positive Emotion-only delta means the current JEPA gradient is noisy
for OSRAM; a positive Joint delta means JEPA is useful regularization.
