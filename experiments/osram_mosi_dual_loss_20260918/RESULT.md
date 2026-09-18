# MOSI dual-loss diagnostic

Internal diagnostic only. The scalar output is trained with MSE on all valid continuous labels plus BCEWithLogits on nonzero sign labels; test reporting uses the existing Non0 binary protocol.
The causal no-JEPA OSRAM, eta=0.6, Flat readout, cyclic missing-rate schedule, features, optimizer, and per-seed × per-rate Test-oracle selection are unchanged.

## Per-rate weighted F1 and accuracy

| rate | regression W-F1 | dual-loss W-F1 | Δ F1 | regression Acc | dual-loss Acc | Δ Acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.32±0.20 | 87.42±0.37 | +0.09 | 87.35 | 87.44 | +0.09 |
| 0.1 | 84.96±0.68 | 85.32±0.38 | +0.36 | 85.00 | 85.40 | +0.40 |
| 0.2 | 82.21±1.06 | 82.40±0.98 | +0.19 | 82.23 | 82.47 | +0.24 |
| 0.3 | 80.94±0.61 | 80.96±0.58 | +0.02 | 80.95 | 81.10 | +0.15 |
| 0.4 | 78.62±2.08 | 78.24±1.22 | -0.37 | 78.69 | 78.32 | -0.37 |
| 0.5 | 76.52±1.14 | 76.63±1.52 | +0.10 | 76.55 | 76.74 | +0.18 |
| 0.6 | 75.54±0.26 | 75.62±0.72 | +0.09 | 75.64 | 75.76 | +0.12 |
| 0.7 | 74.49±2.20 | 74.55±2.78 | +0.05 | 74.51 | 74.51 | +0.00 |

8-rate mean: regression 80.08%, dual-loss 80.14% (+0.07 pp).
High-missing (.5/.6/.7): regression 75.52%, dual-loss 75.60% (+0.08 pp).

## T-missing vs T-present

Original continuous neutral labels are excluded; binary negative/positive labels are retained.

| group | aggregation | regression | dual-loss | delta |
|---|---|---:|---:|---:|
| T-missing | pattern_macro | 66.17 | 66.46 | +0.29 |
| T-missing | sample_pooled | 66.18 | 66.45 | +0.26 |
| T-present | pattern_macro | 86.43 | 86.47 | +0.04 |
| T-present | sample_pooled | 86.67 | 86.62 | -0.05 |

## Seven-pattern sample-pooled W-F1

| pattern | regression | dual-loss | delta |
|---|---:|---:|---:|
| A | 66.26 | 67.55 | +1.28 |
| T | 86.28 | 86.25 | -0.03 |
| V | 65.58 | 64.93 | -0.65 |
| AT | 85.71 | 86.89 | +1.18 |
| AV | 66.65 | 66.89 | +0.24 |
| TV | 86.71 | 85.90 | -0.81 |
| ATV | 87.03 | 86.86 | -0.17 |
