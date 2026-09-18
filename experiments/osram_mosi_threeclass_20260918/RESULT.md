# MOSI three-class-training / Non0 binary-test diagnostic

Internal diagnostic only. Training uses three-class CE (negative/neutral/positive); test selection and reporting remove original continuous label 0 and compare only negative versus positive logits.
The causal no-JEPA OSRAM, eta=0.6, Flat readout, cyclic masks, features, optimizer, and per-seed × per-rate Test-oracle selection are unchanged.

## Per-rate weighted F1 and accuracy

| rate | regression W-F1 | three-class W-F1 | Δ F1 | regression Acc | three-class Acc | Δ Acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.32±0.20 | 86.74±0.30 | -0.59 | 87.35 | 86.77 | -0.58 |
| 0.1 | 84.96±0.68 | 84.73±0.74 | -0.24 | 85.00 | 84.76 | -0.24 |
| 0.2 | 82.21±1.06 | 81.61±0.98 | -0.60 | 82.23 | 81.62 | -0.61 |
| 0.3 | 80.94±0.61 | 80.72±0.93 | -0.22 | 80.95 | 80.82 | -0.12 |
| 0.4 | 78.62±2.08 | 77.84±0.99 | -0.77 | 78.69 | 77.93 | -0.76 |
| 0.5 | 76.52±1.14 | 75.74±1.24 | -0.79 | 76.55 | 75.82 | -0.73 |
| 0.6 | 75.54±0.26 | 75.36±1.08 | -0.17 | 75.64 | 75.67 | +0.03 |
| 0.7 | 74.49±2.20 | 74.30±2.62 | -0.19 | 74.51 | 74.54 | +0.03 |

8-rate mean: regression 80.08%, three-class 79.63% (-0.44 pp).
High-missing (.5/.6/.7): regression 75.52%, three-class 75.14% (-0.38 pp).

## T-missing vs T-present

Original continuous neutral labels are excluded; binary negative/positive labels are retained.

| group | aggregation | regression | three-class | delta |
|---|---|---:|---:|---:|
| T-missing | pattern_macro | 66.17 | 65.84 | -0.33 |
| T-missing | sample_pooled | 66.18 | 65.87 | -0.32 |
| T-present | pattern_macro | 86.43 | 86.02 | -0.41 |
| T-present | sample_pooled | 86.67 | 86.18 | -0.49 |

## Seven-pattern sample-pooled W-F1

| pattern | regression | three-class | delta |
|---|---:|---:|---:|
| A | 66.26 | 65.64 | -0.63 |
| T | 86.28 | 85.23 | -1.05 |
| V | 65.58 | 65.55 | -0.04 |
| AT | 85.71 | 85.79 | +0.09 |
| AV | 66.65 | 66.33 | -0.32 |
| TV | 86.71 | 86.63 | -0.08 |
| ATV | 87.03 | 86.44 | -0.59 |
