# MOSI binary-task no-JEPA diagnostic

Internal diagnostic only; the sole configuration change from the inherited no-JEPA causal OSRAM is `mosi_task_mode: regression -> binary`.
Test masks, cyclic training schedule, eta=0.6, Flat readout, optimizer, features, and per-seed × per-rate Test-oracle selection are unchanged.

## Per-rate weighted F1

| rate | no-JEPA regression | binary | delta | positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.32±0.20 | 86.88±0.27 | -0.44 | 0/5 |
| 0.1 | 84.96±0.68 | 84.69±0.62 | -0.27 | 2/5 |
| 0.2 | 82.21±1.06 | 81.96±0.83 | -0.25 | 2/5 |
| 0.3 | 80.94±0.61 | 80.34±0.76 | -0.60 | 0/5 |
| 0.4 | 78.62±2.08 | 77.93±1.21 | -0.69 | 2/5 |
| 0.5 | 76.52±1.14 | 76.30±1.38 | -0.22 | 1/5 |
| 0.6 | 75.54±0.26 | 76.44±0.90 | +0.90 | 4/5 |
| 0.7 | 74.49±2.20 | 73.83±3.16 | -0.66 | 1/5 |

8-rate mean: no-JEPA regression 80.08%, binary 79.80% (-0.28 pp).
High-missing (.5/.6/.7): no-JEPA regression 75.52%, binary 75.52% (+0.00 pp).

## T-missing vs T-present

Scores exclude label==0. Pattern-macro averages the three/four pattern scores; sample-pooled concatenates all valid samples.

| group | aggregation | no-JEPA regression | binary | delta |
|---|---|---:|---:|---:|
| T-missing | pattern_macro | 66.17 | 66.08 | -0.08 |
| T-missing | sample_pooled | 66.18 | 66.07 | -0.12 |
| T-present | pattern_macro | 86.43 | 86.19 | -0.24 |
| T-present | sample_pooled | 86.67 | 86.32 | -0.35 |

## Seven-pattern sample-pooled W-F1

| pattern | no-JEPA regression | binary | delta |
|---|---:|---:|---:|
| A | 66.26 | 64.66 | -1.60 |
| T | 86.28 | 85.96 | -0.32 |
| V | 65.58 | 66.24 | +0.66 |
| AT | 85.71 | 85.77 | +0.06 |
| AV | 66.65 | 67.34 | +0.69 |
| TV | 86.71 | 86.52 | -0.20 |
| ATV | 87.03 | 86.51 | -0.52 |
