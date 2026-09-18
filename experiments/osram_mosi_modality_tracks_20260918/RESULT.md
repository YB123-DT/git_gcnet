# MOSI modality-track no-JEPA diagnostic

Only the emotion readout local path changes: OSRAM fused-node query/read/write and all JEPA/MMoE paths remain unchanged. Missing modality tracks are zeroed, then modality tracks, Base, missing-masked Gap contexts and availability are sent to one flat MLP.
This is an internal Test-oracle diagnostic, not a formal paper result. Epoch selection is independent per seed and missing rate.

## Per-rate weighted F1

| rate | no-JEPA | modality-tracks | delta | positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.32±0.20 | 87.04±0.71 | -0.28 | 1/5 |
| 0.1 | 84.96±0.68 | 84.75±0.53 | -0.22 | 2/5 |
| 0.2 | 82.21±1.06 | 81.92±1.05 | -0.29 | 2/5 |
| 0.3 | 80.94±0.61 | 80.25±1.22 | -0.69 | 1/5 |
| 0.4 | 78.62±2.08 | 77.32±1.80 | -1.30 | 1/5 |
| 0.5 | 76.52±1.14 | 76.40±1.31 | -0.13 | 2/5 |
| 0.6 | 75.54±0.26 | 75.87±0.97 | +0.34 | 4/5 |
| 0.7 | 74.49±2.20 | 72.73±2.70 | -1.76 | 0/5 |

8-rate mean: no-JEPA 80.08%, modality-tracks 79.54% (-0.54 pp; 1/5 seeds positive).
High-missing (.5/.6/.7): no-JEPA 75.52%, modality-tracks 75.00% (-0.52 pp; 1/5 seeds positive).

## T-missing vs T-present

| group | aggregation | no-JEPA | modality-tracks | delta |
|---|---|---:|---:|---:|
| T-missing | pattern_macro | 66.17 | 65.14 | -1.03 |
| T-missing | sample_pooled | 66.18 | 65.12 | -1.06 |
| T-present | pattern_macro | 86.43 | 86.39 | -0.04 |
| T-present | sample_pooled | 86.67 | 86.40 | -0.27 |

## Seven-pattern sample-pooled W-F1

| pattern | no-JEPA | modality-tracks | delta |
|---|---:|---:|---:|
| A | 66.26 | 65.00 | -1.26 |
| T | 86.28 | 86.32 | +0.04 |
| V | 65.58 | 64.72 | -0.86 |
| AT | 85.71 | 86.06 | +0.36 |
| AV | 66.65 | 65.70 | -0.95 |
| TV | 86.71 | 86.75 | +0.04 |
| ATV | 87.03 | 86.42 | -0.61 |
