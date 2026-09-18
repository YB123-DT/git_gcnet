# Text-Core MOSI diagnostic

Internal diagnostic only; each seed × missing rate selects its own Test W-F1 epoch.

## Per-rate weighted F1

| rate | no-JEPA | Text-Core | delta | positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.32±0.20 | 86.76±0.29 | -0.56 | 0/5 |
| 0.1 | 84.96±0.68 | 84.34±0.97 | -0.62 | 1/5 |
| 0.2 | 82.21±1.06 | 82.28±1.30 | +0.06 | 2/5 |
| 0.3 | 80.94±0.61 | 80.47±0.47 | -0.46 | 2/5 |
| 0.4 | 78.62±2.08 | 78.33±0.91 | -0.29 | 2/5 |
| 0.5 | 76.52±1.14 | 76.32±1.56 | -0.20 | 1/5 |
| 0.6 | 75.54±0.26 | 75.92±1.03 | +0.38 | 3/5 |
| 0.7 | 74.49±2.20 | 73.98±2.52 | -0.51 | 1/5 |

8-rate mean: no-JEPA 80.08%, Text-Core 79.80%.
High-missing (.5/.6/.7): no-JEPA 75.52%, Text-Core 75.41%.

## T-missing vs T-present

W-F1 excludes label==0. `pattern-macro` first averages valid W-F1 within each pattern, then averages patterns; `sample-pooled` concatenates the underlying valid samples before computing W-F1.

| group | aggregation | no-JEPA | Text-Core | delta | valid samples (no-JEPA/Text-Core) |
|---|---|---:|---:|---:|---:|
| T-missing | pattern_macro | 65.42 | 66.05 | +0.63 | 8401/8401 |
| T-missing | sample_pooled | 66.18 | 66.73 | +0.55 | 8401/8401 |
| T-present | pattern_macro | 86.33 | 85.41 | -0.92 | 17839/17839 |
| T-present | sample_pooled | 86.67 | 86.04 | -0.63 | 17839/17839 |

## Seven-pattern summary

| pattern | aggregation | no-JEPA | Text-Core | delta |
|---|---|---:|---:|---:|
| A | pattern-macro | 65.93 | 66.27 | +0.34 |
| A | sample-pooled | 66.26 | 66.68 | +0.42 |
| T | pattern-macro | 85.10 | 83.27 | -1.83 |
| T | sample-pooled | 86.28 | 85.55 | -0.73 |
| V | pattern-macro | 65.03 | 66.05 | +1.02 |
| V | sample-pooled | 65.58 | 66.50 | +0.92 |
| AT | pattern-macro | 85.97 | 86.28 | +0.31 |
| AT | sample-pooled | 85.71 | 86.14 | +0.43 |
| AV | pattern-macro | 65.29 | 65.82 | +0.52 |
| AV | sample-pooled | 66.65 | 67.02 | +0.37 |
| TV | pattern-macro | 86.60 | 85.79 | -0.81 |
| TV | sample-pooled | 86.71 | 86.02 | -0.69 |
| ATV | pattern-macro | 87.66 | 86.30 | -1.36 |
| ATV | sample-pooled | 87.03 | 86.16 | -0.86 |

## Task-slot diagnostics

Real Text slot W-F1: 85.17%
Predicted Text slot W-F1: 65.40%
Centered cosine (old per-sample protocol; T-missing only): 0.1027±0.0346
Prediction/target std ratio (T-missing only; old mean-channel population-std protocol): 0.8289±0.1090

`pattern_per_seed.csv` excludes label==0 from `count` and W-F1 and records the excluded count explicitly.
