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

| group | no-JEPA | Text-Core |
|---|---:|---:|
| T-missing | 64.90 | 65.41 |
| T-present | 85.34 | 84.34 |

## Task-slot diagnostics

Real Text slot W-F1: 85.17%
Predicted Text slot W-F1: 65.40%
Centered cosine (predicted vs complete Text task slot): 0.1402
Prediction/target std ratio: 0.8636

## Seven-pattern details

See `pattern_per_seed.csv`; the key Text-missing patterns are A, V and AV.
