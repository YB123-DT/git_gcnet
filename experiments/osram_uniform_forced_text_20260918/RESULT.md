# Continuous-rate forced-Text no-JEPA MOSI diagnostic

Internal diagnostic only; training samples use per-utterance r~Uniform(0,1), P(Text forced missing)=0.25, and at least one observed modality. Test checkpoints use the inherited per-seed × per-rate Test-oracle protocol.

## Per-rate weighted F1

| rate | no-JEPA | uniform-forced-text | delta | positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.32±0.20 | 87.07±0.34 | -0.26 | 1/5 |
| 0.1 | 84.96±0.68 | 84.57±1.03 | -0.39 | 2/5 |
| 0.2 | 82.21±1.06 | 81.90±1.57 | -0.31 | 2/5 |
| 0.3 | 80.94±0.61 | 80.84±0.61 | -0.10 | 1/5 |
| 0.4 | 78.62±2.08 | 77.88±0.72 | -0.74 | 2/5 |
| 0.5 | 76.52±1.14 | 76.14±1.11 | -0.38 | 2/5 |
| 0.6 | 75.54±0.26 | 75.52±1.45 | -0.01 | 3/5 |
| 0.7 | 74.49±2.20 | 74.01±2.23 | -0.48 | 1/5 |

8-rate mean: no-JEPA 80.08%, uniform-forced-text 79.74% (-0.33 pp).
High-missing (.5/.6/.7): no-JEPA 75.52%, uniform-forced-text 75.23% (-0.29 pp).

## Training mask audit

Mean sampled r: 0.5003; realized missing fraction: 0.4517; forced-Text fraction: 0.2511; target probability: 0.25.

## T-missing vs T-present

W-F1 excludes label==0. Pattern-macro averages per-pattern W-F1; sample-pooled concatenates underlying valid samples before computing W-F1.

| group | aggregation | no-JEPA | uniform-forced-text | delta | valid samples |
|---|---|---:|---:|---:|---:|
| T-missing | pattern_macro | 65.42 | 65.43 | +0.01 | 8401/8401 |
| T-missing | sample_pooled | 66.18 | 65.88 | -0.31 | 8401/8401 |
| T-present | pattern_macro | 86.33 | 85.89 | -0.44 | 17839/17839 |
| T-present | sample_pooled | 86.67 | 86.37 | -0.30 | 17839/17839 |

## Seven-pattern sample-pooled W-F1

| pattern | no-JEPA | uniform-forced-text | delta |
|---|---:|---:|---:|
| A | 66.26 | 65.67 | -0.60 |
| T | 86.28 | 85.96 | -0.32 |
| V | 65.58 | 65.53 | -0.05 |
| AT | 85.71 | 86.17 | +0.46 |
| AV | 66.65 | 66.35 | -0.30 |
| TV | 86.71 | 86.17 | -0.55 |
| ATV | 87.03 | 86.60 | -0.43 |
