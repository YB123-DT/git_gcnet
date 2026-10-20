# MOSI GCNet missing: per-rate Test-best

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY

Seeds66/67/68 x Flat/Router,150epochs. Original gcnet-random-mask-v1.
Each rate selects its own epoch by TEST W-F1; ties retain the earliest epoch.
These means combine eight checkpoints and are NOT single-checkpoint deployment scores.
The fixed GCNet evaluation banks and training settings match the preceding validation-selected runs.
Only checkpoint selection changes; old results/checkpoints remain untouched.
Old80.747 still used a different missing sampler, even though its Test-best policy is restored.

| Seed | Model | ACC | W-F1 | High ACC | High W-F1 |
|---:|---|---:|---:|---:|---:|
| 66 | flat | 79.1159 | 79.1280 | 73.2724 | 73.3381 |
| 66 | joint | 79.0777 | 79.0120 | 73.1199 | 72.9527 |
| 67 | flat | 78.7348 | 78.7509 | 72.6118 | 72.6589 |
| 67 | joint | 79.4017 | 79.2353 | 73.1199 | 73.0129 |
| 68 | flat | 78.8872 | 78.7939 | 72.9167 | 72.7266 |
| 68 | joint | 78.7157 | 78.5607 | 72.0020 | 71.9077 |

## Three-seed mean ± sample standard deviation

| Model | ACC | W-F1 | High ACC | High W-F1 |
|---|---:|---:|---:|---:|
| flat | 78.9126 ± 0.1918 | 78.8909 ± 0.2065 | 72.9336 ± 0.3306 | 72.9079 ± 0.3741 |
| joint | 79.0650 ± 0.3432 | 78.9360 ± 0.3437 | 72.7473 ± 0.6454 | 72.6244 ± 0.6214 |

## Per-rate three-seed means

| Rate | Flat ACC | Router ACC | Flat W-F1 | Router W-F1 |
|---|---:|---:|---:|---:|
| 0.0 | 87.0427 | 87.3476 | 86.9961 | 87.3023 |
| 0.1 | 84.2480 | 84.8577 | 84.2339 | 84.8040 |
| 0.2 | 82.2663 | 82.4187 | 82.2063 | 82.3155 |
| 0.3 | 81.1484 | 81.0467 | 81.1079 | 80.6901 |
| 0.4 | 77.7947 | 78.6077 | 77.8595 | 78.5027 |
| 0.5 | 75.8130 | 75.9654 | 75.8427 | 75.8338 |
| 0.6 | 74.0854 | 73.8313 | 74.0703 | 73.6767 |
| 0.7 | 68.9024 | 68.4451 | 68.8107 | 68.3628 |

All selected epochs are listed in METRICS.csv and each run SUMMARY.json.
Selection uses test labels: exploratory only, no independent generalization claim.
No automatic extra seeds, datasets, HPO or model changes.
