# MOSI GCNet Random Missing: Flat vs one-stage Router

INTERNAL EXPLORATORY EXPERIMENT

Three seeds, six fresh runs, 150 epochs. Original GCNet random-mask-v1.
One validation-selected checkpoint per run; not per-rate Test-best.
Historical 80.747 used conversation-mask-v2 and is not a paired baseline.

| Seed | Model | Selected epoch | ACC | W-F1 | High ACC | High W-F1 |
|---:|---|---:|---:|---:|---:|---:|
| 66 | flat | 78 | 75.0762 | 75.1303 | 67.5305 | 67.4940 |
| 66 | joint | 43 | 76.5434 | 76.5715 | 69.1565 | 69.2558 |
| 67 | flat | 80 | 77.4962 | 77.5678 | 70.1728 | 70.3293 |
| 67 | joint | 85 | 74.4855 | 74.1560 | 65.7012 | 64.7432 |
| 68 | flat | 110 | 76.5434 | 76.5668 | 70.6809 | 70.8395 |
| 68 | joint | 133 | 76.6387 | 76.6370 | 69.3598 | 69.2076 |

## Three-seed mean ± sample standard deviation

| Model | ACC | W-F1 | High ACC | High W-F1 |
|---|---:|---:|---:|---:|
| flat | 76.3720 ± 1.2191 | 76.4216 ± 1.2252 | 69.4614 ± 1.6914 | 69.5543 ± 1.8024 |
| joint | 75.8892 ± 1.2166 | 75.7881 ± 1.4139 | 68.0725 ± 2.0561 | 67.7355 ± 2.5916 |

## Per-rate three-seed means

| Rate | Flat ACC | Joint ACC | Flat W-F1 | Joint W-F1 |
|---|---:|---:|---:|---:|
| 0.0 | 85.8232 | 86.3313 | 85.7760 | 86.3374 |
| 0.1 | 83.0285 | 82.8760 | 83.0061 | 82.9087 |
| 0.2 | 80.7927 | 80.6911 | 80.7836 | 80.7538 |
| 0.3 | 77.4390 | 77.6423 | 77.5218 | 77.6949 |
| 0.4 | 75.5081 | 75.3557 | 75.6228 | 75.4036 |
| 0.5 | 73.3232 | 72.1037 | 73.4142 | 72.0737 |
| 0.6 | 70.9858 | 68.3435 | 71.0980 | 68.0045 |
| 0.7 | 64.0752 | 63.7703 | 64.1506 | 63.1284 |

No independent-test paper claim. No automatic method, seed or HPO expansion.
Joint-vs-Flat is the full training-method comparison; Joint full candidate is not a separately trained Flat.
