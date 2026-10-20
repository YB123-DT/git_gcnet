# MOSI Direct Source Weighting: three seeds

INTERNAL TEST-ORACLE SCREEN

Seed66 is reused, not retrained. Seeds67/68 use the unchanged algorithm, original s0 runtime,GCNet missing and150epochs.
Per-rate earliest Test-best W-F1; matching same-prediction ACC. Not one deployment checkpoint.

| Rate | Seed66 W-F1 | Seed67 W-F1 | Seed68 W-F1 | Mean ± sample std | Flat mean | Delta pp |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.145 | 87.241 | 87.120 | 87.169 ± 0.064 | 86.996 | +0.173 |
| 0.1 | 84.451 | 83.735 | 83.870 | 84.019 ± 0.381 | 84.234 | -0.215 |
| 0.2 | 83.119 | 82.384 | 81.957 | 82.487 ± 0.588 | 82.206 | +0.281 |
| 0.3 | 81.330 | 80.358 | 80.865 | 80.851 ± 0.486 | 81.108 | -0.257 |
| 0.4 | 79.117 | 77.949 | 77.774 | 78.280 ± 0.730 | 77.860 | +0.420 |
| 0.5 | 76.508 | 75.790 | 75.285 | 75.861 ± 0.614 | 75.843 | +0.019 |
| 0.6 | 75.023 | 73.354 | 72.720 | 73.699 ± 1.190 | 74.070 | -0.371 |
| 0.7 | 68.902 | 67.817 | 68.180 | 68.300 ± 0.552 | 68.811 | -0.511 |

| Scope | Flat W-F1 | Direct W-F1 | Delta pp | Flat ACC | Direct ACC |
|---|---:|---:|---:|---:|---:|
| mean | 78.891 ± 0.206 | 78.833 ± 0.536 | -0.058 | 78.913 | 78.843 |
| high | 72.908 ± 0.374 | 72.620 ± 0.754 | -0.288 | 72.934 | 72.544 |

High means.5/.6/.7 averaged within each seed before computing across-seed statistics.
All selected prediction artifacts were recomputed and matched to same-seed Flat cohorts/masks.
Three seeds are a finite exploratory repeat, not a new hyperparameter search or independent-test claim.
