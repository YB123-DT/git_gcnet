# MOSI direct source weighting: interim progress

INTERNAL TEST-ORACLE SCREEN

Checked 2026-10-20T12:59:06.169740+00:00; training 108/150 on biggpu GPU0.
Direct coefficients are2*sigmoid(score); original Base/Gap are NOT added back.
Direct results are best-so-far through108epochs; Flat/residual references completed150.
These are exploratory per-rate Test-best W-F1, not a single deployment checkpoint.

| Rate | Flat | Residual | Direct, interim | Direct minus Flat pp | Direct selected epoch |
|---|---:|---:|---:|---:|---:|
| 0.0 | 86.735 | 87.338 | 87.145 | +0.411 | 60 |
| 0.1 | 84.233 | 84.664 | 84.451 | +0.219 | 76 |
| 0.2 | 82.201 | 81.804 | 83.119 | +0.919 | 82 |
| 0.3 | 82.122 | 80.730 | 81.330 | -0.792 | 74 |
| 0.4 | 77.720 | 78.527 | 79.117 | +1.397 | 63 |
| 0.5 | 75.595 | 76.399 | 76.508 | +0.913 | 85 |
| 0.6 | 74.907 | 74.780 | 75.023 | +0.116 | 63 |
| 0.7 | 69.512 | 68.163 | 68.902 | -0.610 | 63 |
| 8-rate mean | 79.128 | 79.051 | 79.450 | +0.322 | — |
| High .5/.6/.7 | 73.338 | 73.114 | 73.478 | +0.140 | — |

Same baseline initialization,split,seed,test mask banks and approved training config checked.
All108 completed epochs passed runtime training-mask/update-count/LR parity against Flat.
Latest new-module gradients:14connected/14nonzero; LR is1e-4 after epoch100.
Best-so-far metrics are from existing selected-epoch metadata; final prediction artifact
recomputation is pending complete training. No new inference/training, masks or parameters
were changed for this status analysis. PROGRESS.json records exact source paths and checks.

Preliminary signal is positive in the average but not uniform across rates:
.3 and.7 remain below Flat. Do not infer multi-seed reliability, statistical significance
or an isolated residual-removal effect (the coefficient range also changed).
Continue only the already running150-epoch job; no extra seeds/search were started.
