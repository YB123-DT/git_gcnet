# IEMOCAP-6 Nested128: five folds, three seeds

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY

Fixed nested_dim128, 100 epochs, Adam LR=1e-3, WD=1e-5, batch32; cmerc-repro.
Original gcnet-random-mask-v1 cyclic training; fixed accepted per-fold test masks.
Each rate selects TEST ACC with earliest strict improvement; UA/W-F1/CE use the SAME call.
These eight-rate selections are not a single deployable checkpoint.
Only completed trajectories enter this report: 14 fresh runs plus existing seed66 fold5.
Nested128: 13,950,441 total parameters; 432,387 additional Nested parameters.
Five-fold metrics are equally weighted fold means, not pooled utterance metrics.
Across-seed SD uses n-1 on the three five-fold means.
No strict comparison to historical Flat: old IEMOCAP runs resampled evaluation masks and
used a different fold-initialization treatment. No matched fifteen-run Flat was trained here.
The selection of Nested128 itself used seed66 fold5; this is exploratory, not independent generalization.

## Five-fold results by seed

| Seed | ACC | UA | W-F1 | CE | High ACC | High UA | High W-F1 | High CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 62.009 | 60.574 | 61.602 | 1.131 | 60.176 | 58.579 | 59.503 | 1.149 |
| 67 | 61.792 | 60.025 | 61.347 | 1.132 | 59.500 | 57.593 | 58.931 | 1.140 |
| 68 | 62.180 | 60.807 | 61.876 | 1.049 | 59.651 | 57.943 | 59.270 | 1.110 |
| Mean ± sample SD | 61.994 ± 0.195 | 60.469 ± 0.402 | 61.608 ± 0.264 | 1.104 ± 0.047 | 59.776 ± 0.354 | 58.038 ± 0.500 | 59.235 ± 0.287 | 1.133 ± 0.020 |

## Per-rate three-seed five-fold mean

| Missing | ACC | UA | W-F1 | CE |
|---|---:|---:|---:|---:|
| 0.0 | 64.894 ± 0.551 | 63.342 ± 1.105 | 64.568 ± 0.550 | 1.068 ± 0.088 |
| 0.1 | 64.163 ± 0.233 | 63.241 ± 0.685 | 63.860 ± 0.437 | 1.048 ± 0.036 |
| 0.2 | 62.922 ± 0.464 | 61.409 ± 0.325 | 62.648 ± 0.390 | 1.117 ± 0.061 |
| 0.3 | 62.833 ± 0.229 | 61.189 ± 0.218 | 62.573 ± 0.171 | 1.075 ± 0.087 |
| 0.4 | 61.811 ± 0.255 | 60.453 ± 0.705 | 61.511 ± 0.339 | 1.125 ± 0.085 |
| 0.5 | 61.597 ± 0.129 | 59.823 ± 0.361 | 61.022 ± 0.462 | 1.075 ± 0.040 |
| 0.6 | 59.645 ± 0.428 | 57.620 ± 0.695 | 59.270 ± 0.387 | 1.164 ± 0.011 |
| 0.7 | 58.085 ± 0.608 | 56.672 ± 0.635 | 57.413 ± 0.477 | 1.160 ± 0.009 |

PER_FOLD_RATE.csv preserves all 120 fold/seed/rate selections and epochs.
PER_SEED_RATE.csv preserves the 24 equally weighted five-fold rate summaries.
Best checkpoints, last.pt, masks and exact selected predictions remain in the remote run directories.
No new model, loss, HPO or extra dataset was introduced. This report makes no mechanism claim.
