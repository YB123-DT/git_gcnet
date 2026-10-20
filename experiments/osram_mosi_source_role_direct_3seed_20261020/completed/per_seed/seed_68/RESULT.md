# CMUMOSI Direct Source Weighting seed68

INTERNAL TEST-ORACLE SCREEN

One-stage original task MSE; no Router, second head, extra query, write change or auxiliary loss.
Existing matched Flat reference reused; original initialization, evaluation masks and training-mask traces checked.
Legacy scalar metric key router means the primary prediction, not a Router module.
Per-rate TEST W-F1 best; different rates may use different weights, not one deployment checkpoint.

| Rate | Flat ACC | New ACC | Flat W-F1 | New W-F1 | Delta W-F1 pp |
|---|---:|---:|---:|---:|---:|
| 0.0 | 86.8902 | 87.1951 | 86.7946 | 87.1199 | +0.3253 |
| 0.1 | 84.2988 | 83.8415 | 84.3472 | 83.8703 | -0.4770 |
| 0.2 | 81.8598 | 82.1646 | 81.7340 | 81.9568 | +0.2228 |
| 0.3 | 80.9451 | 80.9451 | 80.9752 | 80.8649 | -0.1103 |
| 0.4 | 78.3537 | 77.8963 | 78.3202 | 77.7741 | -0.5461 |
| 0.5 | 76.3720 | 75.1524 | 76.1908 | 75.2854 | -0.9055 |
| 0.6 | 74.0854 | 72.7134 | 74.0004 | 72.7200 | -1.2804 |
| 0.7 | 68.2927 | 67.9878 | 67.9886 | 68.1804 | +0.1918 |
| mean | 78.8872 | 78.4870 | 78.7939 | 78.4715 | -0.3224 |
| high | 72.9167 | 71.9512 | 72.7266 | 72.0619 | -0.6647 |

Source-role mode: direct.
Direct weighted source sum: gamma=2*sigmoid(score), NO original read add-back; initialization is tolerance-equal only.
Added parameters: 96487; total: 13606280.
Training wall seconds: 2002.2; peak allocated MiB: 3207.4.
No independent-test generalization claim; source labels denote additive joint Value-slot contributions, not pure modality semantics.
Diagnostics in per-epoch history are explicitly the LAST TRAINING BATCH, not epoch-global means.
