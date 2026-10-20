# CMUMOSI Direct Source Weighting seed66

INTERNAL TEST-ORACLE SCREEN

One-stage original task MSE; no Router, second head, extra query, write change or auxiliary loss.
Existing matched Flat reference reused; original initialization, evaluation masks and training-mask traces checked.
Legacy scalar metric key router means the primary prediction, not a Router module.
Per-rate TEST W-F1 best; different rates may use different weights, not one deployment checkpoint.

| Rate | Flat ACC | New ACC | Flat W-F1 | New W-F1 | Delta W-F1 pp |
|---|---:|---:|---:|---:|---:|
| 0.0 | 86.7378 | 87.1951 | 86.7346 | 87.1452 | +0.4107 |
| 0.1 | 84.2988 | 84.4512 | 84.2326 | 84.4512 | +0.2186 |
| 0.2 | 82.3171 | 83.0793 | 82.2008 | 83.1194 | +0.9186 |
| 0.3 | 82.1646 | 81.4024 | 82.1220 | 81.3300 | -0.7920 |
| 0.4 | 77.5915 | 79.2683 | 77.7199 | 79.1170 | +1.3972 |
| 0.5 | 75.4573 | 76.3720 | 75.5949 | 76.5081 | +0.9132 |
| 0.6 | 74.8476 | 75.0000 | 74.9072 | 75.0233 | +0.1161 |
| 0.7 | 69.5122 | 68.9024 | 69.5122 | 68.9024 | -0.6098 |
| mean | 79.1159 | 79.4588 | 79.1280 | 79.4496 | +0.3216 |
| high | 73.2724 | 73.4248 | 73.3381 | 73.4779 | +0.1398 |

Source-role mode: direct.
Direct weighted source sum: gamma=2*sigmoid(score), NO original read add-back; initialization is tolerance-equal only.
Added parameters: 96487; total: 13606280.
Training wall seconds: 1627.3; peak allocated MiB: 3207.7.
No independent-test generalization claim; source labels denote additive joint Value-slot contributions, not pure modality semantics.
Diagnostics in per-epoch history are explicitly the LAST TRAINING BATCH, not epoch-global means.
