# CMUMOSI Coupled Source Role seed66

INTERNAL TEST-ORACLE SCREEN

One-stage original task MSE; no Router, second head, extra query, write change or auxiliary loss.
Existing matched Flat reference reused; original initialization, evaluation masks and training-mask traces checked.
Legacy scalar metric key router means the primary prediction, not a Router module.
Per-rate TEST W-F1 best; different rates may use different weights, not one deployment checkpoint.

| Rate | Flat ACC | New ACC | Flat W-F1 | New W-F1 | Delta W-F1 pp |
|---|---:|---:|---:|---:|---:|
| 0.0 | 86.7378 | 87.3476 | 86.7346 | 87.3381 | +0.6035 |
| 0.1 | 84.2988 | 84.9085 | 84.2326 | 84.6643 | +0.4316 |
| 0.2 | 82.3171 | 82.1646 | 82.2008 | 81.8040 | -0.3968 |
| 0.3 | 82.1646 | 80.7927 | 82.1220 | 80.7297 | -1.3923 |
| 0.4 | 77.5915 | 78.8110 | 77.7199 | 78.5271 | +0.8072 |
| 0.5 | 75.4573 | 76.3720 | 75.5949 | 76.3992 | +0.8043 |
| 0.6 | 74.8476 | 75.0000 | 74.9072 | 74.7799 | -0.1273 |
| 0.7 | 69.5122 | 68.1402 | 69.5122 | 68.1627 | -1.3495 |
| mean | 79.1159 | 79.1921 | 79.1280 | 79.0506 | -0.0774 |
| high | 73.2724 | 73.1707 | 73.3381 | 73.1139 | -0.2242 |

Added parameters: 96487; total: 13606280.
Training wall seconds: 1662.8; peak allocated MiB: 3238.3.
No independent-test generalization claim; source labels denote additive joint Value-slot contributions, not pure modality semantics.
Diagnostics in per-epoch history are explicitly the LAST TRAINING BATCH, not epoch-global means.
