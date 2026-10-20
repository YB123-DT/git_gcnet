# CMUMOSI Direct Source Weighting seed67

INTERNAL TEST-ORACLE SCREEN

One-stage original task MSE; no Router, second head, extra query, write change or auxiliary loss.
Existing matched Flat reference reused; original initialization, evaluation masks and training-mask traces checked.
Legacy scalar metric key router means the primary prediction, not a Router module.
Per-rate TEST W-F1 best; different rates may use different weights, not one deployment checkpoint.

| Rate | Flat ACC | New ACC | Flat W-F1 | New W-F1 | Delta W-F1 pp |
|---|---:|---:|---:|---:|---:|
| 0.0 | 87.5000 | 87.1951 | 87.4590 | 87.2409 | -0.2182 |
| 0.1 | 84.1463 | 83.9939 | 84.1218 | 83.7349 | -0.3870 |
| 0.2 | 82.6220 | 82.4695 | 82.6840 | 82.3843 | -0.2997 |
| 0.3 | 80.3354 | 80.3354 | 80.2266 | 80.3580 | +0.1314 |
| 0.4 | 77.4390 | 77.8963 | 77.5386 | 77.9488 | +0.4102 |
| 0.5 | 75.6098 | 75.7622 | 75.7422 | 75.7901 | +0.0479 |
| 0.6 | 73.3232 | 73.3232 | 73.3031 | 73.3539 | +0.0508 |
| 0.7 | 68.9024 | 67.6829 | 68.9314 | 67.8171 | -1.1143 |
| mean | 78.7348 | 78.5823 | 78.7509 | 78.5785 | -0.1724 |
| high | 72.6118 | 72.2561 | 72.6589 | 72.3204 | -0.3385 |

Source-role mode: direct.
Direct weighted source sum: gamma=2*sigmoid(score), NO original read add-back; initialization is tolerance-equal only.
Added parameters: 96487; total: 13606280.
Training wall seconds: 1964.3; peak allocated MiB: 3209.3.
No independent-test generalization claim; source labels denote additive joint Value-slot contributions, not pure modality semantics.
Diagnostics in per-epoch history are explicitly the LAST TRAINING BATCH, not epoch-global means.
