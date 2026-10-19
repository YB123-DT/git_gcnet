# MELD / flat one-stage run

INTERNAL SINGLE-SEED EXPLORATION

Source: `79619b8a2fd9f6c7e82bc67546aa8f9ac82105db`. One validation-selected checkpoint; selected epoch: 37.
Fresh initialization, original GCNet random missing; no persistent augmentation.
For Joint, full denotes its own jointly trained full candidate, NOT a separate Flat control.

| Rate | Full ACC | Final ACC | Full UA | Final UA | Final W-F1 | Final CE |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 65.6322 | 65.6322 | 44.6685 | 44.6685 | 64.4536 | 1.1047 |
| 0.1 | 64.2146 | 64.2146 | 43.2478 | 43.2478 | 62.8327 | 1.1329 |
| 0.2 | 63.1418 | 63.1418 | 41.7453 | 41.7453 | 61.3751 | 1.1572 |
| 0.3 | 61.3410 | 61.3410 | 38.2195 | 38.2195 | 58.9455 | 1.2054 |
| 0.4 | 59.7318 | 59.7318 | 37.1094 | 37.1094 | 57.2450 | 1.2252 |
| 0.5 | 58.0077 | 58.0077 | 33.3294 | 33.3294 | 54.5792 | 1.2292 |
| 0.6 | 56.5517 | 56.5517 | 32.1480 | 32.1480 | 52.4200 | 1.3156 |
| 0.7 | 54.8659 | 54.8659 | 29.5614 | 29.5614 | 50.0314 | 1.3417 |

Compare the separate matched Flat job before interpreting any benefit.
No multi-seed, multi-fold or independent generalization claim.
