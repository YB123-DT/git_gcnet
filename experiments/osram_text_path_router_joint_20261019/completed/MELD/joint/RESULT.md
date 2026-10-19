# MELD / joint one-stage run

INTERNAL SINGLE-SEED EXPLORATION

Source: `79619b8a2fd9f6c7e82bc67546aa8f9ac82105db`. One validation-selected checkpoint; selected epoch: 37.
Fresh initialization, original GCNet random missing; no persistent augmentation.
For Joint, full denotes its own jointly trained full candidate, NOT a separate Flat control.

| Rate | Full ACC | Final ACC | Full UA | Final UA | Final W-F1 | Final CE |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 65.4789 | 65.4789 | 44.4420 | 44.4420 | 64.2834 | 1.1060 |
| 0.1 | 64.0996 | 64.0996 | 43.3375 | 43.3375 | 62.7170 | 1.1341 |
| 0.2 | 63.1034 | 63.1801 | 41.6766 | 41.6994 | 61.3831 | 1.1583 |
| 0.3 | 61.4176 | 61.4176 | 38.2664 | 38.2664 | 59.0296 | 1.2060 |
| 0.4 | 59.8467 | 59.8084 | 37.0987 | 37.0874 | 57.2779 | 1.2261 |
| 0.5 | 57.8927 | 57.8927 | 33.2317 | 33.2317 | 54.4584 | 1.2296 |
| 0.6 | 56.3218 | 56.3602 | 31.9808 | 32.0222 | 52.2170 | 1.3166 |
| 0.7 | 54.7126 | 54.7126 | 29.3614 | 29.3614 | 49.8135 | 1.3421 |

Compare the separate matched Flat job before interpreting any benefit.
No multi-seed, multi-fold or independent generalization claim.
