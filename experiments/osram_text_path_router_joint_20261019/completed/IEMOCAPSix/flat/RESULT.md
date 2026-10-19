# IEMOCAPSix / flat one-stage run

INTERNAL SINGLE-SEED EXPLORATION

Source: `79619b8a2fd9f6c7e82bc67546aa8f9ac82105db`. One validation-selected checkpoint; selected epoch: 75.
Fresh initialization, original GCNet random missing; no persistent augmentation.
For Joint, full denotes its own jointly trained full candidate, NOT a separate Flat control.

| Rate | Full ACC | Final ACC | Full UA | Final UA | Final W-F1 | Final CE |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 61.4911 | 61.4911 | 62.4990 | 62.4990 | 61.8149 | 1.0925 |
| 0.1 | 60.2588 | 60.2588 | 61.2197 | 61.2197 | 60.5808 | 1.0840 |
| 0.2 | 60.2588 | 60.2588 | 61.1726 | 61.1726 | 60.7129 | 1.1315 |
| 0.3 | 58.4720 | 58.4720 | 59.5840 | 59.5840 | 58.9987 | 1.0856 |
| 0.4 | 58.9649 | 58.9649 | 59.5585 | 59.5585 | 59.3689 | 1.1234 |
| 0.5 | 59.6426 | 59.6426 | 59.9843 | 59.9843 | 60.0365 | 1.0818 |
| 0.6 | 56.5619 | 56.5619 | 57.8003 | 57.8003 | 56.8506 | 1.1100 |
| 0.7 | 55.0832 | 55.0832 | 55.5753 | 55.5753 | 55.7067 | 1.2172 |

Compare the separate matched Flat job before interpreting any benefit.
No multi-seed, multi-fold or independent generalization claim.
