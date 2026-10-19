# IEMOCAPSix / joint one-stage run

INTERNAL SINGLE-SEED EXPLORATION

Source: `79619b8a2fd9f6c7e82bc67546aa8f9ac82105db`. One validation-selected checkpoint; selected epoch: 94.
Fresh initialization, original GCNet random missing; no persistent augmentation.
For Joint, full denotes its own jointly trained full candidate, NOT a separate Flat control.

| Rate | Full ACC | Final ACC | Full UA | Final UA | Final W-F1 | Final CE |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 62.9082 | 62.9082 | 62.4653 | 62.4653 | 63.0587 | 1.1207 |
| 0.1 | 62.2921 | 62.4769 | 61.9723 | 62.1152 | 62.6239 | 1.1094 |
| 0.2 | 60.8749 | 60.8133 | 60.7711 | 60.7274 | 61.1464 | 1.1484 |
| 0.3 | 60.9982 | 60.8749 | 60.2536 | 60.1481 | 61.1806 | 1.0867 |
| 0.4 | 58.1023 | 58.4104 | 58.0443 | 58.3894 | 58.7027 | 1.1334 |
| 0.5 | 58.8417 | 59.2113 | 59.4071 | 59.7773 | 59.2732 | 1.1073 |
| 0.6 | 57.4245 | 57.4245 | 57.9911 | 57.9785 | 57.5303 | 1.1284 |
| 0.7 | 55.6377 | 56.0074 | 55.1905 | 55.4772 | 56.2493 | 1.2279 |

Compare the separate matched Flat job before interpreting any benefit.
No multi-seed, multi-fold or independent generalization claim.
