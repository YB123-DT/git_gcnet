# MOSI backbone LR1e-5 seed66 paired screen

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY

150epochs from scratch. Backbone constant1e-5; Router3e-4 first100,3e-5 last50.
Original GCNet missing; identical initialization/cohorts and training masks; no new loss or architecture.
Per-rate Test-best W-F1; ACC uses same predictions. Eight-rate mean is not one deployment checkpoint.

| Model | ACC | W-F1 | High ACC | High W-F1 |
|---|---:|---:|---:|---:|
| flat | 76.5816 | 76.6128 | 71.2398 | 71.3512 |
| joint | 76.3148 | 76.3643 | 70.7825 | 70.9540 |

| Rate | Flat ACC | Router ACC | Flat W-F1 | Router W-F1 |
|---|---:|---:|---:|---:|
| 0.0 | 83.6890 | 83.8415 | 83.6405 | 83.7466 |
| 0.1 | 81.7073 | 81.4024 | 81.6583 | 81.3836 |
| 0.2 | 78.5061 | 78.6585 | 78.5400 | 78.6877 |
| 0.3 | 78.3537 | 78.0488 | 78.4310 | 77.9633 |
| 0.4 | 76.6768 | 76.2195 | 76.5786 | 76.2714 |
| 0.5 | 74.6951 | 74.3902 | 74.8464 | 74.5331 |
| 0.6 | 70.4268 | 69.8171 | 70.5413 | 69.9953 |
| 0.7 | 68.5976 | 68.1402 | 68.6660 | 68.3336 |
