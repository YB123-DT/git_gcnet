# MOSI68 Signature cyclic training comparison

Source `4d5aa09d999b49d7194f72f6bf75c8ab75edbef1`. Eight independent frozen per-rate parents; internal TEST-oracle.

| Rate/scope | Method | ACC | WF1 | MSE | Δ WF1 vs Flat | Δ WF1 vs Fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | Flat | 79.1730 | 79.0742 | 1.4517 | 0.0000 | 0.0292 |
| mean | Fixed | 79.0206 | 79.0450 | 1.4570 | -0.0292 | 0.0000 |
| mean | Cyclic | 79.1159 | 79.1278 | 1.4447 | 0.0536 | 0.0829 |
| high | Flat | 72.7642 | 72.7704 | 1.8326 | 0.0000 | -0.3095 |
| high | Fixed | 72.9675 | 73.0799 | 1.8149 | 0.3095 | 0.0000 |
| high | Cyclic | 73.2215 | 73.3072 | 1.8008 | 0.5368 | 0.2273 |
| 0.0 | Flat | 87.0427 | 87.0458 | 1.0641 | 0.0000 | -0.2564 |
| 0.0 | Fixed | 87.3476 | 87.3022 | 0.9934 | 0.2564 | 0.0000 |
| 0.0 | Cyclic | 87.5000 | 87.4178 | 0.9444 | 0.3720 | 0.1156 |
| 0.1 | Flat | 84.1463 | 83.9815 | 1.0736 | 0.0000 | -0.5573 |
| 0.1 | Fixed | 84.6037 | 84.5388 | 1.1037 | 0.5573 | 0.0000 |
| 0.1 | Cyclic | 84.4512 | 84.4003 | 1.1141 | 0.4188 | -0.1386 |
| 0.2 | Flat | 82.3171 | 82.1035 | 1.3240 | 0.0000 | -0.0714 |
| 0.2 | Fixed | 82.3171 | 82.1749 | 1.3308 | 0.0714 | 0.0000 |
| 0.2 | Cyclic | 82.3171 | 82.1881 | 1.3369 | 0.0846 | 0.0131 |
| 0.3 | Flat | 82.0122 | 81.8252 | 1.2321 | 0.0000 | 0.8091 |
| 0.3 | Fixed | 80.9451 | 81.0161 | 1.3363 | -0.8091 | 0.0000 |
| 0.3 | Cyclic | 81.2500 | 81.2716 | 1.2952 | -0.5536 | 0.2555 |
| 0.4 | Flat | 79.5732 | 79.3264 | 1.4220 | 0.0000 | 1.2385 |
| 0.4 | Fixed | 78.0488 | 78.0879 | 1.4470 | -1.2385 | 0.0000 |
| 0.4 | Cyclic | 77.7439 | 77.8234 | 1.4648 | -1.5030 | -0.2645 |
| 0.5 | Flat | 76.0671 | 75.8279 | 1.5828 | 0.0000 | 0.0992 |
| 0.5 | Fixed | 75.6098 | 75.7287 | 1.6152 | -0.0992 | 0.0000 |
| 0.5 | Cyclic | 75.4573 | 75.5411 | 1.5801 | -0.2868 | -0.1876 |
| 0.6 | Flat | 73.1707 | 73.2580 | 1.8879 | 0.0000 | -2.2575 |
| 0.6 | Fixed | 75.4573 | 75.5155 | 1.8086 | 2.2575 | 0.0000 |
| 0.6 | Cyclic | 75.1524 | 75.2522 | 1.8576 | 1.9942 | -0.2634 |
| 0.7 | Flat | 69.0549 | 69.2253 | 2.0270 | 0.0000 | 1.2297 |
| 0.7 | Fixed | 67.8354 | 67.9956 | 2.0209 | -1.2297 | 0.0000 |
| 0.7 | Cyclic | 69.0549 | 69.1283 | 1.9646 | -0.0970 | 1.1327 |

## Completed protocol and limits

Server biggpu,cmerc-repro; source4d5aa09d999b49d7194f72f6bf75c8ab75edbef1. All8runs completed150epochs/300updates with the original cyclic schedule,all8actual training rates and identical training mask traces. Evaluation stayed at the respective current rate. Same protected per-rate Flat parents,test masks,Signature code,seed and optimizer convention as the fixed68 comparison. The historical fixed initial-module digest was not saved,so this is not fabricated as retrospective bytewise initialization proof. Per-rate Test-oracle and one seed remain internal exploration. Cyclic mean W-F1 improves only0.053639pp overFlat while ACC decreases0.057165pp; no robust performance gain is claimed.
