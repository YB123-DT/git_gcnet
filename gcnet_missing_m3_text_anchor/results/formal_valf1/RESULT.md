# CMU-MOSI miss0 five-seed result

Checkpoint selection used validation W-F1, matching the inherited `m3_mosi`
control. The test set was evaluated once after selection.

| Seed | Text-anchor W-F1 | Control W-F1 | Delta |
|---:|---:|---:|---:|
| 66 | 85.78 | 86.71 | -0.93 |
| 67 | 85.23 | 85.91 | -0.69 |
| 68 | 85.05 | 86.31 | -1.25 |
| 69 | 84.84 | 86.68 | -1.84 |
| 70 | 84.54 | 87.47 | -2.93 |
| Mean | 85.09 | 86.62 | -1.53 |

Gate: **FAIL**. Positive seeds: 0/5. Collapsed seeds: 0/5. The
Text-Anchored residual backbone is closed and is not extended to missing rates.
