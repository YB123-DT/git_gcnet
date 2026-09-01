# Complete-M3 Temporal Residual — CMU-MOSI miss0

The original 86.62% model was recovered from its saved checkpoints. Its
inference structure is three modality projectors followed by a 768→256→1
fusion head. Applying per-dimension train-split standardization exactly
reproduced seed 66 W-F1 (86.7136%).

The candidate adds one zero-initialized depthwise temporal residual at the
256-dimensional fusion hidden. Stage-1 projectors, features, splits, seeds, and
validation-W-F1 selection are preserved.

| Seed | Temporal residual | Inherited M3 | Delta |
|---:|---:|---:|---:|
| 66 | 85.94 | 86.71 | -0.77 |
| 67 | 84.52 | 85.91 | -1.40 |
| 68 | 86.32 | 86.31 | +0.01 |
| 69 | 85.07 | 86.68 | -1.62 |
| 70 | 86.05 | 87.47 | -1.41 |
| Mean | 85.58 | 86.62 | -1.04 |

Gate: **FAIL**. Positive seeds: 1/5. Collapsed seeds: 0/5. The candidate is
closed and is not extended to missing rates.
