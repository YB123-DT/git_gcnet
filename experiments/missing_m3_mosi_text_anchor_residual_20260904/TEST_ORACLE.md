# Per-rate Test-oracle Diagnostic (Seed 66)

This diagnostic deliberately evaluates the test set after every epoch and
allows each missing rate to select its own best test epoch. It therefore uses
up to eight different checkpoints and is **not a valid benchmark result**.

| Miss | Anchor epoch | Anchor W-F1 | Slot epoch | Slot W-F1 | Delta |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 63 | 86.73 | 50 | 86.41 | +0.33 |
| 0.1 | 63 | 85.53 | 46 | 85.58 | -0.05 |
| 0.2 | 63 | 82.81 | 45 | 82.21 | +0.60 |
| 0.3 | 63 | 80.98 | 50 | 80.86 | +0.11 |
| 0.4 | 63 | 80.21 | 43 | 79.31 | +0.90 |
| 0.5 | 69 | 77.38 | 50 | 77.37 | +0.01 |
| 0.6 | 69 | 76.35 | 50 | 74.03 | +2.32 |
| 0.7 | 69 | 76.06 | 43 | 73.63 | +2.43 |
| Mean of independent rate oracles | -- | **80.76** | -- | **79.93** | **+0.83** |

Seven of eight rate-specific test maxima are positive. In practice the Anchor
oracle uses only two checkpoints: epoch 63 for rates 0.0--0.4 and epoch 69 for
rates 0.5--0.7. This demonstrates potential capacity, especially at high
missingness, but it does not rescue the formal result: the valid
validation-selected five-seed experiment remains `-1.08` points.

The next scientifically valid question would be rate-specific **validation**
selection with test evaluated only after selection. This file cannot be used
to choose the residual cap or report final performance.

