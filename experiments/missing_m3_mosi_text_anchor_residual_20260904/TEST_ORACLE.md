# Per-rate Test-oracle Diagnostic (Five Seeds)

This diagnostic evaluates the test set after every epoch and lets every
missing rate select its own best test epoch. A seed can therefore use up to
eight different checkpoints. The results quantify available model capacity,
but they contain direct test-selection leakage and are **not valid benchmark
results**.

## Text-anchor residual results

Each cell is `W-F1 (best test epoch)`.

| Seed | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Rate mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 86.73 (63) | 85.53 (63) | 82.81 (63) | 80.98 (63) | 80.21 (63) | 77.38 (69) | 76.35 (69) | 76.06 (69) | 80.76 |
| 67 | 87.15 (61) | 86.24 (65) | 84.23 (61) | 81.91 (61) | 76.52 (64) | 76.65 (46) | 76.98 (66) | 75.13 (64) | 80.60 |
| 68 | 86.73 (53) | 84.77 (53) | 82.63 (55) | 80.20 (41) | 79.29 (54) | 77.40 (52) | 75.95 (82) | 70.74 (54) | 79.71 |
| 69 | 85.87 (60) | 84.06 (71) | 81.13 (60) | 81.47 (46) | 79.47 (71) | 76.63 (71) | 76.21 (73) | 76.10 (71) | 80.12 |
| 70 | 87.08 (78) | 85.38 (59) | 81.25 (71) | 80.83 (47) | 79.98 (53) | 76.52 (95) | 76.44 (53) | 76.06 (53) | 80.44 |
| Mean | **86.71** | **85.20** | **82.41** | **81.08** | **79.09** | **76.92** | **76.39** | **74.82** | **80.33** |
| SD | 0.51 | 0.82 | 1.28 | 0.65 | 1.49 | 0.44 | 0.38 | 2.32 | 0.42 |

All five histories contain exactly 100 epochs, and all 4,000 rate-by-epoch
W-F1 values are finite. The five seed-level rate means are `80.76`, `80.60`,
`79.71`, `80.12`, and `80.44`.

Compared with the valid validation-selected experiment, the per-rate Test
oracle raises the mean from `77.46` to `80.33` (`+2.87`). This gap is expected
selection bias rather than evidence of a deployable improvement. The high
missing-rate result is also unstable: miss 0.7 ranges from `70.74` to `76.10`.

## Paired Slot control available for seed 66 only

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
| Mean | -- | **80.76** | -- | **79.93** | **+0.83** |

Seed 66 suggests latent capacity at high missingness, but no five-seed paired
Slot Test-oracle control exists. It must not be generalized into a five-seed
causal claim. The scientifically valid text-anchor result remains the
validation-selected five-seed result: all eight rate means decreased and the
overall delta was `-1.08` points.

Raw histories and final prediction artifacts are under `test_oracle/seed_*`.
The per-rate maxima above come from the histories; the final NPZ files belong
to the trainer's single mean-oracle checkpoint and do not represent all eight
independently selected epochs.
