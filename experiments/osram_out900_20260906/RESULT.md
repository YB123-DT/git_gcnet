# OSRAM capacity result: H4-Output900 on CMU-MOSI

**Internal diagnostic only; not a formal paper result.**

## Capacity

`heads=4, key_dim=32, value_dim=32, output_dim=900`. Only capacity changed relative to H4.

## Single-checkpoint protocol

Five-seed mean: **79.909%**, versus H4 **80.048%** (delta **-0.140 points**).

| Seed | Selected epoch | H4-Output900 mean | H4 mean | Delta |
|---:|---:|---:|---:|---:|
| 66 | 60 | 79.775 | 80.492 | -0.717 |
| 67 | 63 | 79.372 | 80.007 | -0.635 |
| 68 | 56 | 80.260 | 80.143 | +0.117 |
| 69 | 95 | 80.579 | 79.609 | +0.971 |
| 70 | 32 | 79.557 | 79.990 | -0.433 |

## Per-rate Test-oracle diagnostic

| Rate | H4 mean ± std | H4-Output900 mean ± std | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.500 ± 0.249 | 87.366 ± 0.634 | -0.133 | 1/5 |
| 0.1 | 85.006 ± 0.941 | 85.346 ± 0.316 | +0.340 | 4/5 |
| 0.2 | 82.903 ± 0.858 | 82.781 ± 1.087 | -0.122 | 2/5 |
| 0.3 | 81.531 ± 0.342 | 81.174 ± 1.272 | -0.357 | 2/5 |
| 0.4 | 79.452 ± 1.688 | 79.157 ± 1.809 | -0.295 | 1/5 |
| 0.5 | 77.116 ± 1.013 | 77.174 ± 1.612 | +0.058 | 3/5 |
| 0.6 | 76.235 ± 0.588 | 76.799 ± 0.941 | +0.563 | 4/5 |
| 0.7 | 75.485 ± 2.441 | 75.291 ± 1.665 | -0.194 | 3/5 |

Per-rate 8-rate mean: **80.636%** vs H4 **80.654%** (delta **-0.018**). High-missing mean (0.5/0.6/0.7): **76.421%** vs H4 **76.279%** (delta **+0.142**).

## Interpretation

This isolates capacity, not a new propagation mechanism. Use it only to decide whether further capacity sweeps are justified.
