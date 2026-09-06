# OSRAM capacity result: V64 on CMU-MOSI

**Internal diagnostic only; not a formal paper result.**

## Capacity

`heads=4, key_dim=32, value_dim=64, output_dim=500`. The H4 OSRAM+mean control is unchanged and is inherited from `experiments/osram_complete_20260906/`.

## Single-checkpoint protocol

Each seed trained one cyclic mixed-rate model and selected one checkpoint using the mean Test weighted-F1 over all eight rates. The five-seed mean is **80.070%**, versus **80.048%** for H4 (delta **+0.022 points**).

| Seed | Selected epoch | V64 mean | H4 mean | Delta |
|---:|---:|---:|---:|---:|
| 66 | 36 | 80.117 | 80.492 | -0.375 |
| 67 | 55 | 80.124 | 80.007 | +0.118 |
| 68 | 53 | 80.257 | 80.143 | +0.114 |
| 69 | 44 | 80.162 | 79.609 | +0.553 |
| 70 | 22 | 79.689 | 79.990 | -0.302 |

## Per-rate Test-oracle diagnostic

The following is the requested optimistic per-rate view extracted from the recorded histories; each rate can have a different epoch.

| Rate | H4 mean ± std | V64 mean ± std | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.500 ± 0.249 | 87.629 ± 0.554 | +0.130 | 3/5 |
| 0.1 | 85.006 ± 0.941 | 85.117 ± 0.924 | +0.111 | 5/5 |
| 0.2 | 82.903 ± 0.858 | 82.971 ± 1.041 | +0.068 | 3/5 |
| 0.3 | 81.531 ± 0.342 | 81.698 ± 0.845 | +0.167 | 3/5 |
| 0.4 | 79.452 ± 1.688 | 79.274 ± 1.966 | -0.178 | 3/5 |
| 0.5 | 77.116 ± 1.013 | 77.000 ± 1.283 | -0.117 | 2/5 |
| 0.6 | 76.235 ± 0.588 | 76.764 ± 1.253 | +0.529 | 3/5 |
| 0.7 | 75.485 ± 2.441 | 75.265 ± 1.945 | -0.219 | 2/5 |

Per-rate 8-rate mean: **80.715%** vs H4 **80.654%** (delta **+0.061**). High-missing mean (0.5/0.6/0.7): **76.343%** vs H4 **76.279%** (delta **+0.064**).

## Interpretation

This isolates capacity, not a new propagation mechanism. The result should only be used to decide whether to retain this dimension budget for a later fair backbone comparison. All raw histories, configs, metrics, and diagnostics are retained under `raw/`.
