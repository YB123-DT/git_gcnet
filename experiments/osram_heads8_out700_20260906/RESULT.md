# OSRAM capacity result: H8+Out700 on CMU-MOSI

**Internal diagnostic only; not a formal paper result.**

## Capacity

`heads=8, key_dim=32, value_dim=32, output_dim=700`. The H4 OSRAM+mean control is unchanged and is inherited from `experiments/osram_complete_20260906/`.

## Single-checkpoint protocol

Each seed trained one cyclic mixed-rate model and selected one checkpoint using the mean Test weighted-F1 over all eight rates. The five-seed mean is **80.267%**, versus **80.048%** for H4 (delta **+0.219 points**).

| Seed | Selected epoch | H8+Out700 mean | H4 mean | Delta |
|---:|---:|---:|---:|---:|
| 66 | 61 | 80.229 | 80.492 | -0.264 |
| 67 | 39 | 79.850 | 80.007 | -0.156 |
| 68 | 46 | 80.880 | 80.143 | +0.737 |
| 69 | 38 | 80.498 | 79.609 | +0.889 |
| 70 | 38 | 79.877 | 79.990 | -0.114 |

## Per-rate Test-oracle diagnostic

The following is the requested optimistic per-rate view extracted from the recorded histories; each rate can have a different epoch.

| Rate | H4 mean ± std | H8+Out700 mean ± std | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.500 ± 0.249 | 87.639 ± 0.669 | +0.140 | 3/5 |
| 0.1 | 85.006 ± 0.941 | 85.365 ± 0.357 | +0.359 | 3/5 |
| 0.2 | 82.903 ± 0.858 | 82.693 ± 1.090 | -0.210 | 3/5 |
| 0.3 | 81.531 ± 0.342 | 81.439 ± 0.651 | -0.093 | 3/5 |
| 0.4 | 79.452 ± 1.688 | 79.350 ± 1.521 | -0.102 | 2/5 |
| 0.5 | 77.116 ± 1.013 | 77.300 ± 0.825 | +0.184 | 3/5 |
| 0.6 | 76.235 ± 0.588 | 76.804 ± 0.535 | +0.568 | 5/5 |
| 0.7 | 75.485 ± 2.441 | 75.654 ± 1.940 | +0.170 | 3/5 |

Per-rate 8-rate mean: **80.781%** vs H4 **80.654%** (delta **+0.127**). High-missing mean (0.5/0.6/0.7): **76.586%** vs H4 **76.279%** (delta **+0.307**).

## Interpretation

This isolates capacity, not a new propagation mechanism. The result should only be used to decide whether to retain this dimension budget for a later fair backbone comparison. All raw histories, configs, metrics, and diagnostics are retained under `raw/`.
