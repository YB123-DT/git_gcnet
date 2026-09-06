# OSRAM capacity result: Out700 on CMU-MOSI

**Internal diagnostic only; not a formal paper result.**

## Capacity

`heads=4, key_dim=32, value_dim=32, output_dim=700`. The H4 OSRAM+mean control is unchanged and is inherited from `experiments/osram_complete_20260906/`.

## Single-checkpoint protocol

Each seed trained one cyclic mixed-rate model and selected one checkpoint using the mean Test weighted-F1 over all eight rates. The five-seed mean is **79.823%**, versus **80.048%** for H4 (delta **-0.225 points**).

| Seed | Selected epoch | Out700 mean | H4 mean | Delta |
|---:|---:|---:|---:|---:|
| 66 | 36 | 80.542 | 80.492 | +0.050 |
| 67 | 45 | 79.079 | 80.007 | -0.928 |
| 68 | 43 | 80.223 | 80.143 | +0.080 |
| 69 | 44 | 79.499 | 79.609 | -0.110 |
| 70 | 46 | 79.773 | 79.990 | -0.218 |

## Per-rate Test-oracle diagnostic

The following is the requested optimistic per-rate view extracted from the recorded histories; each rate can have a different epoch.

| Rate | H4 mean ± std | Out700 mean ± std | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.500 ± 0.249 | 87.512 ± 0.447 | +0.012 | 3/5 |
| 0.1 | 85.006 ± 0.941 | 85.220 ± 1.095 | +0.214 | 3/5 |
| 0.2 | 82.903 ± 0.858 | 82.928 ± 1.279 | +0.025 | 3/5 |
| 0.3 | 81.531 ± 0.342 | 81.104 ± 1.226 | -0.427 | 2/5 |
| 0.4 | 79.452 ± 1.688 | 79.099 ± 1.806 | -0.354 | 2/5 |
| 0.5 | 77.116 ± 1.013 | 76.779 ± 1.294 | -0.337 | 2/5 |
| 0.6 | 76.235 ± 0.588 | 76.781 ± 0.796 | +0.546 | 4/5 |
| 0.7 | 75.485 ± 2.441 | 75.026 ± 2.117 | -0.459 | 2/5 |

Per-rate 8-rate mean: **80.556%** vs H4 **80.654%** (delta **-0.097**). High-missing mean (0.5/0.6/0.7): **76.195%** vs H4 **76.279%** (delta **-0.083**).

## Interpretation

This isolates capacity, not a new propagation mechanism. The result should only be used to decide whether to retain this dimension budget for a later fair backbone comparison. All raw histories, configs, metrics, and diagnostics are retained under `raw/`.
