# OSRAM capacity result: H8 on CMU-MOSI

**Internal diagnostic only; not a formal paper result.**

## Capacity

`heads=8, key_dim=32, value_dim=32, output_dim=500`. The H4 OSRAM+mean control is unchanged and is inherited from `experiments/osram_complete_20260906/`.

## Single-checkpoint protocol

Each seed trained one cyclic mixed-rate model and selected one checkpoint using the mean Test weighted-F1 over all eight rates. The five-seed mean is **79.987%**, versus **80.048%** for H4 (delta **-0.061 points**).

| Seed | Selected epoch | H8 mean | H4 mean | Delta |
|---:|---:|---:|---:|---:|
| 66 | 47 | 80.416 | 80.492 | -0.076 |
| 67 | 21 | 79.423 | 80.007 | -0.583 |
| 68 | 34 | 79.557 | 80.143 | -0.586 |
| 69 | 42 | 80.201 | 79.609 | +0.593 |
| 70 | 42 | 80.339 | 79.990 | +0.349 |

## Per-rate Test-oracle diagnostic

The following is the requested optimistic per-rate view extracted from the recorded histories; each rate can have a different epoch.

| Rate | H4 mean ± std | H8 mean ± std | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.500 ± 0.249 | 87.187 ± 0.522 | -0.313 | 2/5 |
| 0.1 | 85.006 ± 0.941 | 85.278 ± 0.207 | +0.272 | 2/5 |
| 0.2 | 82.903 ± 0.858 | 82.847 ± 0.858 | -0.056 | 3/5 |
| 0.3 | 81.531 ± 0.342 | 81.118 ± 0.778 | -0.413 | 2/5 |
| 0.4 | 79.452 ± 1.688 | 79.166 ± 2.008 | -0.287 | 2/5 |
| 0.5 | 77.116 ± 1.013 | 77.365 ± 1.009 | +0.249 | 3/5 |
| 0.6 | 76.235 ± 0.588 | 76.761 ± 1.029 | +0.526 | 3/5 |
| 0.7 | 75.485 ± 2.441 | 75.203 ± 2.482 | -0.282 | 3/5 |

Per-rate 8-rate mean: **80.616%** vs H4 **80.654%** (delta **-0.038**). High-missing mean (0.5/0.6/0.7): **76.443%** vs H4 **76.279%** (delta **+0.164**).

## Interpretation

This isolates capacity, not a new propagation mechanism. The result should only be used to decide whether to retain this dimension budget for a later fair backbone comparison. All raw histories, configs, metrics, and diagnostics are retained under `raw/`.
