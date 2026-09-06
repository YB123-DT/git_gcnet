# OSRAM capacity result: KV64 on CMU-MOSI

**Internal diagnostic only; not a formal paper result.**

## Capacity

`heads=4, key_dim=64, value_dim=64, output_dim=500`. The H4 OSRAM+mean control is unchanged and is inherited from `experiments/osram_complete_20260906/`.

## Single-checkpoint protocol

Each seed trained one cyclic mixed-rate model and selected one checkpoint using the mean Test weighted-F1 over all eight rates. The five-seed mean is **79.690%**, versus **80.048%** for H4 (delta **-0.359 points**).

| Seed | Selected epoch | KV64 mean | H4 mean | Delta |
|---:|---:|---:|---:|---:|
| 66 | 37 | 79.645 | 80.492 | -0.847 |
| 67 | 39 | 79.575 | 80.007 | -0.431 |
| 68 | 56 | 79.827 | 80.143 | -0.316 |
| 69 | 34 | 79.577 | 79.609 | -0.031 |
| 70 | 61 | 79.824 | 79.990 | -0.167 |

## Per-rate Test-oracle diagnostic

The following is the requested optimistic per-rate view extracted from the recorded histories; each rate can have a different epoch.

| Rate | H4 mean ± std | KV64 mean ± std | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.500 ± 0.249 | 87.351 ± 0.257 | -0.149 | 2/5 |
| 0.1 | 85.006 ± 0.941 | 85.084 ± 0.727 | +0.078 | 2/5 |
| 0.2 | 82.903 ± 0.858 | 82.531 ± 1.249 | -0.372 | 1/5 |
| 0.3 | 81.531 ± 0.342 | 80.983 ± 0.453 | -0.548 | 0/5 |
| 0.4 | 79.452 ± 1.688 | 78.904 ± 2.064 | -0.548 | 2/5 |
| 0.5 | 77.116 ± 1.013 | 76.830 ± 1.446 | -0.287 | 1/5 |
| 0.6 | 76.235 ± 0.588 | 76.181 ± 0.633 | -0.054 | 4/5 |
| 0.7 | 75.485 ± 2.441 | 75.116 ± 1.954 | -0.369 | 1/5 |

Per-rate 8-rate mean: **80.372%** vs H4 **80.654%** (delta **-0.281**). High-missing mean (0.5/0.6/0.7): **76.042%** vs H4 **76.279%** (delta **-0.237**).

## Interpretation

This isolates capacity, not a new propagation mechanism. The result should only be used to decide whether to retain this dimension budget for a later fair backbone comparison. All raw histories, configs, metrics, and diagnostics are retained under `raw/`.
