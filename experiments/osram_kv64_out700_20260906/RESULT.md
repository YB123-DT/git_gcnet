# OSRAM capacity result: H4-KV64-700 on CMU-MOSI

**Internal diagnostic only; not a formal paper result.**

## Capacity

`heads=4, key_dim=64, value_dim=64, output_dim=700`. Only capacity changed relative to H4.

## Single-checkpoint protocol

Five-seed mean: **80.082%**, versus H4 **80.048%** (delta **+0.034 points**).

| Seed | Selected epoch | H4-KV64-700 mean | H4 mean | Delta |
|---:|---:|---:|---:|---:|
| 66 | 72 | 80.321 | 80.492 | -0.171 |
| 67 | 46 | 80.202 | 80.007 | +0.195 |
| 68 | 54 | 79.932 | 80.143 | -0.211 |
| 69 | 55 | 80.492 | 79.609 | +0.884 |
| 70 | 53 | 79.463 | 79.990 | -0.527 |

## Per-rate Test-oracle diagnostic

| Rate | H4 mean ± std | H4-KV64-700 mean ± std | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.500 ± 0.249 | 87.402 ± 0.499 | -0.098 | 3/5 |
| 0.1 | 85.006 ± 0.941 | 85.092 ± 0.865 | +0.086 | 3/5 |
| 0.2 | 82.903 ± 0.858 | 82.760 ± 1.226 | -0.143 | 2/5 |
| 0.3 | 81.531 ± 0.342 | 81.342 ± 0.736 | -0.189 | 1/5 |
| 0.4 | 79.452 ± 1.688 | 78.832 ± 1.156 | -0.620 | 1/5 |
| 0.5 | 77.116 ± 1.013 | 77.133 ± 1.165 | +0.017 | 4/5 |
| 0.6 | 76.235 ± 0.588 | 76.645 ± 0.518 | +0.409 | 4/5 |
| 0.7 | 75.485 ± 2.441 | 75.003 ± 2.316 | -0.482 | 1/5 |

Per-rate 8-rate mean: **80.526%** vs H4 **80.654%** (delta **-0.128**). High-missing mean (0.5/0.6/0.7): **76.260%** vs H4 **76.279%** (delta **-0.019**).

## Interpretation

This isolates capacity, not a new propagation mechanism. Use it only to decide whether further capacity sweeps are justified.
