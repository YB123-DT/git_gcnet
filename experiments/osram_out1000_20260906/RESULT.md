# OSRAM capacity result: H4-Output1000 on CMU-MOSI

**Internal diagnostic only; not a formal paper result.**

## Capacity

`heads=4, key_dim=32, value_dim=32, output_dim=1000`. Only capacity changed relative to H4.

## Single-checkpoint protocol

Five-seed mean: **79.900%**, versus H4 **80.048%** (delta **-0.149 points**).

| Seed | Selected epoch | H4-Output1000 mean | H4 mean | Delta |
|---:|---:|---:|---:|---:|
| 66 | 36 | 79.917 | 80.492 | -0.575 |
| 67 | 39 | 80.371 | 80.007 | +0.364 |
| 68 | 32 | 79.973 | 80.143 | -0.170 |
| 69 | 65 | 79.500 | 79.609 | -0.109 |
| 70 | 51 | 79.737 | 79.990 | -0.254 |

## Per-rate Test-oracle diagnostic

| Rate | H4 mean ± std | H4-Output1000 mean ± std | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.500 ± 0.249 | 87.549 ± 0.706 | +0.050 | 3/5 |
| 0.1 | 85.006 ± 0.941 | 85.099 ± 0.865 | +0.092 | 3/5 |
| 0.2 | 82.903 ± 0.858 | 83.273 ± 0.708 | +0.370 | 5/5 |
| 0.3 | 81.531 ± 0.342 | 81.536 ± 0.533 | +0.005 | 2/5 |
| 0.4 | 79.452 ± 1.688 | 79.203 ± 1.452 | -0.250 | 2/5 |
| 0.5 | 77.116 ± 1.013 | 76.699 ± 0.696 | -0.418 | 1/5 |
| 0.6 | 76.235 ± 0.588 | 76.276 ± 0.693 | +0.041 | 3/5 |
| 0.7 | 75.485 ± 2.441 | 75.065 ± 2.368 | -0.420 | 2/5 |

Per-rate 8-rate mean: **80.587%** vs H4 **80.654%** (delta **-0.066**). High-missing mean (0.5/0.6/0.7): **76.013%** vs H4 **76.279%** (delta **-0.266**).

## Interpretation

This isolates capacity, not a new propagation mechanism. Use it only to decide whether further capacity sweeps are justified.
