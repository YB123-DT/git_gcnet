# OSRAM capacity result: H8+V64+Output700 on CMU-MOSI

**Internal diagnostic only; not a formal paper result.**

## Capacity

`heads=8, key_dim=32, value_dim=64, output_dim=700`. Only capacity changed relative to H4.

## Single-checkpoint protocol

The five-seed mean is **80.124%**, versus H4 **80.048%** (delta **+0.076 points**).

| Seed | Selected epoch | H8+V64+Out700 mean | H4 mean | Delta |
|---:|---:|---:|---:|---:|
| 66 | 54 | 80.287 | 80.492 | -0.205 |
| 67 | 56 | 79.925 | 80.007 | -0.082 |
| 68 | 42 | 80.224 | 80.143 | +0.080 |
| 69 | 36 | 80.444 | 79.609 | +0.835 |
| 70 | 64 | 79.740 | 79.990 | -0.251 |

## Per-rate Test-oracle diagnostic

| Rate | H4 mean ± std | H8+V64+Out700 mean ± std | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.500 ± 0.249 | 87.356 ± 0.366 | -0.144 | 1/5 |
| 0.1 | 85.006 ± 0.941 | 85.474 ± 0.367 | +0.468 | 4/5 |
| 0.2 | 82.903 ± 0.858 | 82.742 ± 0.980 | -0.161 | 2/5 |
| 0.3 | 81.531 ± 0.342 | 81.284 ± 0.652 | -0.247 | 2/5 |
| 0.4 | 79.452 ± 1.688 | 79.436 ± 1.653 | -0.017 | 2/5 |
| 0.5 | 77.116 ± 1.013 | 77.012 ± 0.973 | -0.104 | 3/5 |
| 0.6 | 76.235 ± 0.588 | 76.979 ± 0.860 | +0.744 | 4/5 |
| 0.7 | 75.485 ± 2.441 | 75.332 ± 1.892 | -0.153 | 4/5 |

Per-rate 8-rate mean: **80.702%** vs H4 **80.654%** (delta **+0.048**). High-missing mean (0.5/0.6/0.7): **76.441%** vs H4 **76.279%** (delta **+0.162**).

## Interpretation

The combination does not provide a meaningful additional gain over H8+Output700; it is retained as a negative/near-neutral capacity control. Raw JSON histories and diagnostics are in `raw/`.
