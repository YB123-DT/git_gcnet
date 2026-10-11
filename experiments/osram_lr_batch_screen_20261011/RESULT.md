# Learning-rate and batch-size screen

**INTERNAL DIAGNOSTIC ONLY — per-rate Test-oracle checkpoint selection.**

## MOSI seed 66, 150 epochs, constant learning rate 1e-4

| Model | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 8-rate mean | High (.5/.6/.7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat | 86.604 | 84.634 | 81.047 | 79.673 | 79.394 | 76.564 | 74.193 | 74.021 | 79.516 | 74.926 |
| Nested | 86.293 | 84.326 | 81.485 | 81.050 | 79.730 | 77.675 | 73.877 | 73.249 | 79.711 | 74.934 |
| Nested - Flat | -0.311 | -0.308 | +0.437 | +1.377 | +0.336 | +1.112 | -0.316 | -0.772 | +0.194 | +0.008 |

Selected epochs were Flat `[61, 67, 68, 103, 47, 38, 36, 86]` and Nested
`[82, 67, 55, 55, 42, 38, 61, 80]` for missing rates 0.0 through 0.7.

The three MOSEI jobs were still running when this MOSI result was recorded.
