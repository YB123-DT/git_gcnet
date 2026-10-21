# MOSEI seed66 Direct: Local / Base / Gap readout audit

INTERNAL DIAGNOSTIC ONLY

L = original Local adapter input plus original Local Skip; LB = L+Base; LG = L+active Gap; Full = L+Base+active Gap.
Learned Direct source weights remain unchanged. Only final Flat evidence inputs are zeroed; no training or new masks.
Same checkpoint within each rate, different existing W-F1 Test-best checkpoints across rates. No single-checkpoint deployment claim.

| Rate/scope | L ACC | LB ACC | LG ACC | Full ACC | L W-F1 | LB W-F1 | LG W-F1 | Full W-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 85.388 | 87.149 | 85.388 | 87.149 | 85.338 | 87.146 | 85.338 | 87.146 |
| 0.1 | 83.682 | 85.333 | 84.893 | 86.021 | 83.618 | 85.345 | 84.933 | 86.063 |
| 0.2 | 82.251 | 84.975 | 85.195 | 85.526 | 81.730 | 84.827 | 85.080 | 85.467 |
| 0.3 | 79.307 | 84.645 | 84.397 | 84.920 | 77.976 | 84.478 | 84.198 | 84.870 |
| 0.4 | 76.692 | 80.407 | 81.976 | 82.774 | 74.690 | 79.499 | 81.456 | 82.462 |
| 0.5 | 74.931 | 80.490 | 81.948 | 81.893 | 72.524 | 79.874 | 81.700 | 81.786 |
| 0.6 | 72.647 | 78.453 | 79.527 | 79.994 | 69.301 | 77.380 | 78.992 | 79.579 |
| 0.7 | 70.061 | 77.160 | 78.921 | 78.866 | 64.875 | 75.995 | 78.403 | 78.576 |
| mean | 78.120 | 82.327 | 82.781 | 83.393 | 76.256 | 81.818 | 82.513 | 83.244 |
| high | 72.546 | 78.701 | 80.132 | 80.251 | 68.900 | 77.750 | 79.699 | 79.980 |

## Paired increments (eight-rate equal means; counts sum repeated exposures)

| Comparison | Delta ACC(pp) | Delta W-F1(pp) | Corrections | Harms |
|---|---:|---:|---:|---:|
| L_to_LB | +4.207 | +5.562 | 1911 | 688 |
| L_to_LG | +4.661 | +6.256 | 2093 | 738 |
| LB_to_Full | +1.066 | +1.426 | 789 | 479 |
| LG_to_Full | +0.612 | +0.731 | 614 | 436 |
| L_to_Full | +5.273 | +6.987 | 2674 | 1141 |

CE/UA are not MOSEI regression/polarity metrics; ACC/W-F1 use nonzero labels and prediction>0. MSE includes zero labels.
Readout removal can be out-of-distribution. This is not a separately trained architecture ablation, nor proof of a writing/query bottleneck.
CHANGES.csv gives per-rate corrections/harms; positive gain means adding the named evidence helped under this intervention.
