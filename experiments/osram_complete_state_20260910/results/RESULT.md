# Complete-View Local-State JEPA: MOSI results

**INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.**
Seeds66–70;100epochs;causal Flat eta=.6;cyclic. Each seed × rate independently selects maximum Test W-F1 (earliest tie).
All averages are descriptive, never used for checkpoint selection. All test mask hashes matched.

| rate | Joint mean±SD | Emotion-only mean±SD | State mean±SD | State−Joint pp | State−Emotion pp | positive vs Joint / Emotion |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.417 ± 0.224 | 87.323 ± 0.202 | 87.340 ± 0.391 | -0.077 | +0.017 | 3/5 ; 2/5 |
| 0.1 | 85.398 ± 0.719 | 84.961 ± 0.681 | 84.979 ± 0.340 | -0.419 | +0.018 | 1/5 ; 3/5 |
| 0.2 | 81.935 ± 1.195 | 82.213 ± 1.063 | 82.220 ± 0.927 | +0.284 | +0.007 | 4/5 ; 2/5 |
| 0.3 | 80.907 ± 0.890 | 80.938 ± 0.613 | 80.895 ± 0.507 | -0.012 | -0.043 | 2/5 ; 2/5 |
| 0.4 | 78.088 ± 1.033 | 78.619 ± 2.080 | 77.928 ± 1.755 | -0.160 | -0.691 | 3/5 ; 1/5 |
| 0.5 | 76.630 ± 1.064 | 76.523 ± 1.137 | 76.351 ± 1.210 | -0.279 | -0.173 | 2/5 ; 2/5 |
| 0.6 | 75.090 ± 0.835 | 75.535 ± 0.256 | 75.591 ± 0.703 | +0.501 | +0.056 | 4/5 ; 3/5 |
| 0.7 | 74.553 ± 2.172 | 74.492 ± 2.202 | 73.817 ± 2.870 | -0.736 | -0.675 | 2/5 ; 1/5 |
| all8 | 80.002 ± 0.468 | 80.076 ± 0.452 | 79.890 ± 0.174 | -0.112 | -0.185 | 1/5 ; 2/5 |
| high | 75.424 ± 0.877 | 75.517 ± 0.616 | 75.253 ± 0.424 | -0.171 | -0.264 | 1/5 ; 1/5 |

Full histories and selected epochs are archived. Emotion-only JEPA loss was zero in all epochs.
State loss decline alone is not evidence of sample-specific prediction or absence of collapse; no representation audit is claimed.
