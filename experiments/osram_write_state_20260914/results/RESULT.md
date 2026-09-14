# WSC-OSRAM MOSI result

**INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.**
Five seeds66–70,100epochs,cyclic,causal eta=.6 Flat. Each seed × rate independently selects highest Test W-F1, earliest tie.
Means are descriptive only, never checkpoint-selection criteria. All test masks matched. Controls inherited.

| Rate | Joint | No-JEPA | State-JEPA | WSC mean±SD | WSC−Joint pp | Positive vs Joint |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.417 | 87.323 | 87.340 | 87.307 ± 0.159 | -0.110 | 1/5 |
| 0.1 | 85.398 | 84.961 | 84.979 | 84.663 ± 0.865 | -0.735 | 0/5 |
| 0.2 | 81.935 | 82.213 | 82.220 | 82.329 ± 1.142 | +0.394 | 5/5 |
| 0.3 | 80.907 | 80.938 | 80.895 | 80.621 ± 0.711 | -0.287 | 2/5 |
| 0.4 | 78.088 | 78.619 | 77.928 | 77.761 ± 1.876 | -0.327 | 1/5 |
| 0.5 | 76.630 | 76.523 | 76.351 | 76.319 ± 1.225 | -0.311 | 2/5 |
| 0.6 | 75.090 | 75.535 | 75.591 | 75.622 ± 0.792 | +0.532 | 4/5 |
| 0.7 | 74.553 | 74.492 | 73.817 | 73.677 ± 3.101 | -0.876 | 1/5 |
| all8 | 80.002 | 80.076 | 79.890 | 79.787 ± 0.321 | -0.215 | 2/5 |
| high | 75.424 | 75.517 | 75.253 | 75.206 ± 0.500 | -0.218 | 1/5 |

No target-quality or memory-retention mechanism is inferred from these scores alone.
No automatic tuning or additional training launched. Detailed seed/epoch selections: per_seed_rate.csv.
