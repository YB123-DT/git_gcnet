# GCNet Session-5 all missing rates, 10 trials

IEMOCAP-Six Session 5 results from ten complete training trials (seeds 66--75)
per missing rate. The GCNet path was used with JEPA disabled. Results still use
the inherited test-as-validation selection logic and therefore are diagnostic,
not the final leakage-free benchmark.

| Requested missing | W-F1 mean (%) | Sample std | 95% CI half-width | Min--max |
|---:|---:|---:|---:|---:|
| 0.0 | 63.37 | 2.05 | 1.47 | 58.09--65.02 |
| 0.1 | 64.49 | 1.11 | 0.80 | 62.73--65.70 |
| 0.2 | 63.23 | 1.54 | 1.10 | 61.24--66.36 |
| 0.3 | 62.43 | 1.82 | 1.30 | 60.36--66.06 |
| 0.4 | 61.78 | 1.95 | 1.39 | 58.79--65.49 |
| 0.5 | 60.68 | 1.46 | 1.04 | 58.19--62.44 |
| 0.6 | 59.96 | 2.20 | 1.57 | 56.85--64.21 |
| 0.7 | 60.63 | 2.44 | 1.75 | 56.17--64.56 |

The average over all eight rates is 62.07%. Ten-trial averaging removes the
extreme single-run zigzags, but the mean curve is not strictly monotonic: 0.1
is above 0.0 and requested 0.7 is above 0.6. Both reversals have overlapping
confidence intervals. Moreover, requested 0.7 is implemented as exactly 2/3
missing because one modality must remain, so it is not more severe than 0.6 by
the nominal 0.1 increment.

Machine-readable per-seed records are in `summary_all_rates.json`.
