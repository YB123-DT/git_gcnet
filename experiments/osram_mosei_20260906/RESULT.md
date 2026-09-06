# OSRAM CMU-MOSEI result

**Internal diagnostic only; not a formal paper result.**

The table reports five-seed mean ± sample standard deviation of Test weighted
F1 (percentage points). Each rate independently uses its best Test epoch from
the recorded 100-epoch history; it is not one common checkpoint across rates.

| Missing rate | W-F1 mean ± SD (%) |
|---:|---:|
| 0.0 | 87.38 ± 0.12 |
| 0.1 | 86.45 ± 0.17 |
| 0.2 | 85.46 ± 0.53 |
| 0.3 | 84.63 ± 0.25 |
| 0.4 | 83.82 ± 0.43 |
| 0.5 | 82.92 ± 0.21 |
| 0.6 | 82.20 ± 0.49 |
| 0.7 | 81.20 ± 0.88 |

The eight-rate mean is **84.26%**. The high-missing mean over 0.5/0.6/0.7 is
**82.11%**.

The per-seed eight-rate means are:

| Seed | Per-rate Test-oracle mean (%) |
|---:|---:|
| 66 | 84.45 |
| 67 | 84.54 |
| 68 | 84.21 |
| 69 | 83.96 |
| 70 | 84.14 |

The actual saved mean-oracle checkpoints were selected at epochs 35, 19, 32,
31, and 26 for seeds 66--70 respectively. Those single-checkpoint metrics are
kept in raw/seed_*/metrics.json; the table above is the independent per-rate
diagnostic extraction.

## OSRAM diagnostics

Across the five final-batch diagnostic records, the mean base-context norm is
35.22 and the mean memory Frobenius norm is 22.41. The observed-address
residual is active:

| Target slot | rho | eta | Gap/Base ratio |
|---|---:|---:|---:|
| Audio | 0.860 | 0.219 | 0.464 |
| Text | 0.866 | 0.211 | 0.456 |
| Visual | 0.858 | 0.221 | 0.506 |

Gap contexts are therefore nonzero; the run did not collapse to Local + Base
only, and no non-finite prediction or hidden-state value was observed.

## Decision

This run is sufficient to retain OSRAM as a candidate for the next controlled
comparison, but it does not establish a gain over GCNet because a same-protocol
CMU-MOSEI GCNet control was not included in this batch. Do not turn the
Test-oracle values into a paper claim.
