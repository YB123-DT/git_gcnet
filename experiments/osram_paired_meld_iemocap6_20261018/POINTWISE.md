# Pointwise Flat vs paired task-only

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY. All comparisons are seed66 vs seed66.
IEMOCAP6 uses equal averaging over its five complete folds at EACH condition; no averaging across missing conditions in these tables.
A/T/V name observed modalities, T=Text. Paired loss weights=.5/.5, extra drop=.2, no InfoNCE loss.

## MELD — Persistent observed sets

| Condition | Flat ACC | Paired ACC | Delta ACC (pp) | Flat UA | Paired UA | Delta UA (pp) | Flat WF1 | Paired WF1 | Delta WF1 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 48.084 | 48.199 | +0.115 | 20.791 | 21.137 | +0.346 | 40.104 | 40.577 | +0.473 |
| T | 66.322 | 66.015 | -0.307 | 48.991 | 48.719 | -0.272 | 65.512 | 65.152 | -0.360 |
| V | 48.084 | 48.123 | +0.038 | 14.323 | 14.334 | +0.011 | 31.393 | 31.403 | +0.009 |
| AT | 67.011 | 66.858 | -0.153 | 46.868 | 46.815 | -0.053 | 65.648 | 65.544 | -0.104 |
| AV | 49.119 | 50.268 | +1.149 | 17.715 | 18.640 | +0.925 | 37.972 | 39.294 | +1.322 |
| TV | 66.475 | 66.130 | -0.345 | 46.734 | 47.526 | +0.792 | 65.215 | 65.193 | -0.023 |
| ATV | 67.510 | 67.548 | +0.038 | 44.330 | 44.216 | -0.114 | 65.691 | 65.689 | -0.002 |

## MELD — Random missing rates

| Condition | Flat ACC | Paired ACC | Delta ACC (pp) | Flat UA | Paired UA | Delta UA (pp) | Flat WF1 | Paired WF1 | Delta WF1 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 67.510 | 67.548 | +0.038 | 44.330 | 44.216 | -0.114 | 65.691 | 65.689 | -0.002 |
| 0.1 | 65.709 | 65.747 | +0.038 | 45.581 | 45.987 | +0.406 | 64.245 | 64.534 | +0.289 |
| 0.2 | 64.100 | 63.946 | -0.153 | 40.928 | 42.910 | +1.983 | 61.880 | 62.188 | +0.309 |
| 0.3 | 62.261 | 62.414 | +0.153 | 40.207 | 40.915 | +0.708 | 59.996 | 60.310 | +0.314 |
| 0.4 | 60.920 | 60.728 | -0.192 | 38.847 | 40.412 | +1.565 | 58.209 | 58.698 | +0.490 |
| 0.5 | 59.770 | 59.502 | -0.268 | 33.188 | 33.207 | +0.019 | 55.672 | 55.535 | -0.137 |
| 0.6 | 57.433 | 56.705 | -0.728 | 32.255 | 32.448 | +0.193 | 53.141 | 52.748 | -0.393 |
| 0.7 | 56.590 | 56.015 | -0.575 | 32.026 | 31.679 | -0.347 | 52.285 | 51.715 | -0.570 |

## IEMOCAPSix — Persistent observed sets

| Condition | Flat ACC | Paired ACC | Delta ACC (pp) | Flat UA | Paired UA | Delta UA (pp) | Flat WF1 | Paired WF1 | Delta WF1 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 44.489 | 45.287 | +0.798 | 39.222 | 38.472 | -0.750 | 41.901 | 42.970 | +1.069 |
| T | 51.171 | 52.601 | +1.431 | 48.335 | 50.076 | +1.740 | 50.658 | 51.734 | +1.076 |
| V | 34.195 | 36.733 | +2.538 | 31.047 | 33.544 | +2.496 | 30.978 | 34.433 | +3.455 |
| AT | 61.185 | 61.524 | +0.340 | 60.219 | 59.543 | -0.676 | 60.813 | 60.734 | -0.078 |
| AV | 48.563 | 47.770 | -0.794 | 46.397 | 44.503 | -1.895 | 47.139 | 47.076 | -0.064 |
| TV | 56.597 | 57.112 | +0.515 | 53.995 | 56.085 | +2.090 | 56.112 | 56.882 | +0.770 |
| ATV | 64.902 | 64.282 | -0.620 | 62.363 | 62.106 | -0.257 | 64.385 | 64.165 | -0.220 |

## IEMOCAPSix — Random missing rates

| Condition | Flat ACC | Paired ACC | Delta ACC (pp) | Flat UA | Paired UA | Delta UA (pp) | Flat WF1 | Paired WF1 | Delta WF1 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 64.902 | 64.282 | -0.620 | 62.363 | 62.106 | -0.257 | 64.385 | 64.165 | -0.220 |
| 0.1 | 64.293 | 63.972 | -0.321 | 61.665 | 61.574 | -0.091 | 63.760 | 63.677 | -0.083 |
| 0.2 | 63.011 | 63.138 | +0.127 | 61.062 | 61.758 | +0.696 | 62.526 | 63.115 | +0.589 |
| 0.3 | 62.969 | 62.613 | -0.356 | 61.411 | 60.740 | -0.671 | 62.614 | 62.157 | -0.457 |
| 0.4 | 61.828 | 61.574 | -0.253 | 59.520 | 60.493 | +0.973 | 61.054 | 61.570 | +0.516 |
| 0.5 | 61.423 | 60.569 | -0.854 | 59.859 | 58.722 | -1.137 | 60.729 | 60.418 | -0.311 |
| 0.6 | 59.740 | 59.251 | -0.489 | 57.844 | 57.557 | -0.287 | 59.045 | 58.736 | -0.310 |
| 0.7 | 57.925 | 58.469 | +0.544 | 55.835 | 55.241 | -0.594 | 56.944 | 57.918 | +0.975 |

## Reading individual points

- MELD Persistent AV (Text absent) improves ACC +1.149 / UA +0.925 / WF1 +1.322 pp. T and AT decline; TV increases UA but decreases ACC. No blanket robustness gain.
- IEMOCAP6 Persistent V improves ACC +2.538 / UA +2.496 / WF1 +3.455 pp; T and TV also improve ACC and UA. A and AT trade higher ACC for lower UA; AV decreases ACC and UA; full ATV declines.
- Random-rate UA gains do not imply ACC gains: e.g. MELD .2/.4 and IEMOCAP6 .4. IEMOCAP6 .7 increases ACC but decreases UA.

Persistent singleton/pair/full use the pre-fixed .7/.3/.0 BEST mapping; they are not separately trained condition-specific models.
Random points use each model's original per-rate ACC-selected epoch and exact saved corresponding UA/WF1, not a common-mask replay.
These single-seed pointwise observations are not significance tests or grounds for choosing a new test-label-conditioned inference rule.
