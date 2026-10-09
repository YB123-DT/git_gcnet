# Frozen Memory Residual Learning

**INTERNAL DIAGNOSTIC ONLY**

Status: COMPLETE; 168/168 runs.

Original backbone and historical probes inherit test-selected checkpoint provenance. Residual heads use fixed last epoch 100, with no test epoch selection. The backbone remains frozen. The three seeds are residual-head initializations, not independent backbone seeds. Gold-history uses unavailable-at-inference true historical labels and is an offline diagnostic only; it is not a guaranteed mathematical upper bound.

Metrics below average eight rates equally within each residual-head seed, then report mean ± sample standard deviation across seeds. High missing averages .5/.6/.7. No predictions are pooled across rates. Incomplete eight-rate results are not called eight-rate means.

| Group | 8-rate W-F1 (%) | High W-F1 (%) | Complete 8-rate seeds |
|---|---:|---:|---:|
| Original | 81.068 ± 0.000 | 76.352 ± 0.000 | 3 |
| A_local | 80.113 ± 0.159 | 74.570 ± 0.129 | 3 |
| A_donor | 78.384 ± 0.262 | 72.290 ± 0.834 | 3 |
| A_real | 77.579 ± 0.049 | 71.464 ± 0.509 | 3 |
| B_local_history | 80.246 ± 0.165 | 74.933 ± 0.158 | 3 |
| B_donor_history | 80.070 ± 0.110 | 74.796 ± 0.274 | 3 |
| B_real_history | 79.977 ± 0.275 | 74.777 ± 0.263 | 3 |
| B_gold_history | 79.973 ± 0.219 | 75.066 ± 0.400 | 3 |

## Per-rate W-F1 (%)

| Group | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 88.205 ± 0.000 | 86.507 ± 0.000 | 83.187 ± 0.000 | 80.763 ± 0.000 | 80.827 ± 0.000 | 77.494 ± 0.000 | 75.790 ± 0.000 | 75.773 ± 0.000 |
| A_local | 87.190 ± 0.245 | 84.987 ± 0.109 | 83.816 ± 0.311 | 81.627 ± 0.304 | 79.571 ± 0.396 | 75.186 ± 0.617 | 74.452 ± 0.698 | 74.071 ± 0.100 |
| A_donor | 87.556 ± 0.356 | 84.608 ± 0.293 | 80.995 ± 0.572 | 78.110 ± 1.735 | 78.932 ± 0.447 | 73.745 ± 1.033 | 72.839 ± 1.273 | 70.286 ± 1.719 |
| A_real | 87.335 ± 0.610 | 83.438 ± 0.202 | 81.448 ± 0.860 | 77.139 ± 0.834 | 76.875 ± 0.411 | 71.955 ± 0.159 | 71.088 ± 0.973 | 71.349 ± 0.571 |
| B_local_history | 87.304 ± 0.343 | 85.141 ± 0.224 | 83.466 ± 0.299 | 81.674 ± 0.189 | 79.584 ± 0.603 | 75.475 ± 1.000 | 74.857 ± 0.644 | 74.467 ± 0.258 |
| B_donor_history | 87.202 ± 0.273 | 85.351 ± 0.133 | 82.913 ± 0.745 | 81.374 ± 0.391 | 79.333 ± 0.905 | 75.423 ± 1.056 | 74.359 ± 0.915 | 74.605 ± 0.167 |
| B_real_history | 86.794 ± 0.235 | 85.366 ± 0.258 | 82.862 ± 0.592 | 81.223 ± 0.362 | 79.238 ± 0.595 | 75.832 ± 0.947 | 74.369 ± 0.639 | 74.131 ± 0.282 |
| B_gold_history | 87.105 ± 0.204 | 84.609 ± 0.153 | 82.711 ± 0.678 | 81.225 ± 0.389 | 78.939 ± 0.833 | 76.572 ± 1.400 | 74.295 ± 0.623 | 74.331 ± 0.482 |

## Paired gains (percentage points)

| Model − reference | Seed | 8-rate ΔW-F1 | High ΔW-F1 |
|---|---:|---:|---:|
| A_real − Original | 66 | -3.541 | -4.913 |
| A_real − Original | 67 | -3.485 | -5.385 |
| A_real − Original | 68 | -3.443 | -4.367 |
| A_real − A_local | 66 | -2.702 | -3.258 |
| A_real − A_local | 67 | -2.595 | -3.606 |
| A_real − A_local | 68 | -2.305 | -2.453 |
| A_real − A_donor | 66 | -0.779 | -0.267 |
| A_real − A_donor | 67 | -1.093 | -2.278 |
| A_real − A_donor | 68 | -0.544 | +0.066 |
| B_real_history − Original | 66 | -1.231 | -1.796 |
| B_real_history − Original | 67 | -0.774 | -1.284 |
| B_real_history − Original | 68 | -1.269 | -1.644 |
| B_real_history − B_local_history | 66 | -0.396 | -0.247 |
| B_real_history − B_local_history | 67 | -0.124 | -0.040 |
| B_real_history − B_local_history | 68 | -0.289 | -0.180 |
| B_real_history − B_donor_history | 66 | -0.132 | +0.055 |
| B_real_history − B_donor_history | 67 | +0.106 | +0.226 |
| B_real_history − B_donor_history | 68 | -0.254 | -0.336 |

## Interpretation boundaries

Real Memory must beat Original, Local Control, and Donor Control in paired comparisons. A gain supports exploitable residual information only relative to this frozen backbone and corrector class. Historical-scalar gains do not establish pure semantic mediation. One backbone cannot establish multi-backbone stability or a general mechanism.

Corrections/harms are relative to Original on nonneutral labels. The sums in per_seed.csv repeat utterances across missing rates and are not counts of unique utterances. No significance test treats rates as independent observations.

Detailed files: per_rate.csv, per_seed.csv, comparisons.csv, SUMMARY.json. Accuracy and W-F1 in JSON/CSV are fractions except explicitly named _pp differences; report tables are percentages. Provenance and effective source protocols are retained in SUMMARY.json.
