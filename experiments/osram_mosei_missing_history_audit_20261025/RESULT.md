# MOSEI missing-target history exposure: frozen Base/Gap audit

INTERNAL DIAGNOSTIC ONLY

Completed for original Flat128 and N10 seed66, all eight original per-rate Test-best checkpoints/exact GCNet masks, cmerc-repro. No new training, missing-mask generation, Query/write intervention or coefficient selection.

## Definitions and limits

- `never`:current target missing and never observed strictly before now; includes first turns.
- `never_with_history`:same, but excludes first turns, so other observed modalities have already entered shared Memory.
- `seen`:current target missing and previously observed at least once.
- `FIRST`:no earlier utterance at all. These samples are not mixed with the nonempty-history comparison.
- Readout removal zeros an entire Base/target-Gap task-input slot, not only the extra Nested correction or a source-specific contribution. Other slots and parameters stay fixed.
- **N10 `Local` retains task-side Nested-corrected Local and raw LocalSkip. It is not a history-free baseline.** Removing its Gap-m slot also does not erase earlier Nested mixing into other slots. Flat128 `Local` is original Local with all classification Memory slots zero.
- N10 norms describe post-Nested task-input slots, not raw pre-correction CHM reads; Flat128 norms describe original reads. Do not interpret their difference as a change in Memory storage.
- Frozen zeroing can be out of distribution. These scores measure trained-model pathway sensitivity, not standalone representation quality or causal effects of having seen a modality.
- Binary scores exclude neutral gold; MSE retains it. Supports are reported. Empty cohorts are N/A. Between-cohort label/current-modality/history-length distributions differ.

## Main observed distinctions

All676 first turns (501 nonneutral) per rate have exactly zero Base/all Gap norm in both models, and every readout gives the same first-turn prediction within1e-6. The genuinely empty-history boundary behaves correctly.

At miss0.7, target Text missing with nonempty history that never contained Text:485 nonneutral examples. Flat Full ACC/W-F171.340/65.495 versus original Local67.835/54.835. Shared Memory can still provide useful A/V-derived context without historical Text; never-seen Text is not an empty Memory.

In this Flat subset, removing Gap-T improves ACC/W-F1 to72.577/66.021 (8corrections/2harms). With past Text observed (1618examples), the same removal reduces ACC/W-F1 from77.009/77.028 to74.289/73.235 (85corrections/129harms). This is a conditional difference, not a universal instruction to disable Gap-T when historical Text is absent.

N10 does not have the same simple pattern:in its never-with-history Text subset, removing Gap-T changes ACC70.515→71.546 but W-F168.974→68.561 (17corrections/12harms). ACC/W-F1 can disagree because classwise errors change. It is not valid to label the branch helpful/harmful using only one metric.

Across rates, effects are not uniform. For example Flat's never-with-history Text group at miss0.5 has lower W-F1 after Gap-T removal, while N10's group at miss0.2 has only19nonneutral examples and also deteriorates. No coefficient/gating rule or new model was trained or selected.

## Four-readout Text examples at miss0.7

| Model | Strict-past Text | N | Full ACC/WF1 | No Base ACC/WF1 | No Gap-T ACC/WF1 | No Base/Gap-T ACC/WF1 |
|---|---|---:|---:|---:|---:|---:|
| Flat128 | never_with_history | 485 | 71.340/65.495 | 70.515/61.781 | 72.577/66.021 | 68.660/57.723 |
| Flat128 | seen | 1618 | 77.009/77.028 | 76.391/75.965 | 74.289/73.235 | 65.204/57.974 |
| N10 | never_with_history | 485 | 70.515/68.974 | 71.134/66.061 | 71.546/68.561 | 71.340/64.049 |
| N10 | seen | 1618 | 76.205/76.152 | 75.896/75.710 | 74.845/74.354 | 70.519/68.377 |

## Every rate, target and nonempty-history condition

All values below are W-F1(%). `LG-m` retains only that Gap with Local, `noB` retains all active Gap but no Base, `noG-m` retains Base and other active Gap, `noB/noG-m` retains other active Gap. No missing-target cohort exists at rate0.0. N is nonneutral; complete/neutral/positive/negative counts, ACC, Recall, MSE and corrections/harms are in results/METRICS.csv.

### Flat128: current A missing

| Rate | Past target | N | Local slot | LB | LG-m | Full | noB | noG-m | noB/noG-m | Base norm | Gap-m norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | never_with_history | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.0 | seen | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.1 | never_with_history | 7 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 4.900 | 1.777 |
| 0.1 | seen | 321 | 86.136 | 88.112 | 87.757 | 89.386 | 89.085 | 89.085 | 87.812 | 7.265 | 4.268 |
| 0.2 | never_with_history | 14 | 92.298 | 100.000 | 92.298 | 100.000 | 92.298 | 100.000 | 92.298 | 3.852 | 1.322 |
| 0.2 | seen | 616 | 81.290 | 85.418 | 83.041 | 87.343 | 84.936 | 87.156 | 84.504 | 6.594 | 3.956 |
| 0.3 | never_with_history | 59 | 79.614 | 79.614 | 79.614 | 83.320 | 83.320 | 83.320 | 81.189 | 4.143 | 1.606 |
| 0.3 | seen | 894 | 77.345 | 84.124 | 79.814 | 86.479 | 83.318 | 86.246 | 82.453 | 6.598 | 4.254 |
| 0.4 | never_with_history | 95 | 76.926 | 79.893 | 76.926 | 80.365 | 75.363 | 79.893 | 76.926 | 4.089 | 2.228 |
| 0.4 | seen | 1151 | 75.098 | 80.279 | 77.714 | 85.064 | 82.943 | 83.340 | 81.412 | 6.611 | 4.540 |
| 0.5 | never_with_history | 195 | 71.988 | 74.793 | 71.988 | 77.904 | 73.842 | 74.793 | 74.979 | 4.040 | 2.293 |
| 0.5 | seen | 1361 | 74.880 | 83.011 | 79.344 | 85.962 | 84.969 | 86.076 | 84.389 | 6.150 | 4.358 |
| 0.6 | never_with_history | 301 | 80.191 | 82.582 | 80.603 | 80.876 | 80.338 | 81.008 | 80.235 | 3.567 | 2.154 |
| 0.6 | seen | 1618 | 74.936 | 80.300 | 79.174 | 81.778 | 82.038 | 81.671 | 80.939 | 5.776 | 4.315 |
| 0.7 | never_with_history | 384 | 72.225 | 78.003 | 73.387 | 80.419 | 78.605 | 80.060 | 78.215 | 3.703 | 2.411 |
| 0.7 | seen | 1696 | 67.287 | 76.181 | 72.117 | 80.551 | 80.302 | 80.511 | 78.419 | 5.295 | 4.041 |

### Flat128: current T missing

| Rate | Past target | N | Local slot | LB | LG-m | Full | noB | noG-m | noB/noG-m | Base norm | Gap-m norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | never_with_history | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.0 | seen | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.1 | never_with_history | 2 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 3.264 | 0.253 |
| 0.1 | seen | 317 | 62.656 | 72.614 | 76.694 | 76.875 | 77.072 | 72.955 | 65.794 | 7.427 | 5.621 |
| 0.2 | never_with_history | 19 | 69.659 | 76.096 | 69.659 | 76.096 | 69.659 | 76.096 | 69.659 | 2.075 | 0.708 |
| 0.2 | seen | 592 | 56.184 | 77.207 | 79.306 | 79.543 | 79.714 | 77.949 | 61.182 | 6.909 | 5.967 |
| 0.3 | never_with_history | 42 | 51.130 | 55.639 | 51.130 | 58.050 | 51.130 | 59.792 | 51.130 | 2.001 | 0.889 |
| 0.3 | seen | 871 | 53.028 | 75.301 | 77.903 | 79.023 | 78.766 | 76.019 | 60.148 | 6.690 | 5.849 |
| 0.4 | never_with_history | 96 | 49.376 | 51.718 | 49.376 | 51.164 | 48.878 | 51.164 | 48.878 | 2.417 | 1.122 |
| 0.4 | seen | 1167 | 47.066 | 68.879 | 75.168 | 79.714 | 77.334 | 73.076 | 56.305 | 6.688 | 5.544 |
| 0.5 | never_with_history | 199 | 66.378 | 65.303 | 66.378 | 68.027 | 67.208 | 66.065 | 66.378 | 2.099 | 1.125 |
| 0.5 | seen | 1400 | 49.338 | 73.647 | 77.575 | 77.992 | 77.669 | 75.776 | 61.049 | 6.438 | 5.687 |
| 0.6 | never_with_history | 279 | 62.192 | 62.761 | 62.934 | 67.230 | 64.815 | 66.202 | 62.062 | 2.054 | 1.401 |
| 0.6 | seen | 1541 | 56.139 | 73.299 | 74.800 | 75.055 | 76.113 | 74.914 | 70.537 | 6.116 | 5.469 |
| 0.7 | never_with_history | 485 | 54.835 | 63.491 | 57.840 | 65.495 | 61.781 | 66.021 | 57.723 | 2.274 | 1.385 |
| 0.7 | seen | 1618 | 43.218 | 67.142 | 73.654 | 77.028 | 75.965 | 73.235 | 57.974 | 5.773 | 5.170 |

### Flat128: current V missing

| Rate | Past target | N | Local slot | LB | LG-m | Full | noB | noG-m | noB/noG-m | Base norm | Gap-m norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | never_with_history | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.0 | seen | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.1 | never_with_history | 6 | 82.857 | 82.857 | 82.857 | 82.857 | 82.857 | 82.857 | 82.857 | 4.438 | 1.395 |
| 0.1 | seen | 316 | 84.989 | 85.443 | 86.668 | 87.303 | 86.975 | 85.749 | 86.953 | 7.365 | 6.348 |
| 0.2 | never_with_history | 19 | 70.140 | 70.140 | 70.140 | 70.140 | 70.140 | 70.140 | 70.140 | 3.562 | 1.425 |
| 0.2 | seen | 586 | 78.214 | 84.308 | 81.428 | 87.143 | 84.139 | 85.745 | 82.110 | 6.627 | 5.571 |
| 0.3 | never_with_history | 49 | 93.797 | 93.797 | 93.797 | 93.948 | 91.837 | 91.837 | 93.797 | 4.149 | 1.558 |
| 0.3 | seen | 875 | 80.854 | 86.161 | 83.887 | 86.843 | 85.714 | 86.311 | 84.059 | 6.418 | 5.460 |
| 0.4 | never_with_history | 102 | 84.842 | 88.149 | 85.925 | 90.124 | 86.994 | 90.124 | 86.994 | 4.090 | 1.650 |
| 0.4 | seen | 1145 | 77.129 | 82.500 | 81.423 | 86.287 | 84.459 | 85.048 | 83.359 | 6.550 | 5.804 |
| 0.5 | never_with_history | 197 | 76.592 | 79.655 | 76.592 | 78.339 | 76.293 | 78.789 | 78.064 | 3.782 | 1.976 |
| 0.5 | seen | 1383 | 72.440 | 82.304 | 76.973 | 83.218 | 80.971 | 82.793 | 79.820 | 6.104 | 5.283 |
| 0.6 | never_with_history | 314 | 73.157 | 73.169 | 73.040 | 76.846 | 75.543 | 76.033 | 73.588 | 3.733 | 2.230 |
| 0.6 | seen | 1563 | 76.066 | 82.982 | 82.150 | 84.077 | 84.245 | 83.677 | 83.398 | 5.738 | 4.695 |
| 0.7 | never_with_history | 438 | 77.132 | 79.828 | 77.169 | 80.620 | 78.745 | 79.858 | 79.246 | 3.669 | 2.149 |
| 0.7 | seen | 1645 | 66.808 | 77.670 | 72.591 | 81.141 | 78.627 | 81.076 | 77.587 | 5.338 | 4.802 |

### N10: current A missing

| Rate | Past target | N | Local slot | LB | LG-m | Full | noB | noG-m | noB/noG-m | Base norm | Gap-m norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | never_with_history | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.0 | seen | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.1 | never_with_history | 7 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 5.061 | 1.971 |
| 0.1 | seen | 321 | 85.006 | 87.330 | 85.942 | 87.664 | 86.660 | 87.996 | 86.697 | 7.134 | 3.880 |
| 0.2 | never_with_history | 14 | 93.197 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 93.197 | 4.067 | 1.321 |
| 0.2 | seen | 616 | 82.385 | 85.527 | 84.381 | 86.261 | 86.887 | 86.758 | 86.498 | 7.579 | 4.450 |
| 0.3 | never_with_history | 59 | 81.721 | 83.756 | 81.721 | 85.729 | 85.376 | 85.729 | 83.320 | 4.669 | 1.852 |
| 0.3 | seen | 894 | 78.746 | 84.073 | 82.369 | 86.043 | 85.158 | 86.053 | 83.844 | 7.001 | 4.271 |
| 0.4 | never_with_history | 95 | 83.041 | 85.303 | 84.007 | 84.327 | 84.327 | 84.327 | 85.303 | 4.058 | 2.130 |
| 0.4 | seen | 1151 | 77.791 | 82.854 | 80.840 | 85.541 | 84.598 | 84.205 | 83.566 | 7.254 | 4.817 |
| 0.5 | never_with_history | 195 | 73.672 | 74.811 | 73.410 | 75.660 | 74.590 | 75.019 | 76.139 | 4.155 | 2.272 |
| 0.5 | seen | 1361 | 76.621 | 82.751 | 79.450 | 85.688 | 85.690 | 85.637 | 85.249 | 6.963 | 4.848 |
| 0.6 | never_with_history | 301 | 78.428 | 82.808 | 79.354 | 82.219 | 79.810 | 82.118 | 79.672 | 4.337 | 2.459 |
| 0.6 | seen | 1618 | 73.019 | 80.034 | 78.577 | 82.747 | 82.212 | 82.600 | 81.572 | 6.220 | 4.397 |
| 0.7 | never_with_history | 384 | 74.441 | 79.169 | 75.080 | 80.548 | 80.688 | 81.203 | 78.277 | 4.261 | 2.558 |
| 0.7 | seen | 1696 | 68.793 | 76.915 | 74.887 | 81.256 | 80.076 | 80.958 | 79.035 | 5.738 | 4.152 |

### N10: current T missing

| Rate | Past target | N | Local slot | LB | LG-m | Full | noB | noG-m | noB/noG-m | Base norm | Gap-m norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | never_with_history | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.0 | seen | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.1 | never_with_history | 2 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 100.000 | 3.603 | 0.465 |
| 0.1 | seen | 317 | 51.633 | 73.767 | 75.757 | 76.055 | 76.458 | 73.913 | 56.916 | 7.404 | 6.540 |
| 0.2 | never_with_history | 19 | 69.659 | 66.986 | 69.659 | 76.096 | 66.986 | 66.986 | 66.986 | 2.759 | 0.792 |
| 0.2 | seen | 592 | 53.014 | 76.203 | 79.985 | 79.921 | 79.858 | 76.665 | 61.052 | 7.694 | 6.517 |
| 0.3 | never_with_history | 42 | 51.130 | 58.050 | 58.050 | 51.112 | 54.591 | 54.591 | 52.457 | 2.679 | 1.180 |
| 0.3 | seen | 871 | 52.093 | 74.691 | 79.250 | 79.464 | 79.793 | 76.372 | 61.974 | 7.057 | 6.299 |
| 0.4 | never_with_history | 96 | 50.607 | 56.083 | 50.607 | 59.172 | 51.529 | 55.399 | 50.045 | 2.841 | 1.353 |
| 0.4 | seen | 1167 | 51.442 | 74.559 | 78.231 | 78.927 | 78.559 | 75.791 | 62.547 | 7.368 | 6.341 |
| 0.5 | never_with_history | 199 | 69.623 | 70.265 | 69.623 | 69.696 | 71.530 | 71.849 | 69.306 | 2.837 | 1.390 |
| 0.5 | seen | 1400 | 50.214 | 73.030 | 77.162 | 77.424 | 77.169 | 74.305 | 63.743 | 7.047 | 6.082 |
| 0.6 | never_with_history | 279 | 55.569 | 60.317 | 58.873 | 65.485 | 61.464 | 64.807 | 60.000 | 2.879 | 1.674 |
| 0.6 | seen | 1541 | 49.220 | 72.401 | 75.860 | 76.519 | 75.974 | 74.323 | 70.322 | 6.579 | 5.870 |
| 0.7 | never_with_history | 485 | 56.714 | 65.887 | 63.064 | 68.974 | 66.061 | 68.561 | 64.049 | 3.042 | 1.921 |
| 0.7 | seen | 1618 | 46.579 | 70.214 | 74.527 | 76.152 | 75.710 | 74.354 | 68.377 | 6.259 | 5.551 |

### N10: current V missing

| Rate | Past target | N | Local slot | LB | LG-m | Full | noB | noG-m | noB/noG-m | Base norm | Gap-m norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | never_with_history | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.0 | seen | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.1 | never_with_history | 6 | 82.857 | 82.857 | 82.857 | 82.857 | 82.857 | 82.857 | 82.857 | 3.861 | 1.172 |
| 0.1 | seen | 316 | 84.139 | 88.443 | 89.439 | 89.749 | 89.749 | 88.443 | 86.149 | 7.331 | 5.938 |
| 0.2 | never_with_history | 19 | 65.587 | 65.587 | 65.587 | 72.227 | 65.587 | 65.587 | 65.587 | 4.198 | 1.435 |
| 0.2 | seen | 586 | 79.596 | 84.233 | 83.684 | 85.091 | 84.225 | 84.413 | 83.488 | 7.625 | 5.274 |
| 0.3 | never_with_history | 49 | 93.797 | 93.797 | 93.797 | 95.918 | 95.918 | 93.797 | 93.797 | 4.031 | 1.547 |
| 0.3 | seen | 875 | 81.783 | 85.732 | 86.295 | 86.117 | 85.951 | 86.210 | 85.256 | 7.015 | 5.904 |
| 0.4 | never_with_history | 102 | 86.275 | 89.177 | 88.235 | 89.251 | 88.310 | 89.251 | 88.310 | 4.630 | 1.789 |
| 0.4 | seen | 1145 | 78.747 | 84.434 | 83.104 | 85.408 | 84.777 | 84.802 | 84.485 | 7.199 | 5.265 |
| 0.5 | never_with_history | 197 | 78.064 | 81.369 | 78.262 | 80.046 | 79.477 | 80.046 | 78.775 | 4.446 | 2.045 |
| 0.5 | seen | 1383 | 74.399 | 81.275 | 81.063 | 82.268 | 81.849 | 82.098 | 81.237 | 6.816 | 5.195 |
| 0.6 | never_with_history | 314 | 71.667 | 75.222 | 73.961 | 76.501 | 74.058 | 75.266 | 73.992 | 3.854 | 2.269 |
| 0.6 | seen | 1563 | 73.659 | 82.805 | 83.205 | 83.297 | 83.043 | 83.737 | 83.414 | 6.404 | 5.640 |
| 0.7 | never_with_history | 438 | 76.243 | 79.184 | 78.971 | 79.123 | 79.686 | 79.724 | 78.798 | 3.851 | 2.393 |
| 0.7 | seen | 1645 | 67.030 | 78.627 | 77.950 | 80.688 | 80.202 | 80.538 | 78.767 | 6.017 | 5.492 |

## First-turn equality checks

| Model | Rate | N | Full ACC | Full WF1 | Local slot WF1 | Base norm | Gap-A/T/V norm |
|---|---:|---:|---:|---:|---:|---:|---|
| Flat128 | 0.0 | 501 | 85.230 | 84.265 | 84.265 | 0 | 0/0/0 |
| Flat128 | 0.1 | 501 | 83.034 | 82.831 | 82.831 | 0 | 0/0/0 |
| Flat128 | 0.2 | 501 | 82.834 | 81.566 | 81.566 | 0 | 0/0/0 |
| Flat128 | 0.3 | 501 | 81.238 | 79.023 | 79.023 | 0 | 0/0/0 |
| Flat128 | 0.4 | 501 | 79.641 | 76.107 | 76.107 | 0 | 0/0/0 |
| Flat128 | 0.5 | 501 | 79.242 | 75.781 | 75.781 | 0 | 0/0/0 |
| Flat128 | 0.6 | 501 | 77.445 | 75.083 | 75.083 | 0 | 0/0/0 |
| Flat128 | 0.7 | 501 | 77.844 | 72.533 | 72.533 | 0 | 0/0/0 |
| N10 | 0.0 | 501 | 83.832 | 82.742 | 82.742 | 0 | 0/0/0 |
| N10 | 0.1 | 501 | 84.232 | 82.961 | 82.961 | 0 | 0/0/0 |
| N10 | 0.2 | 501 | 82.834 | 81.713 | 81.713 | 0 | 0/0/0 |
| N10 | 0.3 | 501 | 81.437 | 79.091 | 79.091 | 0 | 0/0/0 |
| N10 | 0.4 | 501 | 80.240 | 77.742 | 77.742 | 0 | 0/0/0 |
| N10 | 0.5 | 501 | 80.439 | 77.559 | 77.559 | 0 | 0/0/0 |
| N10 | 0.6 | 501 | 77.246 | 72.631 | 72.631 | 0 | 0/0/0 |
| N10 | 0.7 | 501 | 77.844 | 72.533 | 72.533 | 0 | 0/0/0 |

## Verification and provenance

-Wall time 142.267seconds including two dataset/model setups. PhysicalGPU1, UUID GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049; no other jobs modified.
-45 local focused tests and42 remote snapshot tests passed. The first remote test command included an unrelated old Text-only test absent from the minimal snapshot and exited before tests/inference; rerunning the available42 tests succeeded. No model or mask behavior was changed to resolve that command issue.
-16saved prediction files,2096cohort/readout metric records independently verified against sklearn, exact original labels/IDs/masks, original checkpoint/prediction hashes and a separately constructed strictly-past cohort implementation.
-352actual Memory scans total (176per model);13readouts per batch, no extra scan, no optimizer, no updates. Frozen model/source checkpoint hashes unchanged; readout CPU/CUDA RNG unchanged.
-Observed-target Gap removal preserves full output; backward halves and inactive Gap are zero. First-turn read norms are exactly zero and predictions unchanged.
-Immutable source commit:eaf6c6a53e13f41c4fedabed4c4099d6c24af67b.
-Remote source:/data2/yb/remote_experiments/osram_mosei_missing_history_audit_20261025/source_v1
-Remote output:/data2/yb/remote_experiments/osram_mosei_missing_history_audit_20261025/replay_v1
-tmux:mosei_missing_history_20261025 (completed).
-Full remote SUMMARY SHA256:9c2e7c0f874c83bbd24ae2202f1d84b08136e9d0b1462ee50521302c0b0d6f3d.
-results/SUMMARY.json retains configurations/environment, original references/hashes, per-rate checkpoint provenance, cohort supports/norms/current-availability mix/history length and verification. Full original runtime data record/arrays remain remote, not uploaded.
-results/METRICS.csv contains all2096 readout/cohort rows across eight rates, including never-with-first-turn, seen and first-turn cohorts. CSV text is normalized to LF locally; original remote CSV SHA256 is retained in SUMMARY.
-Per-rate checkpoints were already Test-best selected. This is single-seed internal analysis, not new independently validated performance or a demonstration that unavailable historical modality content is recoverable.

