# OSRAM 12项历史来源贡献分析（12-Source Evidence Audit）

INTERNAL DIAGNOSTIC ONLY

Dataset: MELD; seed66; fold 1.

Source Git SHA: `cb02830e794297e9d3654b5f0c9a4900c6ac0cdf`.

Fixed original miss=0.3 checkpoint: `/home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/runs_v1/S66_WD1/best_miss_0p3.pt`.

Checkpoint SHA256: `6f031b515723b441129e16a2e6217410e0eceee2c468e1cbe7b7c3d8276bdb7d`.

No training, no TASR Adapter, no parameter updates. One fixed original Flat checkpoint is used across all conditions. Random conditions replay saved original GCNet masks; Persistent conditions use fixed whole-conversation visible patterns.

Only seed66 is analyzed; IEMOCAP-6 is fold5 only. No confidence intervals or significance claims. The original checkpoint was selected by historical Test-oracle, not independent test generalization.

## 1. Full-model condition metrics

ACC, seven/six-class explicit Macro Recall (UA), and W-F1 are percentages; CE is in nats. Aggregates are unweighted means across conditions, never pooled sample scores. Persistent ATV is separate from the six-pattern mean.

| Condition | N | ACC | UA | W-F1 | CE |
|---|---:|---:|---:|---:|---:|
| Random 0.0 | 2610 | 66.2835 | 48.1454 | 65.6466 | 1.0904 |
| Random 0.1 | 2610 | 65.2490 | 46.0413 | 64.0714 | 1.1184 |
| Random 0.2 | 2610 | 63.1034 | 42.7742 | 61.5785 | 1.1479 |
| Random 0.3 | 2610 | 62.2605 | 40.2072 | 59.9957 | 1.1782 |
| Random 0.4 | 2610 | 60.6897 | 37.5403 | 57.9297 | 1.2128 |
| Random 0.5 | 2610 | 59.4636 | 33.8428 | 55.6996 | 1.2035 |
| Random 0.6 | 2610 | 57.4330 | 32.1114 | 53.0491 | 1.2938 |
| Random 0.7 | 2610 | 56.2069 | 30.2214 | 51.1727 | 1.3107 |
| Persistent A | 2610 | 48.8889 | 19.6405 | 39.9765 | 1.4545 |
| Persistent T | 2610 | 66.0920 | 47.0227 | 65.0565 | 1.0650 |
| Persistent V | 2610 | 48.0460 | 14.2630 | 31.2349 | 1.5171 |
| Persistent AT | 2610 | 67.0115 | 46.8684 | 65.6481 | 1.0591 |
| Persistent AV | 2610 | 49.1188 | 17.7150 | 37.9721 | 1.4606 |
| Persistent TV | 2610 | 66.4751 | 46.7340 | 65.2155 | 1.0555 |
| Persistent ATV | 2610 | 66.2835 | 48.1454 | 65.6466 | 1.0904 |
| random8 | — | 61.3362 | 38.8605 | 58.6429 | 1.1945 |
| high_missing | — | 57.7011 | 32.0585 | 53.3071 | 1.2693 |
| persistent6 | — | 57.6054 | 32.0406 | 50.8506 | 1.2686 |

## 2. Reconstruction and integrity

The maxima below are collected across every batch and condition, including absolute and relative source-sum errors when recorded. Exact original readout, fixed current input/query, unchanged weights and RNG checks are saved in SUMMARY/per-condition JSON.

| Check | Maximum error |
|---|---:|
| base_sum_max_abs_error | 3.57627869e-07 |
| base_sum_max_relative_error | 3.37134054e-07 |
| gap_sum_max_abs_error | 2.98023224e-07 |
| gap_sum_max_relative_error | 5.2267518e-07 |
| memory_sum_max_abs_error | 5.36441803e-07 |
| memory_sum_max_relative_error | 2.62335362e-07 |
| original_readout_reproduction_max_abs_error | 0 |
| prewrite_memory_sum_max_abs_error | 5.36441803e-07 |
| prewrite_memory_sum_max_relative_error | 2.63542574e-07 |
| source_sum_max_abs_error | 3.57627869e-07 |

## 3. Saved Value-source state strength

These are source-state Frobenius norms of the decayed, strictly prewrite history at each target. They quantify algebraic contribution magnitude, not emotional utility; source norms do not add to the joint norm.

| Condition | Source A norm | Source T norm | Source V norm | Joint norm |
|---|---:|---:|---:|---:|
| random 0.0 | 12.2149 | 13.1920 | 11.9057 | 21.6153 |
| random 0.1 | 12.1465 | 12.8789 | 11.7931 | 21.4035 |
| random 0.2 | 11.7953 | 12.2148 | 11.3756 | 20.7014 |
| random 0.3 | 11.4220 | 11.4473 | 10.9610 | 19.9228 |
| random 0.4 | 10.7247 | 10.8206 | 10.1558 | 18.8458 |
| random 0.5 | 9.8906 | 9.8793 | 9.4920 | 17.6118 |
| random 0.6 | 8.8734 | 8.3509 | 8.5058 | 15.8745 |
| random 0.7 | 7.3332 | 7.9767 | 7.0759 | 14.2958 |
| persistent A | 11.8461 | 0.0000 | 0.0000 | 11.8461 |
| persistent T | 0.0000 | 13.3454 | 0.0000 | 13.3454 |
| persistent V | 0.0000 | 0.0000 | 11.5783 | 11.5783 |
| persistent AT | 12.2905 | 13.2004 | 0.0000 | 18.0731 |
| persistent AV | 12.0492 | 0.0000 | 11.6996 | 16.7812 |
| persistent TV | 0.0000 | 13.1764 | 11.9230 | 17.7851 |
| persistent ATV | 12.2149 | 13.1920 | 11.9057 | 21.6153 |

## 4. Read strength and frozen decision contribution

Every condition has two 4×3 tables. Rows are read destinations; columns are historical Value-write sources. Numbers are not pure modality semantics. Active Base includes zero-history first turns. A currently visible modality has an inactive Gap: N/A, not zero importance.

Read norms and removal effects use exactly the same active-row cohort. ΔCE = CE_removed − CE_full: positive indicates beneficial presence under this intervention. ACC/UA/W-F1 deltas and corrections/harms appear in SLOTS.csv. A removal correction is full-wrong → removed-correct; a removal harm is full-correct → removed-wrong.

### random 0.0

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.1295 (2610) | 3.1492 (2610) | 1.4171 (2610) |
| Gap-A | N/A (0) | N/A (0) | N/A (0) |
| Gap-T | N/A (0) | N/A (0) | N/A (0) |
| Gap-V | N/A (0) | N/A (0) | N/A (0) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | -0.020985 | 0.051500 | -0.004019 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.3618 |
| Gap-A | 0 | N/A |
| Gap-T | 0 | N/A |
| Gap-V | 0 | N/A |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.020645 | 0.057481 | 0.002621 | 0.039457 | 0.000000000000 |

### random 0.1

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.1567 (2610) | 3.0376 (2610) | 1.3788 (2610) |
| Gap-A | 1.0985 (255) | 1.8038 (255) | 1.1266 (255) |
| Gap-T | 1.0225 (258) | 2.4389 (258) | 1.2533 (258) |
| Gap-V | 1.0050 (253) | 1.7050 (253) | 1.3924 (253) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | -0.016782 | 0.044116 | -0.000704 |
| Gap-A | -0.005503 | 0.007659 | 0.002368 |
| Gap-T | -0.005274 | -0.009935 | 0.002021 |
| Gap-V | 0.003147 | -0.001147 | -0.004990 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.3614 |
| Gap-A | 223 | 0.3915 |
| Gap-T | 234 | 0.3688 |
| Gap-V | 228 | 0.3963 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.017092 | 0.054737 | 0.006058 | 0.043703 | 0.000000000000 |

### random 0.2

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.1528 (2610) | 2.8621 (2610) | 1.3148 (2610) |
| Gap-A | 1.0526 (546) | 1.7444 (546) | 1.0643 (546) |
| Gap-T | 0.9889 (539) | 2.2472 (539) | 1.2017 (539) |
| Gap-V | 0.9507 (545) | 1.5479 (545) | 1.3385 (545) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | -0.011501 | 0.037745 | -0.002474 |
| Gap-A | -0.000140 | -0.001223 | 0.001573 |
| Gap-T | -0.004217 | 0.015196 | -0.003338 |
| Gap-V | -0.001223 | 0.001130 | -0.003965 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.3555 |
| Gap-A | 485 | 0.3792 |
| Gap-T | 477 | 0.3572 |
| Gap-V | 471 | 0.3869 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.011303 | 0.058287 | 0.001889 | 0.048873 | 0.000000000000 |

### random 0.3

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.1577 (2610) | 2.6636 (2610) | 1.2592 (2610) |
| Gap-A | 1.0737 (755) | 1.6441 (755) | 1.0539 (755) |
| Gap-T | 0.9998 (837) | 2.1312 (837) | 1.2055 (837) |
| Gap-V | 0.9932 (787) | 1.5343 (787) | 1.3043 (787) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | -0.009192 | 0.034673 | -0.001476 |
| Gap-A | -0.000988 | -0.000063 | 0.001182 |
| Gap-T | -0.003124 | 0.008699 | -0.000484 |
| Gap-V | 0.003464 | 0.000999 | 0.000546 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.3542 |
| Gap-A | 674 | 0.3739 |
| Gap-T | 744 | 0.3523 |
| Gap-V | 710 | 0.3757 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.007540 | 0.058799 | 0.005209 | 0.056467 | 0.000000000000 |

### random 0.4

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.1334 (2610) | 2.5251 (2610) | 1.1749 (2610) |
| Gap-A | 1.0775 (1032) | 1.6497 (1032) | 1.0108 (1032) |
| Gap-T | 1.0246 (1055) | 2.0904 (1055) | 1.1213 (1055) |
| Gap-V | 0.9456 (1033) | 1.4516 (1033) | 1.1979 (1033) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | -0.002556 | 0.013576 | 0.000644 |
| Gap-A | -0.002995 | -0.002182 | 0.000878 |
| Gap-T | 0.001217 | -0.004923 | -0.000556 |
| Gap-V | 0.000080 | -0.001886 | -0.002028 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.3444 |
| Gap-A | 929 | 0.3593 |
| Gap-T | 944 | 0.3437 |
| Gap-V | 920 | 0.3568 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.000203 | 0.034161 | 0.005314 | 0.039272 | 0.000000000000 |

### random 0.5

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.1097 (2610) | 2.3537 (2610) | 1.0908 (2610) |
| Gap-A | 1.0453 (1302) | 1.5392 (1302) | 0.9564 (1302) |
| Gap-T | 1.0149 (1261) | 1.9358 (1261) | 1.0770 (1261) |
| Gap-V | 0.8936 (1326) | 1.3848 (1326) | 1.1495 (1326) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.001401 | 0.018474 | 0.000216 |
| Gap-A | -0.000443 | -0.000334 | 0.001478 |
| Gap-T | -0.000654 | 0.000598 | -0.001558 |
| Gap-V | 0.000855 | -0.002122 | -0.003211 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.3337 |
| Gap-A | 1154 | 0.3493 |
| Gap-T | 1139 | 0.3339 |
| Gap-V | 1190 | 0.3475 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.005299 | 0.041237 | 0.003513 | 0.050049 | 0.000000000000 |

### random 0.6

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.0991 (2610) | 2.0289 (2610) | 1.0049 (2610) |
| Gap-A | 1.0162 (1548) | 1.3303 (1548) | 0.8794 (1548) |
| Gap-T | 0.9271 (1570) | 1.6137 (1570) | 0.9467 (1570) |
| Gap-V | 0.8261 (1552) | 1.1991 (1552) | 0.9981 (1552) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.004830 | 0.008905 | 0.002497 |
| Gap-A | 0.000579 | -0.002272 | 0.001653 |
| Gap-T | 0.000176 | 0.000537 | -0.000886 |
| Gap-V | 0.000800 | -0.000606 | -0.003142 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.3139 |
| Gap-A | 1383 | 0.3244 |
| Gap-T | 1411 | 0.2995 |
| Gap-V | 1381 | 0.3216 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.009299 | 0.028898 | 0.005546 | 0.043744 | 0.000000000000 |

### random 0.7

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.9990 (2610) | 1.9861 (2610) | 0.8268 (2610) |
| Gap-A | 0.8635 (1762) | 1.2569 (1762) | 0.7438 (1762) |
| Gap-T | 0.8180 (1686) | 1.5447 (1686) | 0.7671 (1686) |
| Gap-V | 0.6903 (1772) | 1.1204 (1772) | 0.7889 (1772) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.007324 | 0.013356 | 0.003496 |
| Gap-A | 0.000001 | 0.000407 | 0.000080 |
| Gap-T | 0.000434 | -0.000736 | -0.001443 |
| Gap-V | 0.000073 | 0.001197 | -0.001197 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.2806 |
| Gap-A | 1569 | 0.2902 |
| Gap-T | 1510 | 0.2728 |
| Gap-V | 1581 | 0.2870 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.012152 | 0.036780 | 0.005769 | 0.054701 | 0.000000000000 |

### persistent A

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.5989 (2610) | 0.0000 (2610) | 0.0000 (2610) |
| Gap-A | N/A (0) | N/A (0) | N/A (0) |
| Gap-T | 0.8303 (2610) | 0.0000 (2610) | 0.0000 (2610) |
| Gap-V | 0.7350 (2610) | 0.0000 (2610) | 0.0000 (2610) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.018387 | 0.000000 | 0.000000 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | -0.004185 | 0.000000 | 0.000000 |
| Gap-V | -0.001711 | 0.000000 | 0.000000 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.0000 |
| Gap-A | 0 | N/A |
| Gap-T | 2330 | 0.0000 |
| Gap-V | 2330 | 0.0000 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.011744 | 0.000000 | 0.000000 | 0.011744 | 0.000000000000 |

### persistent T

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 (2610) | 2.9323 (2610) | 0.0000 (2610) |
| Gap-A | 0.0000 (2610) | 1.8018 (2610) | 0.0000 (2610) |
| Gap-T | N/A (0) | N/A (0) | N/A (0) |
| Gap-V | 0.0000 (2610) | 1.7685 (2610) | 0.0000 (2610) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.000000 | 0.038504 | 0.000000 |
| Gap-A | 0.000000 | 0.000939 | 0.000000 |
| Gap-T | N/A | N/A | N/A |
| Gap-V | 0.000000 | -0.000807 | 0.000000 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.0000 |
| Gap-A | 2330 | 0.0000 |
| Gap-T | 0 | N/A |
| Gap-V | 2330 | 0.0000 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.000000 | 0.052693 | -0.000000 | 0.052693 | 0.000000000000 |

### persistent V

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 (2610) | 0.0000 (2610) | 1.6273 (2610) |
| Gap-A | 0.0000 (2610) | 0.0000 (2610) | 1.0795 (2610) |
| Gap-T | 0.0000 (2610) | 0.0000 (2610) | 1.0510 (2610) |
| Gap-V | N/A (0) | N/A (0) | N/A (0) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.000000 | 0.000000 | 0.026195 |
| Gap-A | 0.000000 | 0.000000 | -0.001895 |
| Gap-T | 0.000000 | 0.000000 | -0.002409 |
| Gap-V | N/A | N/A | N/A |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.0000 |
| Gap-A | 2330 | 0.0000 |
| Gap-T | 2330 | 0.0000 |
| Gap-V | 0 | N/A |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.020897 | 0.020897 | 0.000000000000 |

### persistent AT

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.1293 (2610) | 2.8415 (2610) | 0.0000 (2610) |
| Gap-A | N/A (0) | N/A (0) | N/A (0) |
| Gap-T | N/A (0) | N/A (0) | N/A (0) |
| Gap-V | 0.7678 (2610) | 1.7442 (2610) | 0.0000 (2610) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | -0.008276 | 0.056370 | 0.000000 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | 0.000541 | 0.001138 | 0.000000 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.2254 |
| Gap-A | 0 | N/A |
| Gap-T | 0 | N/A |
| Gap-V | 2330 | 0.2383 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.005976 | 0.069354 | 0.000000 | 0.063378 | 0.000000000000 |

### persistent AV

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.2435 (2610) | 0.0000 (2610) | 1.2214 (2610) |
| Gap-A | N/A (0) | N/A (0) | N/A (0) |
| Gap-T | 0.7970 (2610) | 0.0000 (2610) | 0.9915 (2610) |
| Gap-V | N/A (0) | N/A (0) | N/A (0) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.022506 | 0.000000 | 0.010897 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | -0.006446 | 0.000000 | -0.002149 |
| Gap-V | N/A | N/A | N/A |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.2868 |
| Gap-A | 0 | N/A |
| Gap-T | 2330 | 0.2831 |
| Gap-V | 0 | N/A |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.015194 | 0.000000 | 0.011758 | 0.026951 | 0.000000000000 |

### persistent TV

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 (2610) | 2.8188 (2610) | 1.4330 (2610) |
| Gap-A | 0.0000 (2610) | 1.7709 (2610) | 1.0246 (2610) |
| Gap-T | N/A (0) | N/A (0) | N/A (0) |
| Gap-V | N/A (0) | N/A (0) | N/A (0) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.000000 | 0.035710 | -0.010071 |
| Gap-A | 0.000000 | 0.001128 | 0.000653 |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.2611 |
| Gap-A | 2330 | 0.2649 |
| Gap-T | 0 | N/A |
| Gap-V | 0 | N/A |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.000000 | 0.051877 | -0.007641 | 0.044236 | 0.000000000000 |

### persistent ATV

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 1.1295 (2610) | 3.1492 (2610) | 1.4171 (2610) |
| Gap-A | N/A (0) | N/A (0) | N/A (0) |
| Gap-T | N/A (0) | N/A (0) | N/A (0) |
| Gap-V | N/A (0) | N/A (0) | N/A (0) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | -0.020985 | 0.051500 | -0.004019 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 2330 | 0.3618 |
| Gap-A | 0 | N/A |
| Gap-T | 0 | N/A |
| Gap-V | 0 | N/A |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.020645 | 0.057481 | 0.002621 | 0.039457 | 0.000000000000 |

## 5. Full outputs and interpretation limits

- `CONDITIONS.csv`: all 15 full-model conditions, masks and resource checks.
- `SLOTS.csv`: 12 independent removals per condition, active N, read norms, ΔCE/ACC/UA/W-F1, corrections and harms.
- `SHAPLEY.csv`: all three source credits per condition, exact efficiency and distribution summaries.
- `SLOTS_GROUPED.csv`: full slot statistics stratified by current availability, strictly-past ever-seen pattern, per-source ever-seen and coverage. Empty/inactive slots remain blank numeric N/A.
- Per-condition JSON: all eight coalition scores, state norms, cancellation and strata; REPORT.json records aggregate metrics and source JSON hashes.

These measurements separate storage magnitude, original-query reads, and fixed-classifier marginal responses, but do not identify pure modality information, causal semantic relevance, or an optimal new module. Nonlinear independent slot effects are not additive. The three grouped Shapley credits sum to the negative-CE gain of full history reads over all-zero reads in this fixed game. Both full-zero history and partial subtraction can be out-of-distribution (OOD). Large norms, cancellation, or a single-seed metric gain alone do not demonstrate a model defect or a generally useful intervention.

Frozen inference seconds: 107.149; peak allocated GPU MiB: 320.422.
