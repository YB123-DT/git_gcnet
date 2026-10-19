# OSRAM 12项历史来源贡献分析（12-Source Evidence Audit）

INTERNAL DIAGNOSTIC ONLY

Dataset: IEMOCAPSix; seed66; fold 5.

Source Git SHA: `cb02830e794297e9d3654b5f0c9a4900c6ac0cdf`.

Fixed original miss=0.3 checkpoint: `/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016/runs/iemocap6/seed_66/fold_5/best_miss_0p3.pt`.

Checkpoint SHA256: `6a7968ade42513ee868e6dc82f306274a2e7bd53d1943c05f00c1a75af530fd0`.

No training, no TASR Adapter, no parameter updates. One fixed original Flat checkpoint is used across all conditions. Random conditions replay saved original GCNet masks; Persistent conditions use fixed whole-conversation visible patterns.

Only seed66 is analyzed; IEMOCAP-6 is fold5 only. No confidence intervals or significance claims. The original checkpoint was selected by historical Test-oracle, not independent test generalization.

## 1. Full-model condition metrics

ACC, seven/six-class explicit Macro Recall (UA), and W-F1 are percentages; CE is in nats. Aggregates are unweighted means across conditions, never pooled sample scores. Persistent ATV is separate from the six-pattern mean.

| Condition | N | ACC | UA | W-F1 | CE |
|---|---:|---:|---:|---:|---:|
| Random 0.0 | 1623 | 66.5434 | 66.3790 | 66.4099 | 0.9074 |
| Random 0.1 | 1623 | 65.9889 | 65.6089 | 65.7867 | 0.9115 |
| Random 0.2 | 1623 | 62.9082 | 63.0658 | 62.7043 | 0.9579 |
| Random 0.3 | 1623 | 64.3253 | 63.3895 | 64.1268 | 0.9305 |
| Random 0.4 | 1623 | 61.9840 | 61.5640 | 61.9641 | 0.9667 |
| Random 0.5 | 1623 | 61.4911 | 60.4755 | 61.0175 | 0.9864 |
| Random 0.6 | 1623 | 60.7517 | 61.1937 | 60.5289 | 1.0173 |
| Random 0.7 | 1623 | 56.9316 | 55.0611 | 56.5258 | 1.1455 |
| Persistent A | 1623 | 42.1442 | 44.6820 | 40.5819 | 1.4739 |
| Persistent T | 1623 | 53.4812 | 53.3546 | 53.3356 | 1.4156 |
| Persistent V | 1623 | 34.4424 | 30.9731 | 32.3986 | 1.9006 |
| Persistent AT | 1623 | 61.8608 | 61.6534 | 61.2034 | 1.0653 |
| Persistent AV | 1623 | 49.6611 | 52.8347 | 48.9157 | 1.2792 |
| Persistent TV | 1623 | 57.1781 | 55.8972 | 56.9879 | 1.1761 |
| Persistent ATV | 1623 | 66.5434 | 66.3790 | 66.4099 | 0.9074 |
| random8 | — | 62.6155 | 62.0922 | 62.3830 | 0.9779 |
| high_missing | — | 59.7248 | 58.9101 | 59.3574 | 1.0497 |
| persistent6 | — | 49.7946 | 49.8991 | 48.9039 | 1.3851 |

## 2. Reconstruction and integrity

The maxima below are collected across every batch and condition, including absolute and relative source-sum errors when recorded. Exact original readout, fixed current input/query, unchanged weights and RNG checks are saved in SUMMARY/per-condition JSON.

| Check | Maximum error |
|---|---:|
| base_sum_max_abs_error | 1.90734863e-06 |
| base_sum_max_relative_error | 1.69663934e-07 |
| gap_sum_max_abs_error | 1.90734863e-06 |
| gap_sum_max_relative_error | 1.91424633e-07 |
| memory_sum_max_abs_error | 1.90734863e-06 |
| memory_sum_max_relative_error | 3.02101142e-07 |
| original_readout_reproduction_max_abs_error | 0 |
| prewrite_memory_sum_max_abs_error | 1.90734863e-06 |
| prewrite_memory_sum_max_relative_error | 2.94589455e-07 |
| source_sum_max_abs_error | 1.90734863e-06 |

## 3. Saved Value-source state strength

These are source-state Frobenius norms of the decayed, strictly prewrite history at each target. They quantify algebraic contribution magnitude, not emotional utility; source norms do not add to the joint norm.

| Condition | Source A norm | Source T norm | Source V norm | Joint norm |
|---|---:|---:|---:|---:|
| random 0.0 | 31.8035 | 32.2837 | 35.4569 | 57.0678 |
| random 0.1 | 30.6791 | 31.2631 | 34.2815 | 55.2518 |
| random 0.2 | 29.6279 | 30.1259 | 32.8488 | 53.2919 |
| random 0.3 | 28.6418 | 29.2204 | 31.5792 | 51.5694 |
| random 0.4 | 27.3315 | 27.4339 | 29.2485 | 48.5773 |
| random 0.5 | 25.9164 | 26.5343 | 27.8070 | 46.6302 |
| random 0.6 | 23.6938 | 23.6408 | 25.2584 | 42.4621 |
| random 0.7 | 21.9667 | 22.3319 | 23.3310 | 39.7829 |
| persistent A | 32.6471 | 0.0000 | 0.0000 | 32.6471 |
| persistent T | 0.0000 | 35.6080 | 0.0000 | 35.6080 |
| persistent V | 0.0000 | 0.0000 | 33.7007 | 33.7007 |
| persistent AT | 32.2963 | 33.6201 | 0.0000 | 46.3025 |
| persistent AV | 33.1778 | 0.0000 | 35.0116 | 48.3590 |
| persistent TV | 0.0000 | 34.6172 | 34.4959 | 48.8665 |
| persistent ATV | 31.8035 | 32.2837 | 35.4569 | 57.0678 |

## 4. Read strength and frozen decision contribution

Every condition has two 4×3 tables. Rows are read destinations; columns are historical Value-write sources. Numbers are not pure modality semantics. Active Base includes zero-history first turns. A currently visible modality has an inactive Gap: N/A, not zero importance.

Read norms and removal effects use exactly the same active-row cohort. ΔCE = CE_removed − CE_full: positive indicates beneficial presence under this intervention. ACC/UA/W-F1 deltas and corrections/harms appear in SLOTS.csv. A removal correction is full-wrong → removed-correct; a removal harm is full-correct → removed-wrong.

### random 0.0

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 9.9941 (1623) | 12.5454 (1623) | 12.6686 (1623) |
| Gap-A | N/A (0) | N/A (0) | N/A (0) |
| Gap-T | N/A (0) | N/A (0) | N/A (0) |
| Gap-V | N/A (0) | N/A (0) | N/A (0) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.066956 | 0.186017 | 0.046146 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.4032 |
| Gap-A | 0 | N/A |
| Gap-T | 0 | N/A |
| Gap-V | 0 | N/A |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.154697 | 0.266835 | 0.113624 | 0.535155 | 0.000000000000 |

### random 0.1

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 9.7143 (1623) | 12.2738 (1623) | 12.2940 (1623) |
| Gap-A | 10.4100 (171) | 9.1237 (171) | 9.2595 (171) |
| Gap-T | 7.5902 (160) | 11.6945 (160) | 9.6919 (160) |
| Gap-V | 7.1871 (151) | 9.2731 (151) | 12.2104 (151) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.061728 | 0.169409 | 0.041283 |
| Gap-A | 0.074727 | -0.005475 | -0.016008 |
| Gap-T | 0.007636 | 0.044868 | -0.019613 |
| Gap-V | 0.000258 | 0.021860 | 0.003593 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.4023 |
| Gap-A | 169 | 0.4034 |
| Gap-T | 158 | 0.4027 |
| Gap-V | 150 | 0.3949 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.184051 | 0.286779 | 0.121419 | 0.592249 | 0.000000000000 |

### random 0.2

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 9.5672 (1623) | 11.9670 (1623) | 11.9548 (1623) |
| Gap-A | 10.0349 (325) | 8.9793 (325) | 9.1455 (325) |
| Gap-T | 7.5200 (313) | 11.1933 (313) | 9.6726 (313) |
| Gap-V | 6.7788 (313) | 9.2797 (313) | 11.6236 (313) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.044097 | 0.121055 | 0.029469 |
| Gap-A | 0.075305 | 0.005836 | -0.012923 |
| Gap-T | -0.002065 | 0.053435 | -0.023981 |
| Gap-V | 0.003432 | 0.011734 | -0.008275 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.3992 |
| Gap-A | 320 | 0.4012 |
| Gap-T | 310 | 0.3973 |
| Gap-V | 306 | 0.3911 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.187864 | 0.288184 | 0.120149 | 0.596197 | 0.000000000000 |

### random 0.3

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 9.2750 (1623) | 11.6866 (1623) | 11.5416 (1623) |
| Gap-A | 9.7895 (516) | 9.0966 (516) | 9.0729 (516) |
| Gap-T | 7.4427 (472) | 10.9171 (472) | 9.1457 (472) |
| Gap-V | 6.9343 (476) | 9.2484 (476) | 11.3419 (476) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.050230 | 0.106374 | 0.015391 |
| Gap-A | 0.041994 | -0.004254 | -0.014894 |
| Gap-T | 0.011509 | 0.005307 | -0.008956 |
| Gap-V | 0.003496 | 0.012498 | -0.004037 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.4002 |
| Gap-A | 507 | 0.4028 |
| Gap-T | 463 | 0.3996 |
| Gap-V | 465 | 0.3934 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.214885 | 0.279470 | 0.125885 | 0.620240 | 0.000000000000 |

### random 0.4

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 8.9404 (1623) | 11.0397 (1623) | 10.8127 (1623) |
| Gap-A | 9.3481 (654) | 8.5780 (654) | 8.5701 (654) |
| Gap-T | 7.0726 (639) | 10.1688 (639) | 8.5983 (639) |
| Gap-V | 6.6561 (667) | 8.6477 (667) | 10.4096 (667) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.020365 | 0.106876 | 0.009281 |
| Gap-A | 0.031186 | 0.022203 | -0.019173 |
| Gap-T | -0.011232 | 0.026223 | -0.001173 |
| Gap-V | 0.001379 | 0.012871 | 0.009711 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.3961 |
| Gap-A | 642 | 0.3987 |
| Gap-T | 623 | 0.3926 |
| Gap-V | 654 | 0.3884 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.217179 | 0.345857 | 0.134239 | 0.697276 | 0.000000000000 |

### random 0.5

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 8.5673 (1623) | 11.0338 (1623) | 10.3554 (1623) |
| Gap-A | 9.0280 (825) | 8.7373 (825) | 8.3352 (825) |
| Gap-T | 7.1489 (775) | 10.3889 (775) | 8.3676 (775) |
| Gap-V | 6.3947 (809) | 8.6822 (809) | 10.0951 (809) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.035346 | 0.112271 | -0.004607 |
| Gap-A | 0.043332 | 0.016623 | -0.014215 |
| Gap-T | 0.007897 | 0.041822 | -0.013900 |
| Gap-V | 0.000607 | 0.009851 | -0.009757 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.3914 |
| Gap-A | 805 | 0.3905 |
| Gap-T | 759 | 0.3868 |
| Gap-V | 794 | 0.3827 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.259497 | 0.372699 | 0.118968 | 0.751165 | 0.000000000000 |

### random 0.6

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 7.7330 (1623) | 10.0264 (1623) | 9.4181 (1623) |
| Gap-A | 8.1485 (987) | 8.0578 (987) | 7.8449 (987) |
| Gap-T | 6.3539 (976) | 9.3493 (976) | 7.6874 (976) |
| Gap-V | 5.9632 (984) | 8.0175 (984) | 9.0585 (984) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.002218 | 0.084692 | -0.008944 |
| Gap-A | 0.034745 | 0.008573 | -0.009019 |
| Gap-T | -0.011772 | 0.027299 | -0.006935 |
| Gap-V | -0.001415 | 0.008817 | -0.009584 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.3832 |
| Gap-A | 971 | 0.3867 |
| Gap-T | 953 | 0.3786 |
| Gap-V | 967 | 0.3779 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.224932 | 0.384070 | 0.154961 | 0.763963 | 0.000000000000 |

### random 0.7

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 7.1136 (1623) | 9.4887 (1623) | 8.8714 (1623) |
| Gap-A | 7.3221 (1081) | 7.6448 (1081) | 7.3909 (1081) |
| Gap-T | 5.9492 (1065) | 8.7991 (1065) | 7.4006 (1065) |
| Gap-V | 5.4101 (1100) | 7.7330 (1100) | 8.6765 (1100) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.039451 | 0.048018 | -0.026328 |
| Gap-A | 0.041644 | -0.000904 | -0.023528 |
| Gap-T | 0.005737 | 0.022583 | -0.022695 |
| Gap-V | 0.012593 | 0.000857 | -0.018117 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.3755 |
| Gap-A | 1059 | 0.3740 |
| Gap-T | 1042 | 0.3684 |
| Gap-V | 1083 | 0.3677 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.304107 | 0.313185 | 0.103826 | 0.721118 | 0.000000000000 |

### persistent A

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 10.8991 (1623) | 0.0000 (1623) | 0.0000 (1623) |
| Gap-A | N/A (0) | N/A (0) | N/A (0) |
| Gap-T | 7.7232 (1623) | 0.0000 (1623) | 0.0000 (1623) |
| Gap-V | 7.5180 (1623) | 0.0000 (1623) | 0.0000 (1623) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | -0.050952 | 0.000000 | 0.000000 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | -0.007961 | 0.000000 | 0.000000 |
| Gap-V | -0.056124 | 0.000000 | 0.000000 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.0000 |
| Gap-A | 0 | N/A |
| Gap-T | 1592 | 0.0000 |
| Gap-V | 1592 | 0.0000 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.217659 | 0.000000 | 0.000000 | 0.217659 | 0.000000000000 |

### persistent T

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 (1623) | 13.3124 (1623) | 0.0000 (1623) |
| Gap-A | 0.0000 (1623) | 10.5506 (1623) | 0.0000 (1623) |
| Gap-T | N/A (0) | N/A (0) | N/A (0) |
| Gap-V | 0.0000 (1623) | 10.5113 (1623) | 0.0000 (1623) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.000000 | 0.199570 | 0.000000 |
| Gap-A | 0.000000 | 0.028927 | 0.000000 |
| Gap-T | N/A | N/A | N/A |
| Gap-V | 0.000000 | 0.010905 | 0.000000 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.0000 |
| Gap-A | 1592 | 0.0000 |
| Gap-T | 0 | N/A |
| Gap-V | 1592 | 0.0000 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.000000 | 0.643878 | -0.000000 | 0.643878 | 0.000000000000 |

### persistent V

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 (1623) | 0.0000 (1623) | 12.0427 (1623) |
| Gap-A | 0.0000 (1623) | 0.0000 (1623) | 9.3576 (1623) |
| Gap-T | 0.0000 (1623) | 0.0000 (1623) | 9.2516 (1623) |
| Gap-V | N/A (0) | N/A (0) | N/A (0) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.000000 | 0.000000 | -0.058050 |
| Gap-A | 0.000000 | 0.000000 | -0.081143 |
| Gap-T | 0.000000 | 0.000000 | -0.008603 |
| Gap-V | N/A | N/A | N/A |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.0000 |
| Gap-A | 1592 | 0.0000 |
| Gap-T | 1592 | 0.0000 |
| Gap-V | 0 | N/A |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.000000 | -0.000000 | 0.033090 | 0.033090 | 0.000000000000 |

### persistent AT

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 9.8535 (1623) | 12.8970 (1623) | 0.0000 (1623) |
| Gap-A | N/A (0) | N/A (0) | N/A (0) |
| Gap-T | N/A (0) | N/A (0) | N/A (0) |
| Gap-V | 6.9497 (1623) | 10.0037 (1623) | 0.0000 (1623) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.033850 | 0.153481 | 0.000000 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | 0.017645 | 0.001038 | 0.000000 |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.2808 |
| Gap-A | 0 | N/A |
| Gap-T | 0 | N/A |
| Gap-V | 1592 | 0.2872 |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.158593 | 0.324842 | 0.000000 | 0.483435 | 0.000000000000 |

### persistent AV

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 11.9162 (1623) | 0.0000 (1623) | 13.2416 (1623) |
| Gap-A | N/A (0) | N/A (0) | N/A (0) |
| Gap-T | 8.6490 (1623) | 0.0000 (1623) | 9.8415 (1623) |
| Gap-V | N/A (0) | N/A (0) | N/A (0) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.063278 | 0.000000 | -0.020223 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | 0.003127 | 0.000000 | -0.022924 |
| Gap-V | N/A | N/A | N/A |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.2825 |
| Gap-A | 0 | N/A |
| Gap-T | 1592 | 0.2777 |
| Gap-V | 0 | N/A |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.183713 | -0.000000 | 0.070337 | 0.254050 | 0.000000000000 |

### persistent TV

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 (1623) | 13.0174 (1623) | 11.8842 (1623) |
| Gap-A | 0.0000 (1623) | 9.8210 (1623) | 8.6925 (1623) |
| Gap-T | N/A (0) | N/A (0) | N/A (0) |
| Gap-V | N/A (0) | N/A (0) | N/A (0) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.000000 | 0.177367 | -0.003761 |
| Gap-A | 0.000000 | 0.033031 | -0.038028 |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.2764 |
| Gap-A | 1592 | 0.2859 |
| Gap-T | 0 | N/A |
| Gap-V | 0 | N/A |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| -0.000000 | 0.441518 | 0.090710 | 0.532228 | 0.000000000000 |

### persistent ATV

Mean source-read L2 norm (active cohort N in parentheses).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 9.9941 (1623) | 12.5454 (1623) | 12.6686 (1623) |
| Gap-A | N/A (0) | N/A (0) | N/A (0) |
| Gap-T | N/A (0) | N/A (0) | N/A (0) |
| Gap-V | N/A (0) | N/A (0) | N/A (0) |

Removal ΔCE (positive = beneficial presence in the fixed classifier).

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.066956 | 0.186017 | 0.046146 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. This is vector cancellation, not an emotion-conflict probability.

| Read destination | N | Mean cancellation |
|---|---:|---:|
| Base | 1592 | 0.4032 |
| Gap-A | 0 | N/A |
| Gap-T | 0 | N/A |
| Gap-V | 0 | N/A |

Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.

| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |
|---:|---:|---:|---:|---:|
| 0.154697 | 0.266835 | 0.113624 | 0.535155 | 0.000000000000 |

## 5. Full outputs and interpretation limits

- `CONDITIONS.csv`: all 15 full-model conditions, masks and resource checks.
- `SLOTS.csv`: 12 independent removals per condition, active N, read norms, ΔCE/ACC/UA/W-F1, corrections and harms.
- `SHAPLEY.csv`: all three source credits per condition, exact efficiency and distribution summaries.
- `SLOTS_GROUPED.csv`: full slot statistics stratified by current availability, strictly-past ever-seen pattern, per-source ever-seen and coverage. Empty/inactive slots remain blank numeric N/A.
- Per-condition JSON: all eight coalition scores, state norms, cancellation and strata; REPORT.json records aggregate metrics and source JSON hashes.

These measurements separate storage magnitude, original-query reads, and fixed-classifier marginal responses, but do not identify pure modality information, causal semantic relevance, or an optimal new module. Nonlinear independent slot effects are not additive. The three grouped Shapley credits sum to the negative-CE gain of full history reads over all-zero reads in this fixed game. Both full-zero history and partial subtraction can be out-of-distribution (OOD). Large norms, cancellation, or a single-seed metric gain alone do not demonstrate a model defect or a generally useful intervention.

Frozen inference seconds: 77.733; peak allocated GPU MiB: 675.662.
