# TASR finite frozen-backbone screen

**INTERNAL DIAGNOSTIC ONLY**

## Final assessment

All four jobs completed30/30epochs, exit0, one attempt each, no failures/retries.
Training source38251f4; read-only paired analysis ce3c055. Original backbone
digests, F1/F2 training-mask traces, validation masks, optimizer budgets, sample
identities and original-anchor predictions passed the complete paired audit.

- **Source decomposition:** algebraically additive and numerically reproduced
  within float32 roundoff. Across final evaluations, maximum Memory/Gap absolute
  error is1.90735e-6, maximum reported Frobenius-relative error5.22676e-7.
  This does not establish pure-modality semantics or causal attribution.
- **TASR vs same-capacity F1:** IEMOCAP6 fold5 Persistent-six ACC/UA/W-F1
  +0.503/+0.315/+0.528pp. Positive screening signal, not yet multi-seed/fivefold
  evidence; V-only deteriorates. MELD Persistent-six +0.051/+0.082/+0.074pp:
  numerically positive but too small to call a convincing improvement.
- **Random protection vs frozen F0:** MELD ACC/W-F1 −0.029/−0.006pp;
  IEMOCAP6 +0.092/+0.117pp. Both meet the predefined −0.3pp engineering guard.
  Random UA changes are +0.104pp(MELD) and −0.006pp(IEMOCAP6), not universal gains.
- **Persistent improvement vs F0:** IEMOCAP6 ACC/UA +0.555/+0.338pp; MELD
  +0.032/+0.054pp. Do not describe MELD as a resolved historical-use bottleneck.

No additional seeds, folds, network expansion, or hyperparameter search launched.
The full per-condition results and negative cases below remain primary evidence.

seed66 MELD and IEMOCAPSix fold5 only; no new full five-fold or multi-seed training

- Frozen original anchor miss=.3 was selected historically by Test-oracle.
- One new adapter checkpoint is selected using validation random8 ACC, then CE, and reused for every test condition.
- IEMOCAP adapter-only validation holdout was previously seen by the frozen backbone; not independent end-to-end validation.
- Source slots are additive joint Value-column contributions, not pure-modality semantic or causal attribution.
- No baseline ranking against historical per-rate selected checkpoints is implied.

F0 = same protected frozen Flat; F1 = source-neutral; F2 = real-source TASR.
Every entry uses the same anchor and condition mask. Cells: ACC / UA / W-F1 / CE.

## MELD

| Condition | F0 | F1 | F2 | F2−F1 ACC / UA / W-F1 |
|---|---:|---:|---:|---:|
| random 0.0 | 66.284 / 48.145 / 65.647 / 1.090 | 66.284 / 48.145 / 65.647 / 1.090 | 66.284 / 48.145 / 65.647 / 1.090 | +0.000 / +0.000 / +0.000 |
| random 0.1 | 65.249 / 46.041 / 64.071 / 1.118 | 65.249 / 46.240 / 64.085 / 1.118 | 65.287 / 46.291 / 64.125 / 1.117 | +0.038 / +0.051 / +0.040 |
| random 0.2 | 63.103 / 42.774 / 61.578 / 1.148 | 62.989 / 42.716 / 61.465 / 1.147 | 62.874 / 42.585 / 61.349 / 1.146 | -0.115 / -0.131 / -0.116 |
| random 0.3 | 62.261 / 40.207 / 59.996 / 1.178 | 62.222 / 40.220 / 59.981 / 1.177 | 62.146 / 40.472 / 59.951 / 1.176 | -0.077 / +0.252 / -0.030 |
| random 0.4 | 60.690 / 37.540 / 57.930 / 1.213 | 60.766 / 37.746 / 58.008 / 1.211 | 60.690 / 37.787 / 57.983 / 1.211 | -0.077 / +0.041 / -0.025 |
| random 0.5 | 59.464 / 33.843 / 55.700 / 1.204 | 59.540 / 34.034 / 55.777 / 1.202 | 59.540 / 34.064 / 55.788 / 1.201 | +0.000 / +0.030 / +0.011 |
| random 0.6 | 57.433 / 32.111 / 53.049 / 1.294 | 57.318 / 31.857 / 52.830 / 1.292 | 57.318 / 32.049 / 52.952 / 1.292 | +0.000 / +0.193 / +0.122 |
| random 0.7 | 56.207 / 30.221 / 51.173 / 1.311 | 56.437 / 30.344 / 51.382 / 1.309 | 56.322 / 30.325 / 51.303 / 1.308 | -0.115 / -0.019 / -0.079 |
| random mean | 61.336 / 38.861 / 58.643 / 1.194 | 61.351 / 38.913 / 58.647 / 1.193 | 61.307 / 38.965 / 58.637 / 1.193 | -0.043 / +0.052 / -0.010 |
| random high | 57.701 / 32.059 / 53.307 / 1.269 | 57.765 / 32.078 / 53.329 / 1.268 | 57.727 / 32.146 / 53.348 / 1.267 | -0.038 / +0.068 / +0.018 |
| persistent A | 48.889 / 19.641 / 39.976 / 1.455 | 48.889 / 19.641 / 39.973 / 1.454 | 49.004 / 19.771 / 40.121 / 1.454 | +0.115 / +0.130 / +0.148 |
| persistent T | 66.092 / 47.023 / 65.056 / 1.065 | 65.977 / 46.862 / 64.925 / 1.063 | 65.977 / 46.862 / 64.925 / 1.063 | +0.000 / +0.000 / +0.000 |
| persistent V | 48.046 / 14.263 / 31.235 / 1.517 | 48.046 / 14.263 / 31.235 / 1.517 | 48.084 / 14.299 / 31.319 / 1.517 | +0.038 / +0.036 / +0.084 |
| persistent AT | 67.011 / 46.868 / 65.648 / 1.059 | 67.011 / 46.898 / 65.659 / 1.057 | 67.011 / 46.938 / 65.662 / 1.056 | +0.000 / +0.039 / +0.003 |
| persistent AV | 49.119 / 17.715 / 37.972 / 1.461 | 49.119 / 17.818 / 38.122 / 1.459 | 49.234 / 17.925 / 38.285 / 1.458 | +0.115 / +0.107 / +0.163 |
| persistent TV | 66.475 / 46.734 / 65.215 / 1.056 | 66.475 / 46.595 / 65.161 / 1.053 | 66.513 / 46.775 / 65.205 / 1.053 | +0.038 / +0.180 / +0.044 |
| persistent ATV | 66.284 / 48.145 / 65.647 / 1.090 | 66.284 / 48.145 / 65.647 / 1.090 | 66.284 / 48.145 / 65.647 / 1.090 | +0.000 / +0.000 / +0.000 |
| persistent six | 57.605 / 32.041 / 50.851 / 1.269 | 57.586 / 32.013 / 50.846 / 1.267 | 57.637 / 32.095 / 50.919 / 1.267 | +0.051 / +0.082 / +0.074 |
| persistent seven | 58.845 / 34.341 / 52.964 / 1.243 | 58.829 / 34.317 / 52.960 / 1.242 | 58.872 / 34.388 / 53.023 / 1.242 | +0.044 / +0.070 / +0.063 |

### Paired corrections / harms

| Condition | F1 vs F0 | F2 vs F0 | F2 vs F1 | N |
|---|---:|---:|---:|---:|
| random 0.0 | 0 / 0 | 0 / 0 | 0 / 0 | 2610 |
| random 0.1 | 1 / 1 | 2 / 1 | 1 / 0 | 2610 |
| random 0.2 | 1 / 4 | 0 / 6 | 0 / 3 | 2610 |
| random 0.3 | 2 / 3 | 3 / 6 | 1 / 3 | 2610 |
| random 0.4 | 5 / 3 | 6 / 6 | 2 / 4 | 2610 |
| random 0.5 | 6 / 4 | 7 / 5 | 3 / 3 | 2610 |
| random 0.6 | 4 / 7 | 7 / 10 | 7 / 7 | 2610 |
| random 0.7 | 7 / 1 | 7 / 4 | 1 / 4 | 2610 |
| persistent A | 0 / 0 | 5 / 2 | 5 / 2 | 2610 |
| persistent T | 0 / 3 | 0 / 3 | 0 / 0 | 2610 |
| persistent V | 0 / 0 | 1 / 0 | 1 / 0 | 2610 |
| persistent AT | 2 / 2 | 3 / 3 | 1 / 1 | 2610 |
| persistent AV | 5 / 5 | 7 / 4 | 6 / 3 | 2610 |
| persistent TV | 3 / 3 | 5 / 4 | 2 / 1 | 2610 |
| persistent ATV | 0 / 0 | 0 / 0 | 0 / 0 | 2610 |

### Optimization and diagnostics

- F1: selected epoch 3/30; adapter parameters 59423; 990 optimizer steps; peak 267.9 MiB; 7877.8s.
- F1 actual random missing (epoch mean): 0.346255; Persistent pattern counts: {"A": 5172, "AT": 5319, "AV": 5130, "T": 5329, "TV": 5033, "V": 5157}.
- F1 source reconstruction maximum errors: {"gap_sum_max_abs_error": 2.980232238769531e-07, "gap_sum_max_relative_error": 5.226751795817108e-07, "memory_sum_max_abs_error": 5.364418029785156e-07, "memory_sum_max_relative_error": 2.623353623221192e-07}.
- F1 diagnostic status: complete; full gamma/read matrices, coverage and validation curves are in COMPARISON.json.
- F2: selected epoch 20/30; adapter parameters 59423; 990 optimizer steps; peak 267.7 MiB; 7875.9s.
- F2 actual random missing (epoch mean): 0.346255; Persistent pattern counts: {"A": 5172, "AT": 5319, "AV": 5130, "T": 5329, "TV": 5033, "V": 5157}.
- F2 source reconstruction maximum errors: {"gap_sum_max_abs_error": 2.980232238769531e-07, "gap_sum_max_relative_error": 5.226751795817108e-07, "memory_sum_max_abs_error": 5.364418029785156e-07, "memory_sum_max_relative_error": 2.623353623221192e-07}.
- F2 diagnostic status: complete; full gamma/read matrices, coverage and validation curves are in COMPARISON.json.

Random protection (F2 vs F0, ACC and W-F1 ≥−0.3pp): True.
Descriptive Persistent-six F2 extra value over F1 (ACC/UA/W-F1 non-decreasing, at least one positive): True.
This screen is not significance testing. Individual condition failures and trade-offs above must not be hidden.

## IEMOCAPSix

| Condition | F0 | F1 | F2 | F2−F1 ACC / UA / W-F1 |
|---|---:|---:|---:|---:|
| random 0.0 | 66.543 / 66.379 / 66.410 / 0.907 | 66.543 / 66.379 / 66.410 / 0.907 | 66.543 / 66.379 / 66.410 / 0.907 | +0.000 / +0.000 / +0.000 |
| random 0.1 | 65.989 / 65.609 / 65.787 / 0.912 | 65.989 / 65.609 / 65.783 / 0.911 | 65.927 / 65.511 / 65.725 / 0.911 | -0.062 / -0.098 / -0.059 |
| random 0.2 | 62.908 / 63.066 / 62.704 / 0.958 | 62.785 / 62.912 / 62.593 / 0.958 | 62.970 / 63.043 / 62.787 / 0.954 | +0.185 / +0.131 / +0.195 |
| random 0.3 | 64.325 / 63.390 / 64.127 / 0.930 | 64.202 / 63.278 / 64.011 / 0.930 | 64.757 / 63.730 / 64.577 / 0.923 | +0.555 / +0.452 / +0.566 |
| random 0.4 | 61.984 / 61.564 / 61.964 / 0.967 | 62.046 / 61.632 / 62.014 / 0.966 | 61.984 / 61.448 / 61.987 / 0.963 | -0.062 / -0.184 / -0.027 |
| random 0.5 | 61.491 / 60.475 / 61.018 / 0.986 | 61.614 / 60.635 / 61.154 / 0.986 | 61.368 / 60.188 / 60.950 / 0.981 | -0.246 / -0.447 / -0.204 |
| random 0.6 | 60.752 / 61.194 / 60.529 / 1.017 | 60.875 / 61.269 / 60.665 / 1.017 | 61.060 / 61.385 / 60.886 / 1.008 | +0.185 / +0.117 / +0.221 |
| random 0.7 | 56.932 / 55.061 / 56.526 / 1.145 | 56.993 / 55.177 / 56.610 / 1.144 | 57.055 / 55.001 / 56.678 / 1.137 | +0.062 / -0.176 / +0.067 |
| random mean | 62.616 / 62.092 / 62.383 / 0.978 | 62.631 / 62.111 / 62.405 / 0.978 | 62.708 / 62.086 / 62.500 / 0.973 | +0.077 / -0.026 / +0.095 |
| random high | 59.725 / 58.910 / 59.357 / 1.050 | 59.827 / 59.027 / 59.477 / 1.049 | 59.827 / 58.858 / 59.505 / 1.042 | +0.000 / -0.169 / +0.028 |
| persistent A | 42.144 / 44.682 / 40.582 / 1.474 | 42.144 / 44.645 / 40.601 / 1.471 | 43.007 / 45.235 / 41.552 / 1.454 | +0.863 / +0.590 / +0.951 |
| persistent T | 53.481 / 53.355 / 53.336 / 1.416 | 53.481 / 53.300 / 53.326 / 1.415 | 53.789 / 53.537 / 53.630 / 1.405 | +0.308 / +0.237 / +0.305 |
| persistent V | 34.442 / 30.973 / 32.399 / 1.901 | 34.319 / 30.873 / 32.207 / 1.896 | 34.196 / 30.561 / 31.882 / 1.872 | -0.123 / -0.312 / -0.324 |
| persistent AT | 61.861 / 61.653 / 61.203 / 1.065 | 61.922 / 61.697 / 61.266 / 1.065 | 62.107 / 61.713 / 61.488 / 1.064 | +0.185 / +0.015 / +0.222 |
| persistent AV | 49.661 / 52.835 / 48.916 / 1.279 | 49.846 / 52.977 / 49.122 / 1.277 | 50.893 / 53.699 / 50.318 / 1.268 | +1.047 / +0.722 / +1.196 |
| persistent TV | 57.178 / 55.897 / 56.988 / 1.176 | 57.363 / 56.040 / 57.161 / 1.175 | 58.102 / 56.678 / 57.980 / 1.163 | +0.739 / +0.638 / +0.820 |
| persistent ATV | 66.543 / 66.379 / 66.410 / 0.907 | 66.543 / 66.379 / 66.410 / 0.907 | 66.543 / 66.379 / 66.410 / 0.907 | +0.000 / +0.000 / +0.000 |
| persistent six | 49.795 / 49.899 / 48.904 / 1.385 | 49.846 / 49.922 / 48.947 / 1.383 | 50.349 / 50.237 / 49.475 / 1.371 | +0.503 / +0.315 / +0.528 |
| persistent seven | 52.187 / 52.253 / 51.405 / 1.317 | 52.231 / 52.273 / 51.442 / 1.315 | 52.663 / 52.543 / 51.894 / 1.305 | +0.431 / +0.270 / +0.453 |

### Paired corrections / harms

| Condition | F1 vs F0 | F2 vs F0 | F2 vs F1 | N |
|---|---:|---:|---:|---:|
| random 0.0 | 0 / 0 | 0 / 0 | 0 / 0 | 1623 |
| random 0.1 | 0 / 0 | 2 / 3 | 2 / 3 | 1623 |
| random 0.2 | 0 / 2 | 5 / 4 | 6 / 3 | 1623 |
| random 0.3 | 0 / 2 | 10 / 3 | 11 / 2 | 1623 |
| random 0.4 | 3 / 2 | 11 / 11 | 11 / 12 | 1623 |
| random 0.5 | 2 / 0 | 12 / 14 | 10 / 14 | 1623 |
| random 0.6 | 4 / 2 | 18 / 13 | 18 / 15 | 1623 |
| random 0.7 | 1 / 0 | 17 / 15 | 16 / 15 | 1623 |
| persistent A | 2 / 2 | 25 / 11 | 23 / 9 | 1623 |
| persistent T | 2 / 2 | 14 / 9 | 13 / 8 | 1623 |
| persistent V | 3 / 5 | 16 / 20 | 13 / 15 | 1623 |
| persistent AT | 1 / 0 | 9 / 5 | 8 / 5 | 1623 |
| persistent AV | 4 / 1 | 26 / 6 | 23 / 6 | 1623 |
| persistent TV | 3 / 0 | 18 / 3 | 17 / 5 | 1623 |
| persistent ATV | 0 / 0 | 0 / 0 | 0 / 0 | 1623 |

### Optimization and diagnostics

- F1: selected epoch 5/30; adapter parameters 59423; 120 optimizer steps; peak 979.8 MiB; 720.2s.
- F1 actual random missing (epoch mean): 0.349580; Persistent pattern counts: {"A": 507, "AT": 518, "AV": 541, "T": 584, "TV": 581, "V": 509}.
- F1 source reconstruction maximum errors: {"gap_sum_max_abs_error": 1.9073486328125e-06, "gap_sum_max_relative_error": 1.9142463258958742e-07, "memory_sum_max_abs_error": 1.9073486328125e-06, "memory_sum_max_relative_error": 3.0210114232431806e-07}.
- F1 diagnostic status: complete; full gamma/read matrices, coverage and validation curves are in COMPARISON.json.
- F2: selected epoch 26/30; adapter parameters 59423; 120 optimizer steps; peak 980.4 MiB; 720.0s.
- F2 actual random missing (epoch mean): 0.349580; Persistent pattern counts: {"A": 507, "AT": 518, "AV": 541, "T": 584, "TV": 581, "V": 509}.
- F2 source reconstruction maximum errors: {"gap_sum_max_abs_error": 1.9073486328125e-06, "gap_sum_max_relative_error": 1.9142463258958742e-07, "memory_sum_max_abs_error": 1.9073486328125e-06, "memory_sum_max_relative_error": 3.0210114232431806e-07}.
- F2 diagnostic status: complete; full gamma/read matrices, coverage and validation curves are in COMPARISON.json.

Random protection (F2 vs F0, ACC and W-F1 ≥−0.3pp): True.
Descriptive Persistent-six F2 extra value over F1 (ACC/UA/W-F1 non-decreasing, at least one positive): True.
This screen is not significance testing. Individual condition failures and trade-offs above must not be hidden.
