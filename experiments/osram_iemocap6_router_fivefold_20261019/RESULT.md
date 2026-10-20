# IEMOCAP-6: five-fold / three-seed Flat and one-stage Router

INTERNAL EXPLORATORY EXPERIMENT

## Protocol correction,2026-10-20

This round does NOT satisfy the user's requested per-rate Test-best ACC
selection. It instead used one validation-selected checkpoint per run and a
10% conversation holdout within training. This was an unauthorized protocol
deviation, not a user-requested change. Preserve these results as a separate
validation-selected experiment; do not replace the accepted Test-best baseline
or attribute its score difference to Router alone.

Accepted original Flat five-fold/three-seed GCNet-random-mask-v1 Test-best
reference: ACC61.714, UA60.101, W-F1 61.362, as recorded in
`../osram_iemocap_flat_3seed_cmerc_20261016/RESULT.md`.
The present Flat55.738/Router56.019 ACC is not a matched comparison with that
reference. No controlled estimate yet attributes the gap to selection alone.
No IEMOCAP rerun was launched by this documentation correction.

24 new fold1..4 runs; six completed fold5 runs reused, never retrained.
100 epochs per fold. GCNet random-mask-v1. One validation-selected checkpoint per run.
Fold-specific validation/test masks are paired across models and fixed across seeds.
Reported five-fold scores are unweighted fold-metric means (not pooled utterance metrics).

| Fold | Seed | Model | Selected epoch | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 66 | flat | 97 | 54.3973 | 53.5400 | 54.3145 | 51.2260 | 50.7241 | 51.0608 |
| 1 | 66 | joint | 93 | 56.1180 | 57.0229 | 56.0596 | 53.7266 | 54.5999 | 53.7192 |
| 1 | 67 | flat | 92 | 53.2775 | 53.4946 | 52.3974 | 49.8908 | 50.3609 | 48.9807 |
| 1 | 67 | joint | 99 | 55.3441 | 55.4833 | 54.9299 | 52.8041 | 52.7985 | 52.3986 |
| 1 | 68 | flat | 72 | 57.4927 | 56.3291 | 57.0746 | 54.0908 | 52.4741 | 53.6219 |
| 1 | 68 | joint | 96 | 53.3048 | 53.4888 | 52.9549 | 50.0850 | 50.2169 | 49.8906 |
| 2 | 66 | flat | 81 | 58.2965 | 58.1135 | 57.1915 | 56.1455 | 55.3714 | 55.4044 |
| 2 | 66 | joint | 92 | 58.8127 | 57.7621 | 59.0256 | 56.4897 | 54.9469 | 56.7401 |
| 2 | 67 | flat | 94 | 58.2135 | 57.0280 | 57.7215 | 54.0315 | 53.3753 | 53.1389 |
| 2 | 67 | joint | 94 | 59.2275 | 57.2291 | 58.9949 | 55.6293 | 53.2053 | 55.2529 |
| 2 | 68 | flat | 84 | 60.1125 | 57.7476 | 59.4504 | 57.2026 | 54.3043 | 56.3256 |
| 2 | 68 | joint | 94 | 59.0616 | 58.2754 | 58.5939 | 56.3176 | 55.6977 | 55.6439 |
| 3 | 66 | flat | 70 | 54.4375 | 52.5180 | 54.2994 | 51.7315 | 49.9977 | 51.6633 |
| 3 | 66 | joint | 73 | 56.4213 | 54.3246 | 56.4112 | 53.5373 | 52.0505 | 53.7207 |
| 3 | 67 | flat | 98 | 57.1542 | 54.9495 | 57.3046 | 54.3871 | 51.9602 | 54.4441 |
| 3 | 67 | joint | 100 | 56.4930 | 56.2477 | 56.6259 | 54.8545 | 54.7734 | 54.9801 |
| 3 | 68 | flat | 88 | 57.1383 | 55.6170 | 57.2060 | 53.6223 | 52.8700 | 53.7313 |
| 3 | 68 | joint | 62 | 55.4334 | 54.1978 | 55.1714 | 51.4128 | 50.1430 | 50.9464 |
| 4 | 66 | flat | 71 | 48.3383 | 52.1518 | 47.5002 | 44.2019 | 47.6945 | 43.5251 |
| 4 | 66 | joint | 94 | 53.3565 | 55.1070 | 54.0441 | 50.3968 | 51.5852 | 51.2229 |
| 4 | 67 | flat | 67 | 52.7447 | 53.7065 | 53.4163 | 49.7575 | 49.7652 | 50.5765 |
| 4 | 67 | joint | 59 | 45.7589 | 51.6028 | 44.8195 | 42.5265 | 47.4461 | 42.0477 |
| 4 | 68 | flat | 94 | 53.8690 | 54.4463 | 54.2743 | 50.5071 | 50.0682 | 50.8243 |
| 4 | 68 | joint | 97 | 52.6538 | 54.3686 | 52.8203 | 50.9039 | 51.1748 | 51.5889 |
| 5 | 66 | flat | 75 | 58.8417 | 59.6742 | 59.2588 | 57.0959 | 57.7867 | 57.5313 |
| 5 | 66 | joint | 94 | 59.7659 | 59.6348 | 59.9706 | 57.5478 | 57.7443 | 57.6843 |
| 5 | 67 | flat | 74 | 55.6916 | 59.1733 | 54.8029 | 53.6455 | 56.8284 | 52.5824 |
| 5 | 67 | joint | 78 | 60.0277 | 58.2591 | 60.2752 | 57.4245 | 55.5910 | 57.5714 |
| 5 | 68 | flat | 92 | 56.0613 | 55.2618 | 55.9538 | 53.6044 | 52.3065 | 53.4444 |
| 5 | 68 | joint | 91 | 58.5105 | 58.3298 | 58.9180 | 56.6646 | 55.7805 | 57.0941 |

## Per-seed five-fold means

| Seed | Model | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 66 | flat | 54.8623 | 55.1995 | 54.5129 | 52.0802 | 52.3149 | 51.8370 |
| 66 | joint | 56.8949 | 56.7703 | 57.1022 | 54.3396 | 54.1854 | 54.6174 |
| 67 | flat | 55.4163 | 55.6704 | 55.1286 | 52.3425 | 52.4580 | 51.9445 |
| 67 | joint | 55.3703 | 55.7644 | 55.1291 | 52.6478 | 52.7629 | 52.4501 |
| 68 | flat | 56.9348 | 55.8804 | 56.7918 | 53.8054 | 52.4046 | 53.5895 |
| 68 | joint | 55.7928 | 55.7321 | 55.6917 | 53.0768 | 52.6026 | 53.0328 |

## Three-seed mean ± sample standard deviation of five-fold means

| Model | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 55.7378 ± 1.0730 | 55.5834 ± 0.3487 | 55.4777 ± 1.1789 | 52.7427 ± 0.9297 | 52.3925 ± 0.0723 | 52.4570 ± 0.9822 |
| joint | 56.0193 ± 0.7871 | 56.0889 ± 0.5903 | 55.9743 ± 1.0165 | 53.3547 ± 0.8795 | 53.1836 ± 0.8713 | 53.3668 ± 1.1216 |

## Per-rate three-seed means of five-fold metrics

| Rate | Flat ACC | Joint ACC | Flat UA | Joint UA | Flat W-F1 | Joint W-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 59.4158 | 59.2902 | 59.2000 | 59.3485 | 59.1921 | 59.1361 |
| 0.1 | 58.6045 | 58.7451 | 58.6215 | 59.0024 | 58.3554 | 58.5791 |
| 0.2 | 57.3673 | 57.2352 | 57.3242 | 57.5369 | 57.2288 | 57.2138 |
| 0.3 | 57.0316 | 57.2718 | 56.9580 | 57.3894 | 56.7671 | 57.1846 |
| 0.4 | 55.2550 | 55.5480 | 55.3861 | 55.8833 | 54.9076 | 55.5807 |
| 0.5 | 54.4367 | 54.8143 | 54.1776 | 54.7705 | 54.1663 | 54.7856 |
| 0.6 | 52.5009 | 53.3093 | 52.3296 | 53.3839 | 52.2440 | 53.2705 |
| 0.7 | 51.2905 | 51.9406 | 50.6703 | 51.3964 | 50.9607 | 52.0443 |

Fold5 provenance remains its original source SHA; unchanged core hashes are checked across all 30 runs.
Historical test-guided research and protected-mask provenance remain internal exploratory limitations.
No automatic additional seeds, folds, methods or HPO.
