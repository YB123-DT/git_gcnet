# IEMOCAP-6 seed68 frozen Signature path audit

INTERNAL Test-oracle mechanism sensitivity only. No training, new selection or resampled test masks.
ACC and UA are both reported; UA is macro recall over fixed classes0..5, including class0.
Primary means weight each fold/rate cell equally, not each utterance. Empty cohorts are N/A.

| Fold | Rate | Cohort | Mode | N | ACC | UA | W-F1 | Corrections / harms vs normal |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.0 | all | flat | 1373 | 63.7291 | 63.2367 | 63.3065 | 55 / 66 |
| 1 | 0.0 | all | normal | 1373 | 64.5302 | 64.1343 | 64.3469 | 0 / 0 |
| 1 | 0.0 | all | degree2_no_area | 1373 | 64.0204 | 63.4176 | 63.8127 | 16 / 23 |
| 1 | 0.0 | all | interior_shuffle_6601 | 1373 | 64.4574 | 63.9141 | 64.2229 | 18 / 19 |
| 1 | 0.0 | all | interior_shuffle_6602 | 1373 | 63.5834 | 63.0870 | 63.3939 | 13 / 26 |
| 1 | 0.0 | all | interior_shuffle_6603 | 1373 | 63.8019 | 63.3503 | 63.5890 | 11 / 21 |
| 1 | 0.1 | all | flat | 1373 | 63.3649 | 62.5330 | 62.9291 | 0 / 10 |
| 1 | 0.1 | all | normal | 1373 | 64.0932 | 63.1794 | 63.7056 | 0 / 0 |
| 1 | 0.1 | all | degree2_no_area | 1373 | 64.0204 | 63.1360 | 63.6353 | 0 / 1 |
| 1 | 0.1 | all | interior_shuffle_6601 | 1373 | 64.0932 | 63.1794 | 63.7056 | 0 / 0 |
| 1 | 0.1 | all | interior_shuffle_6602 | 1373 | 64.0932 | 63.1794 | 63.7056 | 0 / 0 |
| 1 | 0.1 | all | interior_shuffle_6603 | 1373 | 64.0932 | 63.1794 | 63.7056 | 0 / 0 |
| 1 | 0.2 | all | flat | 1373 | 63.5106 | 62.7191 | 63.2226 | 18 / 32 |
| 1 | 0.2 | all | normal | 1373 | 64.5302 | 63.9984 | 64.2903 | 0 / 0 |
| 1 | 0.2 | all | degree2_no_area | 1373 | 64.5302 | 63.9984 | 64.2903 | 0 / 0 |
| 1 | 0.2 | all | interior_shuffle_6601 | 1373 | 64.5302 | 63.9984 | 64.2903 | 0 / 0 |
| 1 | 0.2 | all | interior_shuffle_6602 | 1373 | 64.5302 | 63.9984 | 64.2903 | 0 / 0 |
| 1 | 0.2 | all | interior_shuffle_6603 | 1373 | 64.5302 | 63.9984 | 64.2903 | 0 / 0 |
| 1 | 0.3 | all | flat | 1373 | 62.2724 | 63.1049 | 61.7870 | 8 / 9 |
| 1 | 0.3 | all | normal | 1373 | 62.3452 | 63.0222 | 61.9797 | 0 / 0 |
| 1 | 0.3 | all | degree2_no_area | 1373 | 62.3452 | 63.0222 | 61.9797 | 0 / 0 |
| 1 | 0.3 | all | interior_shuffle_6601 | 1373 | 62.3452 | 63.0222 | 61.9797 | 0 / 0 |
| 1 | 0.3 | all | interior_shuffle_6602 | 1373 | 62.3452 | 63.0222 | 61.9797 | 0 / 0 |
| 1 | 0.3 | all | interior_shuffle_6603 | 1373 | 62.3452 | 63.0222 | 61.9797 | 0 / 0 |
| 1 | 0.4 | all | flat | 1373 | 60.8886 | 59.0756 | 60.3914 | 52 / 56 |
| 1 | 0.4 | all | normal | 1373 | 61.1799 | 60.3429 | 60.7966 | 0 / 0 |
| 1 | 0.4 | all | degree2_no_area | 1373 | 61.1799 | 60.4598 | 60.7979 | 5 / 5 |
| 1 | 0.4 | all | interior_shuffle_6601 | 1373 | 60.9614 | 60.1367 | 60.5749 | 2 / 5 |
| 1 | 0.4 | all | interior_shuffle_6602 | 1373 | 61.1799 | 60.4557 | 60.7741 | 3 / 3 |
| 1 | 0.4 | all | interior_shuffle_6603 | 1373 | 61.1071 | 60.2995 | 60.7057 | 3 / 4 |
| 1 | 0.5 | all | flat | 1373 | 59.5776 | 58.1317 | 59.6262 | 54 / 62 |
| 1 | 0.5 | all | normal | 1373 | 60.1602 | 59.5264 | 60.3649 | 0 / 0 |
| 1 | 0.5 | all | degree2_no_area | 1373 | 59.9417 | 59.5275 | 60.1359 | 6 / 9 |
| 1 | 0.5 | all | interior_shuffle_6601 | 1373 | 59.7961 | 59.2621 | 60.0011 | 2 / 7 |
| 1 | 0.5 | all | interior_shuffle_6602 | 1373 | 59.6504 | 59.0728 | 59.8474 | 1 / 8 |
| 1 | 0.5 | all | interior_shuffle_6603 | 1373 | 59.8689 | 59.3787 | 60.0663 | 3 / 7 |
| 1 | 0.6 | all | flat | 1373 | 58.8492 | 59.7332 | 58.5530 | 59 / 73 |
| 1 | 0.6 | all | normal | 1373 | 59.8689 | 60.3478 | 59.6735 | 0 / 0 |
| 1 | 0.6 | all | degree2_no_area | 1373 | 59.1406 | 59.4785 | 59.0016 | 16 / 26 |
| 1 | 0.6 | all | interior_shuffle_6601 | 1373 | 59.5776 | 60.0450 | 59.4174 | 6 / 10 |
| 1 | 0.6 | all | interior_shuffle_6602 | 1373 | 59.5776 | 60.0186 | 59.4324 | 7 / 11 |
| 1 | 0.6 | all | interior_shuffle_6603 | 1373 | 59.5047 | 59.9561 | 59.3507 | 6 / 11 |
| 1 | 0.7 | all | flat | 1373 | 57.3926 | 59.3743 | 56.7863 | 14 / 19 |
| 1 | 0.7 | all | normal | 1373 | 57.7567 | 59.3654 | 57.3361 | 0 / 0 |
| 1 | 0.7 | all | degree2_no_area | 1373 | 57.7567 | 59.3654 | 57.3361 | 0 / 0 |
| 1 | 0.7 | all | interior_shuffle_6601 | 1373 | 57.7567 | 59.3654 | 57.3361 | 0 / 0 |
| 1 | 0.7 | all | interior_shuffle_6602 | 1373 | 57.7567 | 59.3654 | 57.3361 | 0 / 0 |
| 1 | 0.7 | all | interior_shuffle_6603 | 1373 | 57.7567 | 59.3654 | 57.3361 | 0 / 0 |
| 2 | 0.0 | all | flat | 1356 | 66.2979 | 64.5607 | 65.9999 | 44 / 57 |
| 2 | 0.0 | all | normal | 1356 | 67.2566 | 66.4016 | 67.4155 | 0 / 0 |
| 2 | 0.0 | all | degree2_no_area | 1356 | 66.0029 | 65.0243 | 66.1948 | 21 / 38 |
| 2 | 0.0 | all | interior_shuffle_6601 | 1356 | 66.2242 | 65.4038 | 66.3392 | 19 / 33 |
| 2 | 0.0 | all | interior_shuffle_6602 | 1356 | 66.5192 | 65.5025 | 66.6232 | 25 / 35 |
| 2 | 0.0 | all | interior_shuffle_6603 | 1356 | 66.0767 | 65.2911 | 66.1482 | 25 / 41 |
| 2 | 0.1 | all | flat | 1356 | 64.8968 | 64.9749 | 64.9707 | 25 / 34 |
| 2 | 0.1 | all | normal | 1356 | 65.5605 | 65.4142 | 65.7651 | 0 / 0 |
| 2 | 0.1 | all | degree2_no_area | 1356 | 65.5605 | 65.4142 | 65.7651 | 0 / 0 |
| 2 | 0.1 | all | interior_shuffle_6601 | 1356 | 65.5605 | 65.4142 | 65.7651 | 0 / 0 |
| 2 | 0.1 | all | interior_shuffle_6602 | 1356 | 65.5605 | 65.4142 | 65.7651 | 0 / 0 |
| 2 | 0.1 | all | interior_shuffle_6603 | 1356 | 65.5605 | 65.4142 | 65.7651 | 0 / 0 |
| 2 | 0.2 | all | flat | 1356 | 65.7817 | 62.8640 | 65.7055 | 3 / 0 |
| 2 | 0.2 | all | normal | 1356 | 65.5605 | 62.7101 | 65.4907 | 0 / 0 |
| 2 | 0.2 | all | degree2_no_area | 1356 | 65.5605 | 62.7101 | 65.4907 | 0 / 0 |
| 2 | 0.2 | all | interior_shuffle_6601 | 1356 | 65.5605 | 62.7101 | 65.4907 | 0 / 0 |
| 2 | 0.2 | all | interior_shuffle_6602 | 1356 | 65.5605 | 62.7101 | 65.4907 | 0 / 0 |
| 2 | 0.2 | all | interior_shuffle_6603 | 1356 | 65.5605 | 62.7101 | 65.4907 | 0 / 0 |
| 2 | 0.3 | all | flat | 1356 | 64.6018 | 64.3656 | 64.7070 | 3 / 2 |
| 2 | 0.3 | all | normal | 1356 | 64.5280 | 64.3566 | 64.6441 | 0 / 0 |
| 2 | 0.3 | all | degree2_no_area | 1356 | 64.5280 | 64.3566 | 64.6441 | 0 / 0 |
| 2 | 0.3 | all | interior_shuffle_6601 | 1356 | 64.5280 | 64.3566 | 64.6441 | 0 / 0 |
| 2 | 0.3 | all | interior_shuffle_6602 | 1356 | 64.5280 | 64.3566 | 64.6441 | 0 / 0 |
| 2 | 0.3 | all | interior_shuffle_6603 | 1356 | 64.5280 | 64.3566 | 64.6441 | 0 / 0 |
| 2 | 0.4 | all | flat | 1356 | 64.1593 | 64.9542 | 64.0183 | 0 / 1 |
| 2 | 0.4 | all | normal | 1356 | 64.2330 | 65.0335 | 64.0858 | 0 / 0 |
| 2 | 0.4 | all | degree2_no_area | 1356 | 64.2330 | 65.0335 | 64.0858 | 0 / 0 |
| 2 | 0.4 | all | interior_shuffle_6601 | 1356 | 64.2330 | 65.0335 | 64.0858 | 0 / 0 |
| 2 | 0.4 | all | interior_shuffle_6602 | 1356 | 64.2330 | 65.0335 | 64.0858 | 0 / 0 |
| 2 | 0.4 | all | interior_shuffle_6603 | 1356 | 64.2330 | 65.0335 | 64.0858 | 0 / 0 |
| 2 | 0.5 | all | flat | 1356 | 62.2419 | 60.1864 | 62.2442 | 43 / 53 |
| 2 | 0.5 | all | normal | 1356 | 62.9794 | 61.9789 | 63.2071 | 0 / 0 |
| 2 | 0.5 | all | degree2_no_area | 1356 | 62.2419 | 61.4383 | 62.4866 | 5 / 15 |
| 2 | 0.5 | all | interior_shuffle_6601 | 1356 | 62.8319 | 61.7942 | 63.0532 | 3 / 5 |
| 2 | 0.5 | all | interior_shuffle_6602 | 1356 | 62.8319 | 61.7070 | 63.0457 | 3 / 5 |
| 2 | 0.5 | all | interior_shuffle_6603 | 1356 | 62.9794 | 61.7451 | 63.1809 | 5 / 5 |
| 2 | 0.6 | all | flat | 1356 | 60.9882 | 58.8161 | 60.8824 | 35 / 34 |
| 2 | 0.6 | all | normal | 1356 | 60.9145 | 60.0402 | 60.9261 | 0 / 0 |
| 2 | 0.6 | all | degree2_no_area | 1356 | 60.9145 | 60.0402 | 60.9261 | 0 / 0 |
| 2 | 0.6 | all | interior_shuffle_6601 | 1356 | 60.9145 | 60.0402 | 60.9261 | 0 / 0 |
| 2 | 0.6 | all | interior_shuffle_6602 | 1356 | 60.9145 | 60.0402 | 60.9261 | 0 / 0 |
| 2 | 0.6 | all | interior_shuffle_6603 | 1356 | 60.9145 | 60.0402 | 60.9261 | 0 / 0 |
| 2 | 0.7 | all | flat | 1356 | 58.7758 | 57.0949 | 58.1522 | 54 / 67 |
| 2 | 0.7 | all | normal | 1356 | 59.7345 | 58.9986 | 59.5890 | 0 / 0 |
| 2 | 0.7 | all | degree2_no_area | 1356 | 60.1032 | 59.3950 | 59.9695 | 16 / 11 |
| 2 | 0.7 | all | interior_shuffle_6601 | 1356 | 59.6608 | 59.1245 | 59.5138 | 7 / 8 |
| 2 | 0.7 | all | interior_shuffle_6602 | 1356 | 59.6608 | 59.0489 | 59.5326 | 6 / 7 |
| 2 | 0.7 | all | interior_shuffle_6603 | 1356 | 59.6608 | 59.0489 | 59.5126 | 6 / 7 |
| 3 | 0.0 | all | flat | 1569 | 62.5876 | 61.7916 | 62.8938 | 8 / 11 |
| 3 | 0.0 | all | normal | 1569 | 62.7788 | 62.3385 | 63.1071 | 0 / 0 |
| 3 | 0.0 | all | degree2_no_area | 1569 | 62.7788 | 62.3385 | 63.1071 | 0 / 0 |
| 3 | 0.0 | all | interior_shuffle_6601 | 1569 | 62.7788 | 62.3385 | 63.1071 | 0 / 0 |
| 3 | 0.0 | all | interior_shuffle_6602 | 1569 | 62.7788 | 62.3385 | 63.1071 | 0 / 0 |
| 3 | 0.0 | all | interior_shuffle_6603 | 1569 | 62.7788 | 62.3385 | 63.1071 | 0 / 0 |
| 3 | 0.1 | all | flat | 1569 | 60.1657 | 59.6249 | 60.5160 | 32 / 65 |
| 3 | 0.1 | all | normal | 1569 | 62.2690 | 61.7437 | 62.6099 | 0 / 0 |
| 3 | 0.1 | all | degree2_no_area | 1569 | 62.2052 | 61.6742 | 62.5481 | 0 / 1 |
| 3 | 0.1 | all | interior_shuffle_6601 | 1569 | 62.2690 | 61.7437 | 62.6099 | 0 / 0 |
| 3 | 0.1 | all | interior_shuffle_6602 | 1569 | 62.2690 | 61.7437 | 62.6099 | 0 / 0 |
| 3 | 0.1 | all | interior_shuffle_6603 | 1569 | 62.2690 | 61.7437 | 62.6099 | 0 / 0 |
| 3 | 0.2 | all | flat | 1569 | 60.4844 | 60.1102 | 60.7991 | 37 / 56 |
| 3 | 0.2 | all | normal | 1569 | 61.6953 | 61.3281 | 61.8639 | 0 / 0 |
| 3 | 0.2 | all | degree2_no_area | 1569 | 61.6316 | 61.2845 | 61.7876 | 2 / 3 |
| 3 | 0.2 | all | interior_shuffle_6601 | 1569 | 61.5679 | 61.2324 | 61.7352 | 1 / 3 |
| 3 | 0.2 | all | interior_shuffle_6602 | 1569 | 61.4404 | 61.1451 | 61.6118 | 0 / 4 |
| 3 | 0.2 | all | interior_shuffle_6603 | 1569 | 61.4404 | 61.1451 | 61.6082 | 0 / 4 |
| 3 | 0.3 | all | flat | 1569 | 59.4009 | 57.5928 | 59.3432 | 65 / 84 |
| 3 | 0.3 | all | normal | 1569 | 60.6119 | 59.3166 | 60.2970 | 0 / 0 |
| 3 | 0.3 | all | degree2_no_area | 1569 | 60.6119 | 59.3166 | 60.2970 | 0 / 0 |
| 3 | 0.3 | all | interior_shuffle_6601 | 1569 | 60.6119 | 59.3166 | 60.2970 | 0 / 0 |
| 3 | 0.3 | all | interior_shuffle_6602 | 1569 | 60.6119 | 59.3166 | 60.2970 | 0 / 0 |
| 3 | 0.3 | all | interior_shuffle_6603 | 1569 | 60.6119 | 59.3166 | 60.2970 | 0 / 0 |
| 3 | 0.4 | all | flat | 1569 | 57.6163 | 58.0175 | 57.4606 | 78 / 93 |
| 3 | 0.4 | all | normal | 1569 | 58.5723 | 57.3506 | 58.8450 | 0 / 0 |
| 3 | 0.4 | all | degree2_no_area | 1569 | 58.2537 | 57.1164 | 58.5549 | 13 / 18 |
| 3 | 0.4 | all | interior_shuffle_6601 | 1569 | 59.1460 | 57.8906 | 59.4395 | 19 / 10 |
| 3 | 0.4 | all | interior_shuffle_6602 | 1569 | 58.7635 | 57.5818 | 59.0913 | 12 / 9 |
| 3 | 0.4 | all | interior_shuffle_6603 | 1569 | 58.7635 | 57.7264 | 59.0713 | 14 / 11 |
| 3 | 0.5 | all | flat | 1569 | 57.1064 | 57.4063 | 56.4615 | 25 / 37 |
| 3 | 0.5 | all | normal | 1569 | 57.8713 | 57.3595 | 57.4825 | 0 / 0 |
| 3 | 0.5 | all | degree2_no_area | 1569 | 57.8713 | 57.3595 | 57.4825 | 0 / 0 |
| 3 | 0.5 | all | interior_shuffle_6601 | 1569 | 57.8713 | 57.3595 | 57.4825 | 0 / 0 |
| 3 | 0.5 | all | interior_shuffle_6602 | 1569 | 57.8713 | 57.3595 | 57.4825 | 0 / 0 |
| 3 | 0.5 | all | interior_shuffle_6603 | 1569 | 57.8713 | 57.3595 | 57.4825 | 0 / 0 |
| 3 | 0.6 | all | flat | 1569 | 56.5328 | 57.6967 | 55.7858 | 56 / 93 |
| 3 | 0.6 | all | normal | 1569 | 58.8910 | 58.4261 | 58.7151 | 0 / 0 |
| 3 | 0.6 | all | degree2_no_area | 1569 | 58.6361 | 58.1355 | 58.5010 | 2 / 6 |
| 3 | 0.6 | all | interior_shuffle_6601 | 1569 | 58.6998 | 58.2200 | 58.5328 | 1 / 4 |
| 3 | 0.6 | all | interior_shuffle_6602 | 1569 | 58.6998 | 58.2868 | 58.5244 | 0 / 3 |
| 3 | 0.6 | all | interior_shuffle_6603 | 1569 | 58.6998 | 58.2285 | 58.5359 | 1 / 4 |
| 3 | 0.7 | all | flat | 1569 | 55.1944 | 53.5570 | 55.2976 | 80 / 84 |
| 3 | 0.7 | all | normal | 1569 | 55.4493 | 53.8401 | 55.4414 | 0 / 0 |
| 3 | 0.7 | all | degree2_no_area | 1569 | 54.9395 | 53.6687 | 54.8993 | 37 / 45 |
| 3 | 0.7 | all | interior_shuffle_6601 | 1569 | 54.8120 | 53.2269 | 54.7977 | 12 / 22 |
| 3 | 0.7 | all | interior_shuffle_6602 | 1569 | 54.9395 | 53.2901 | 54.9173 | 16 / 24 |
| 3 | 0.7 | all | interior_shuffle_6603 | 1569 | 54.8120 | 53.1257 | 54.7674 | 13 / 23 |
| 4 | 0.0 | all | flat | 1512 | 63.0291 | 62.2804 | 62.8066 | 53 / 75 |
| 4 | 0.0 | all | normal | 1512 | 64.4841 | 63.8850 | 64.5959 | 0 / 0 |
| 4 | 0.0 | all | degree2_no_area | 1512 | 64.5503 | 63.9359 | 64.6626 | 2 / 1 |
| 4 | 0.0 | all | interior_shuffle_6601 | 1512 | 64.4841 | 63.9013 | 64.5914 | 1 / 1 |
| 4 | 0.0 | all | interior_shuffle_6602 | 1512 | 64.4841 | 63.8904 | 64.5945 | 1 / 1 |
| 4 | 0.0 | all | interior_shuffle_6603 | 1512 | 64.5503 | 63.9550 | 64.6522 | 1 / 0 |
| 4 | 0.1 | all | flat | 1512 | 62.8968 | 59.6117 | 61.8115 | 22 / 20 |
| 4 | 0.1 | all | normal | 1512 | 62.7646 | 59.8577 | 62.2995 | 0 / 0 |
| 4 | 0.1 | all | degree2_no_area | 1512 | 62.7646 | 59.8577 | 62.2995 | 0 / 0 |
| 4 | 0.1 | all | interior_shuffle_6601 | 1512 | 62.7646 | 59.8577 | 62.2995 | 0 / 0 |
| 4 | 0.1 | all | interior_shuffle_6602 | 1512 | 62.7646 | 59.8577 | 62.2995 | 0 / 0 |
| 4 | 0.1 | all | interior_shuffle_6603 | 1512 | 62.7646 | 59.8577 | 62.2995 | 0 / 0 |
| 4 | 0.2 | all | flat | 1512 | 61.9709 | 59.6432 | 62.0508 | 1 / 2 |
| 4 | 0.2 | all | normal | 1512 | 62.0370 | 59.7407 | 62.1335 | 0 / 0 |
| 4 | 0.2 | all | degree2_no_area | 1512 | 62.0370 | 59.7407 | 62.1335 | 0 / 0 |
| 4 | 0.2 | all | interior_shuffle_6601 | 1512 | 62.0370 | 59.7407 | 62.1335 | 0 / 0 |
| 4 | 0.2 | all | interior_shuffle_6602 | 1512 | 62.0370 | 59.7407 | 62.1335 | 0 / 0 |
| 4 | 0.2 | all | interior_shuffle_6603 | 1512 | 62.0370 | 59.7407 | 62.1335 | 0 / 0 |
| 4 | 0.3 | all | flat | 1512 | 60.5820 | 58.8197 | 60.8104 | 51 / 63 |
| 4 | 0.3 | all | normal | 1512 | 61.3757 | 59.7163 | 61.7423 | 0 / 0 |
| 4 | 0.3 | all | degree2_no_area | 1512 | 61.4418 | 59.7562 | 61.7819 | 10 / 9 |
| 4 | 0.3 | all | interior_shuffle_6601 | 1512 | 61.3095 | 59.7195 | 61.6312 | 7 / 8 |
| 4 | 0.3 | all | interior_shuffle_6602 | 1512 | 61.4418 | 59.7289 | 61.7783 | 9 / 8 |
| 4 | 0.3 | all | interior_shuffle_6603 | 1512 | 61.3757 | 59.7188 | 61.7075 | 8 / 8 |
| 4 | 0.4 | all | flat | 1512 | 60.7143 | 57.7988 | 59.5222 | 104 / 129 |
| 4 | 0.4 | all | normal | 1512 | 62.3677 | 59.9833 | 62.6346 | 0 / 0 |
| 4 | 0.4 | all | degree2_no_area | 1512 | 62.5000 | 60.0961 | 62.7342 | 4 / 2 |
| 4 | 0.4 | all | interior_shuffle_6601 | 1512 | 62.1693 | 59.7675 | 62.4335 | 0 / 3 |
| 4 | 0.4 | all | interior_shuffle_6602 | 1512 | 62.1693 | 59.7675 | 62.4335 | 0 / 3 |
| 4 | 0.4 | all | interior_shuffle_6603 | 1512 | 62.1693 | 59.6856 | 62.4425 | 0 / 3 |
| 4 | 0.5 | all | flat | 1512 | 59.9206 | 57.3062 | 59.8725 | 63 / 103 |
| 4 | 0.5 | all | normal | 1512 | 62.5661 | 60.6693 | 62.7990 | 0 / 0 |
| 4 | 0.5 | all | degree2_no_area | 1512 | 62.3016 | 59.9724 | 62.5006 | 5 / 9 |
| 4 | 0.5 | all | interior_shuffle_6601 | 1512 | 62.6323 | 60.5475 | 62.8433 | 3 / 2 |
| 4 | 0.5 | all | interior_shuffle_6602 | 1512 | 62.6323 | 60.5175 | 62.8611 | 3 / 2 |
| 4 | 0.5 | all | interior_shuffle_6603 | 1512 | 62.6323 | 60.5475 | 62.8382 | 4 / 3 |
| 4 | 0.6 | all | flat | 1512 | 58.2672 | 56.1545 | 58.6585 | 1 / 3 |
| 4 | 0.6 | all | normal | 1512 | 58.3995 | 56.3055 | 58.7734 | 0 / 0 |
| 4 | 0.6 | all | degree2_no_area | 1512 | 58.3995 | 56.3055 | 58.7734 | 0 / 0 |
| 4 | 0.6 | all | interior_shuffle_6601 | 1512 | 58.3995 | 56.3055 | 58.7734 | 0 / 0 |
| 4 | 0.6 | all | interior_shuffle_6602 | 1512 | 58.3995 | 56.3055 | 58.7734 | 0 / 0 |
| 4 | 0.6 | all | interior_shuffle_6603 | 1512 | 58.3995 | 56.3055 | 58.7734 | 0 / 0 |
| 4 | 0.7 | all | flat | 1512 | 56.8783 | 53.1553 | 56.9176 | 2 / 1 |
| 4 | 0.7 | all | normal | 1512 | 56.8122 | 53.3070 | 56.8736 | 0 / 0 |
| 4 | 0.7 | all | degree2_no_area | 1512 | 56.8122 | 53.3070 | 56.8736 | 0 / 0 |
| 4 | 0.7 | all | interior_shuffle_6601 | 1512 | 56.8122 | 53.3070 | 56.8736 | 0 / 0 |
| 4 | 0.7 | all | interior_shuffle_6602 | 1512 | 56.8122 | 53.3070 | 56.8736 | 0 / 0 |
| 4 | 0.7 | all | interior_shuffle_6603 | 1512 | 56.8122 | 53.3070 | 56.8736 | 0 / 0 |
| 5 | 0.0 | all | flat | 1623 | 67.0980 | 65.5290 | 66.7275 | 37 / 41 |
| 5 | 0.0 | all | normal | 1623 | 67.3444 | 66.9453 | 67.4437 | 0 / 0 |
| 5 | 0.0 | all | degree2_no_area | 1623 | 67.3444 | 66.9453 | 67.4437 | 0 / 0 |
| 5 | 0.0 | all | interior_shuffle_6601 | 1623 | 67.3444 | 66.9453 | 67.4437 | 0 / 0 |
| 5 | 0.0 | all | interior_shuffle_6602 | 1623 | 67.3444 | 66.9453 | 67.4437 | 0 / 0 |
| 5 | 0.0 | all | interior_shuffle_6603 | 1623 | 67.3444 | 66.9453 | 67.4437 | 0 / 0 |
| 5 | 0.1 | all | flat | 1623 | 65.8657 | 63.8413 | 65.2738 | 54 / 57 |
| 5 | 0.1 | all | normal | 1623 | 66.0505 | 65.4301 | 66.2427 | 0 / 0 |
| 5 | 0.1 | all | degree2_no_area | 1623 | 66.0505 | 65.5495 | 66.2674 | 6 / 6 |
| 5 | 0.1 | all | interior_shuffle_6601 | 1623 | 65.9273 | 65.3550 | 66.1147 | 4 / 6 |
| 5 | 0.1 | all | interior_shuffle_6602 | 1623 | 65.9889 | 65.4707 | 66.1759 | 4 / 5 |
| 5 | 0.1 | all | interior_shuffle_6603 | 1623 | 65.6192 | 65.0406 | 65.7975 | 1 / 8 |
| 5 | 0.2 | all | flat | 1623 | 65.4344 | 62.9939 | 64.5524 | 103 / 115 |
| 5 | 0.2 | all | normal | 1623 | 66.1738 | 65.7469 | 66.4941 | 0 / 0 |
| 5 | 0.2 | all | degree2_no_area | 1623 | 65.8657 | 65.6647 | 66.2455 | 32 / 37 |
| 5 | 0.2 | all | interior_shuffle_6601 | 1623 | 65.9889 | 65.2681 | 66.2357 | 34 / 37 |
| 5 | 0.2 | all | interior_shuffle_6602 | 1623 | 65.8041 | 65.0636 | 66.0545 | 33 / 39 |
| 5 | 0.2 | all | interior_shuffle_6603 | 1623 | 66.0505 | 65.3877 | 66.3446 | 34 / 36 |
| 5 | 0.3 | all | flat | 1623 | 65.4344 | 62.2382 | 65.0340 | 15 / 19 |
| 5 | 0.3 | all | normal | 1623 | 65.6808 | 62.0053 | 65.1245 | 0 / 0 |
| 5 | 0.3 | all | degree2_no_area | 1623 | 65.6808 | 62.0053 | 65.1245 | 0 / 0 |
| 5 | 0.3 | all | interior_shuffle_6601 | 1623 | 65.6808 | 62.0053 | 65.1245 | 0 / 0 |
| 5 | 0.3 | all | interior_shuffle_6602 | 1623 | 65.6808 | 62.0053 | 65.1245 | 0 / 0 |
| 5 | 0.3 | all | interior_shuffle_6603 | 1623 | 65.6808 | 62.0053 | 65.1245 | 0 / 0 |
| 5 | 0.4 | all | flat | 1623 | 63.1547 | 60.9250 | 62.6908 | 52 / 63 |
| 5 | 0.4 | all | normal | 1623 | 63.8324 | 62.4516 | 64.1519 | 0 / 0 |
| 5 | 0.4 | all | degree2_no_area | 1623 | 63.7092 | 62.2622 | 63.9633 | 6 / 8 |
| 5 | 0.4 | all | interior_shuffle_6601 | 1623 | 63.5860 | 62.2050 | 63.8958 | 1 / 5 |
| 5 | 0.4 | all | interior_shuffle_6602 | 1623 | 63.5860 | 62.2050 | 63.8958 | 1 / 5 |
| 5 | 0.4 | all | interior_shuffle_6603 | 1623 | 63.9556 | 62.6777 | 64.2446 | 5 / 3 |
| 5 | 0.5 | all | flat | 1623 | 62.2921 | 60.0565 | 61.8600 | 46 / 63 |
| 5 | 0.5 | all | normal | 1623 | 63.3395 | 60.4198 | 62.6831 | 0 / 0 |
| 5 | 0.5 | all | degree2_no_area | 1623 | 63.3395 | 60.4375 | 62.6938 | 1 / 1 |
| 5 | 0.5 | all | interior_shuffle_6601 | 1623 | 63.2779 | 60.3218 | 62.6133 | 0 / 1 |
| 5 | 0.5 | all | interior_shuffle_6602 | 1623 | 63.2779 | 60.3218 | 62.6133 | 0 / 1 |
| 5 | 0.5 | all | interior_shuffle_6603 | 1623 | 63.3395 | 60.4198 | 62.6831 | 0 / 0 |
| 5 | 0.6 | all | flat | 1623 | 61.3062 | 59.2239 | 61.3373 | 58 / 77 |
| 5 | 0.6 | all | normal | 1623 | 62.4769 | 60.8270 | 62.6910 | 0 / 0 |
| 5 | 0.6 | all | degree2_no_area | 1623 | 62.3537 | 60.7514 | 62.5722 | 12 / 14 |
| 5 | 0.6 | all | interior_shuffle_6601 | 1623 | 62.2304 | 60.6156 | 62.4595 | 8 / 12 |
| 5 | 0.6 | all | interior_shuffle_6602 | 1623 | 62.0456 | 60.4604 | 62.2925 | 5 / 12 |
| 5 | 0.6 | all | interior_shuffle_6603 | 1623 | 62.0456 | 60.4128 | 62.2846 | 4 / 11 |
| 5 | 0.7 | all | flat | 1623 | 59.0265 | 61.1834 | 59.7979 | 59 / 105 |
| 5 | 0.7 | all | normal | 1623 | 61.8608 | 61.6041 | 62.2232 | 0 / 0 |
| 5 | 0.7 | all | degree2_no_area | 1623 | 61.7375 | 61.3277 | 62.0384 | 20 / 22 |
| 5 | 0.7 | all | interior_shuffle_6601 | 1623 | 61.5527 | 61.2234 | 61.9733 | 5 / 10 |
| 5 | 0.7 | all | interior_shuffle_6602 | 1623 | 61.4295 | 61.0219 | 61.8429 | 5 / 12 |
| 5 | 0.7 | all | interior_shuffle_6603 | 1623 | 61.3678 | 61.0808 | 61.7743 | 4 / 12 |

## Equal-cell arithmetic summaries

| Scope | Fold | Rate | Cohort | Mode | Nonempty / cells | ACC | UA | W-F1 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| fold_eight_rate_mean | 1 | mean | all | flat | 8/8 | 61.1981 | 60.9886 | 60.8252 |
| fold_eight_rate_mean | 1 | mean | all | normal | 8/8 | 61.8081 | 61.7396 | 61.5617 |
| fold_eight_rate_mean | 1 | mean | all | degree2_no_area | 8/8 | 61.6169 | 61.5507 | 61.3737 |
| fold_eight_rate_mean | 1 | mean | all | interior_shuffle_6601 | 8/8 | 61.6897 | 61.6154 | 61.4410 |
| fold_eight_rate_mean | 1 | mean | all | interior_shuffle_6602 | 8/8 | 61.5896 | 61.5249 | 61.3449 |
| fold_eight_rate_mean | 1 | mean | all | interior_shuffle_6603 | 8/8 | 61.6260 | 61.5687 | 61.3779 |
| fold_eight_rate_mean | 1 | mean | shuffle_eligible | flat | 8/8 | 62.4664 | 62.0595 | 62.1203 |
| fold_eight_rate_mean | 1 | mean | shuffle_eligible | normal | 8/8 | 63.0492 | 62.9539 | 62.8576 |
| fold_eight_rate_mean | 1 | mean | shuffle_eligible | degree2_no_area | 8/8 | 62.8577 | 62.7822 | 62.6728 |
| fold_eight_rate_mean | 1 | mean | shuffle_eligible | interior_shuffle_6601 | 8/8 | 62.9114 | 62.8179 | 62.7198 |
| fold_eight_rate_mean | 1 | mean | shuffle_eligible | interior_shuffle_6602 | 8/8 | 62.8020 | 62.7199 | 62.6152 |
| fold_eight_rate_mean | 1 | mean | shuffle_eligible | interior_shuffle_6603 | 8/8 | 62.8413 | 62.7716 | 62.6489 |
| fold_eight_rate_mean | 1 | mean | no_history | flat | 8/8 | 57.1429 | 39.6577 | 51.7883 |
| fold_eight_rate_mean | 1 | mean | no_history | normal | 8/8 | 57.1429 | 39.6577 | 51.7883 |
| fold_eight_rate_mean | 1 | mean | no_history | degree2_no_area | 8/8 | 57.1429 | 39.6577 | 51.7883 |
| fold_eight_rate_mean | 1 | mean | no_history | interior_shuffle_6601 | 8/8 | 57.1429 | 39.6577 | 51.7883 |
| fold_eight_rate_mean | 1 | mean | no_history | interior_shuffle_6602 | 8/8 | 57.1429 | 39.6577 | 51.7883 |
| fold_eight_rate_mean | 1 | mean | no_history | interior_shuffle_6603 | 8/8 | 57.1429 | 39.6577 | 51.7883 |
| fold_eight_rate_mean | 1 | mean | T_present | flat | 8/8 | 61.8224 | 61.1350 | 61.4783 |
| fold_eight_rate_mean | 1 | mean | T_present | normal | 8/8 | 62.5754 | 62.0606 | 62.3439 |
| fold_eight_rate_mean | 1 | mean | T_present | degree2_no_area | 8/8 | 62.4061 | 61.8952 | 62.1685 |
| fold_eight_rate_mean | 1 | mean | T_present | interior_shuffle_6601 | 8/8 | 62.5173 | 62.0056 | 62.2736 |
| fold_eight_rate_mean | 1 | mean | T_present | interior_shuffle_6602 | 8/8 | 62.3554 | 61.8510 | 62.1141 |
| fold_eight_rate_mean | 1 | mean | T_present | interior_shuffle_6603 | 8/8 | 62.4312 | 61.9387 | 62.1889 |
| fold_eight_rate_mean | 1 | mean | T_missing_seen | flat | 7/8 | 61.0284 | 61.6675 | 60.5008 |
| fold_eight_rate_mean | 1 | mean | T_missing_seen | normal | 7/8 | 61.2949 | 62.0170 | 60.9059 |
| fold_eight_rate_mean | 1 | mean | T_missing_seen | degree2_no_area | 7/8 | 61.1369 | 61.9492 | 60.7535 |
| fold_eight_rate_mean | 1 | mean | T_missing_seen | interior_shuffle_6601 | 7/8 | 61.1094 | 61.8311 | 60.7310 |
| fold_eight_rate_mean | 1 | mean | T_missing_seen | interior_shuffle_6602 | 7/8 | 61.1738 | 61.9096 | 60.7936 |
| fold_eight_rate_mean | 1 | mean | T_missing_seen | interior_shuffle_6603 | 7/8 | 61.1645 | 61.8985 | 60.7789 |
| fold_eight_rate_mean | 1 | mean | T_missing_never | flat | 7/8 | 36.0771 | 21.0071 | 30.9439 |
| fold_eight_rate_mean | 1 | mean | T_missing_never | normal | 7/8 | 36.5592 | 22.2604 | 31.2154 |
| fold_eight_rate_mean | 1 | mean | T_missing_never | degree2_no_area | 7/8 | 36.2677 | 22.1016 | 31.0419 |
| fold_eight_rate_mean | 1 | mean | T_missing_never | interior_shuffle_6601 | 7/8 | 36.5592 | 22.2604 | 31.2154 |
| fold_eight_rate_mean | 1 | mean | T_missing_never | interior_shuffle_6602 | 7/8 | 36.5592 | 22.2604 | 31.2154 |
| fold_eight_rate_mean | 1 | mean | T_missing_never | interior_shuffle_6603 | 7/8 | 36.5592 | 22.2604 | 31.2154 |
| fold_eight_rate_mean | 2 | mean | all | flat | 8/8 | 63.4679 | 62.2271 | 63.3350 |
| fold_eight_rate_mean | 2 | mean | all | normal | 8/8 | 63.8459 | 63.1167 | 63.8904 |
| fold_eight_rate_mean | 2 | mean | all | degree2_no_area | 8/8 | 63.6431 | 62.9265 | 63.6953 |
| fold_eight_rate_mean | 2 | mean | all | interior_shuffle_6601 | 8/8 | 63.6892 | 62.9847 | 63.7272 |
| fold_eight_rate_mean | 2 | mean | all | interior_shuffle_6602 | 8/8 | 63.7260 | 62.9766 | 63.7642 |
| fold_eight_rate_mean | 2 | mean | all | interior_shuffle_6603 | 8/8 | 63.6892 | 62.9550 | 63.7192 |
| fold_eight_rate_mean | 2 | mean | shuffle_eligible | flat | 8/8 | 64.1049 | 63.0950 | 63.9983 |
| fold_eight_rate_mean | 2 | mean | shuffle_eligible | normal | 8/8 | 64.5046 | 63.9410 | 64.5977 |
| fold_eight_rate_mean | 2 | mean | shuffle_eligible | degree2_no_area | 8/8 | 64.2621 | 63.6923 | 64.3613 |
| fold_eight_rate_mean | 2 | mean | shuffle_eligible | interior_shuffle_6601 | 8/8 | 64.3280 | 63.7909 | 64.4117 |
| fold_eight_rate_mean | 2 | mean | shuffle_eligible | interior_shuffle_6602 | 8/8 | 64.3684 | 63.7779 | 64.4536 |
| fold_eight_rate_mean | 2 | mean | shuffle_eligible | interior_shuffle_6603 | 8/8 | 64.3288 | 63.7617 | 64.4035 |
| fold_eight_rate_mean | 2 | mean | no_history | flat | 8/8 | 58.7500 | 40.9821 | 55.7769 |
| fold_eight_rate_mean | 2 | mean | no_history | normal | 8/8 | 58.7500 | 40.9821 | 55.7769 |
| fold_eight_rate_mean | 2 | mean | no_history | degree2_no_area | 8/8 | 58.7500 | 40.9821 | 55.7769 |
| fold_eight_rate_mean | 2 | mean | no_history | interior_shuffle_6601 | 8/8 | 58.7500 | 40.9821 | 55.7769 |
| fold_eight_rate_mean | 2 | mean | no_history | interior_shuffle_6602 | 8/8 | 58.7500 | 40.9821 | 55.7769 |
| fold_eight_rate_mean | 2 | mean | no_history | interior_shuffle_6603 | 8/8 | 58.7500 | 40.9821 | 55.7769 |
| fold_eight_rate_mean | 2 | mean | T_present | flat | 8/8 | 65.1978 | 64.0727 | 64.9955 |
| fold_eight_rate_mean | 2 | mean | T_present | normal | 8/8 | 65.6448 | 65.2304 | 65.6367 |
| fold_eight_rate_mean | 2 | mean | T_present | degree2_no_area | 8/8 | 65.3741 | 64.9703 | 65.3748 |
| fold_eight_rate_mean | 2 | mean | T_present | interior_shuffle_6601 | 8/8 | 65.4968 | 65.0720 | 65.4773 |
| fold_eight_rate_mean | 2 | mean | T_present | interior_shuffle_6602 | 8/8 | 65.5337 | 65.0844 | 65.5134 |
| fold_eight_rate_mean | 2 | mean | T_present | interior_shuffle_6603 | 8/8 | 65.4973 | 65.0699 | 65.4732 |
| fold_eight_rate_mean | 2 | mean | T_missing_seen | flat | 7/8 | 61.7104 | 60.8120 | 61.6532 |
| fold_eight_rate_mean | 2 | mean | T_missing_seen | normal | 7/8 | 61.7772 | 61.2151 | 61.7967 |
| fold_eight_rate_mean | 2 | mean | T_missing_seen | degree2_no_area | 7/8 | 61.7768 | 61.2350 | 61.7981 |
| fold_eight_rate_mean | 2 | mean | T_missing_seen | interior_shuffle_6601 | 7/8 | 61.7394 | 61.2182 | 61.7608 |
| fold_eight_rate_mean | 2 | mean | T_missing_seen | interior_shuffle_6602 | 7/8 | 61.7394 | 61.1797 | 61.7617 |
| fold_eight_rate_mean | 2 | mean | T_missing_seen | interior_shuffle_6603 | 7/8 | 61.7605 | 61.1767 | 61.7736 |
| fold_eight_rate_mean | 2 | mean | T_missing_never | flat | 7/8 | 41.3024 | 26.1597 | 36.1072 |
| fold_eight_rate_mean | 2 | mean | T_missing_never | normal | 7/8 | 42.3049 | 27.4898 | 37.2539 |
| fold_eight_rate_mean | 2 | mean | T_missing_never | degree2_no_area | 7/8 | 42.3049 | 27.4898 | 37.2539 |
| fold_eight_rate_mean | 2 | mean | T_missing_never | interior_shuffle_6601 | 7/8 | 42.3049 | 27.4898 | 37.2539 |
| fold_eight_rate_mean | 2 | mean | T_missing_never | interior_shuffle_6602 | 7/8 | 42.3049 | 27.4898 | 37.2539 |
| fold_eight_rate_mean | 2 | mean | T_missing_never | interior_shuffle_6603 | 7/8 | 42.3049 | 27.4898 | 37.2539 |
| fold_eight_rate_mean | 3 | mean | all | flat | 8/8 | 58.6361 | 58.2246 | 58.5697 |
| fold_eight_rate_mean | 3 | mean | all | normal | 8/8 | 59.7674 | 58.9629 | 59.7952 |
| fold_eight_rate_mean | 3 | mean | all | degree2_no_area | 8/8 | 59.6160 | 58.8617 | 59.6472 |
| fold_eight_rate_mean | 3 | mean | all | interior_shuffle_6601 | 8/8 | 59.7196 | 58.9160 | 59.7502 |
| fold_eight_rate_mean | 3 | mean | all | interior_shuffle_6602 | 8/8 | 59.6718 | 58.8827 | 59.7052 |
| fold_eight_rate_mean | 3 | mean | all | interior_shuffle_6603 | 8/8 | 59.6558 | 58.8730 | 59.6849 |
| fold_eight_rate_mean | 3 | mean | shuffle_eligible | flat | 8/8 | 58.8827 | 58.4958 | 58.8948 |
| fold_eight_rate_mean | 3 | mean | shuffle_eligible | normal | 8/8 | 60.0042 | 59.1395 | 60.1120 |
| fold_eight_rate_mean | 3 | mean | shuffle_eligible | degree2_no_area | 8/8 | 59.8347 | 59.0360 | 59.9395 |
| fold_eight_rate_mean | 3 | mean | shuffle_eligible | interior_shuffle_6601 | 8/8 | 59.9162 | 59.0537 | 60.0280 |
| fold_eight_rate_mean | 3 | mean | shuffle_eligible | interior_shuffle_6602 | 8/8 | 59.8696 | 59.0221 | 59.9849 |
| fold_eight_rate_mean | 3 | mean | shuffle_eligible | interior_shuffle_6603 | 8/8 | 59.8456 | 59.0020 | 59.9555 |
| fold_eight_rate_mean | 3 | mean | no_history | flat | 8/8 | 60.9375 | 53.4722 | 59.4146 |
| fold_eight_rate_mean | 3 | mean | no_history | normal | 8/8 | 60.9375 | 53.4722 | 59.4146 |
| fold_eight_rate_mean | 3 | mean | no_history | degree2_no_area | 8/8 | 60.9375 | 53.4722 | 59.4146 |
| fold_eight_rate_mean | 3 | mean | no_history | interior_shuffle_6601 | 8/8 | 60.9375 | 53.4722 | 59.4146 |
| fold_eight_rate_mean | 3 | mean | no_history | interior_shuffle_6602 | 8/8 | 60.9375 | 53.4722 | 59.4146 |
| fold_eight_rate_mean | 3 | mean | no_history | interior_shuffle_6603 | 8/8 | 60.9375 | 53.4722 | 59.4146 |
| fold_eight_rate_mean | 3 | mean | T_present | flat | 8/8 | 59.9965 | 59.4711 | 59.9139 |
| fold_eight_rate_mean | 3 | mean | T_present | normal | 8/8 | 60.9700 | 60.1150 | 60.9978 |
| fold_eight_rate_mean | 3 | mean | T_present | degree2_no_area | 8/8 | 60.8837 | 60.0938 | 60.9176 |
| fold_eight_rate_mean | 3 | mean | T_present | interior_shuffle_6601 | 8/8 | 60.8685 | 60.0326 | 60.9065 |
| fold_eight_rate_mean | 3 | mean | T_present | interior_shuffle_6602 | 8/8 | 60.8069 | 59.9965 | 60.8503 |
| fold_eight_rate_mean | 3 | mean | T_present | interior_shuffle_6603 | 8/8 | 60.7965 | 59.9986 | 60.8308 |
| fold_eight_rate_mean | 3 | mean | T_missing_seen | flat | 7/8 | 56.8748 | 56.5974 | 56.8209 |
| fold_eight_rate_mean | 3 | mean | T_missing_seen | normal | 7/8 | 58.0278 | 57.2102 | 58.0969 |
| fold_eight_rate_mean | 3 | mean | T_missing_seen | degree2_no_area | 7/8 | 57.8745 | 57.0860 | 57.9592 |
| fold_eight_rate_mean | 3 | mean | T_missing_seen | interior_shuffle_6601 | 7/8 | 58.0445 | 57.2391 | 58.1123 |
| fold_eight_rate_mean | 3 | mean | T_missing_seen | interior_shuffle_6602 | 7/8 | 58.0349 | 57.2057 | 58.1031 |
| fold_eight_rate_mean | 3 | mean | T_missing_seen | interior_shuffle_6603 | 7/8 | 57.9961 | 57.1686 | 58.0608 |
| fold_eight_rate_mean | 3 | mean | T_missing_never | flat | 7/8 | 48.2788 | 28.1858 | 46.9975 |
| fold_eight_rate_mean | 3 | mean | T_missing_never | normal | 7/8 | 49.4588 | 29.3508 | 47.9737 |
| fold_eight_rate_mean | 3 | mean | T_missing_never | degree2_no_area | 7/8 | 49.4588 | 29.2185 | 47.7630 |
| fold_eight_rate_mean | 3 | mean | T_missing_never | interior_shuffle_6601 | 7/8 | 49.4588 | 29.3508 | 47.9737 |
| fold_eight_rate_mean | 3 | mean | T_missing_never | interior_shuffle_6602 | 7/8 | 49.4588 | 29.3508 | 47.9737 |
| fold_eight_rate_mean | 3 | mean | T_missing_never | interior_shuffle_6603 | 7/8 | 49.4588 | 29.3508 | 47.9737 |
| fold_eight_rate_mean | 4 | mean | all | flat | 8/8 | 60.5324 | 58.0962 | 60.3063 |
| fold_eight_rate_mean | 4 | mean | all | normal | 8/8 | 61.3509 | 59.1831 | 61.4815 |
| fold_eight_rate_mean | 4 | mean | all | degree2_no_area | 8/8 | 61.3509 | 59.1214 | 61.4699 |
| fold_eight_rate_mean | 4 | mean | all | interior_shuffle_6601 | 8/8 | 61.3261 | 59.1433 | 61.4474 |
| fold_eight_rate_mean | 4 | mean | all | interior_shuffle_6602 | 8/8 | 61.3426 | 59.1394 | 61.4684 |
| fold_eight_rate_mean | 4 | mean | all | interior_shuffle_6603 | 8/8 | 61.3426 | 59.1397 | 61.4650 |
| fold_eight_rate_mean | 4 | mean | shuffle_eligible | flat | 8/8 | 61.2539 | 59.1447 | 61.0723 |
| fold_eight_rate_mean | 4 | mean | shuffle_eligible | normal | 8/8 | 61.9718 | 60.1262 | 62.1377 |
| fold_eight_rate_mean | 4 | mean | shuffle_eligible | degree2_no_area | 8/8 | 61.9617 | 60.0365 | 62.1183 |
| fold_eight_rate_mean | 4 | mean | shuffle_eligible | interior_shuffle_6601 | 8/8 | 61.9447 | 60.0763 | 62.1022 |
| fold_eight_rate_mean | 4 | mean | shuffle_eligible | interior_shuffle_6602 | 8/8 | 61.9629 | 60.0705 | 62.1257 |
| fold_eight_rate_mean | 4 | mean | shuffle_eligible | interior_shuffle_6603 | 8/8 | 61.9628 | 60.0708 | 62.1221 |
| fold_eight_rate_mean | 4 | mean | no_history | flat | 8/8 | 39.1667 | 33.3449 | 33.6844 |
| fold_eight_rate_mean | 4 | mean | no_history | normal | 8/8 | 39.1667 | 33.3449 | 33.6844 |
| fold_eight_rate_mean | 4 | mean | no_history | degree2_no_area | 8/8 | 39.1667 | 33.3449 | 33.6844 |
| fold_eight_rate_mean | 4 | mean | no_history | interior_shuffle_6601 | 8/8 | 39.1667 | 33.3449 | 33.6844 |
| fold_eight_rate_mean | 4 | mean | no_history | interior_shuffle_6602 | 8/8 | 39.1667 | 33.3449 | 33.6844 |
| fold_eight_rate_mean | 4 | mean | no_history | interior_shuffle_6603 | 8/8 | 39.1667 | 33.3449 | 33.6844 |
| fold_eight_rate_mean | 4 | mean | T_present | flat | 8/8 | 61.5880 | 59.4884 | 61.4441 |
| fold_eight_rate_mean | 4 | mean | T_present | normal | 8/8 | 61.9702 | 60.3352 | 62.1717 |
| fold_eight_rate_mean | 4 | mean | T_present | degree2_no_area | 8/8 | 61.9758 | 60.2524 | 62.1635 |
| fold_eight_rate_mean | 4 | mean | T_present | interior_shuffle_6601 | 8/8 | 61.9157 | 60.2598 | 62.1088 |
| fold_eight_rate_mean | 4 | mean | T_present | interior_shuffle_6602 | 8/8 | 61.8871 | 60.2107 | 62.0855 |
| fold_eight_rate_mean | 4 | mean | T_present | interior_shuffle_6603 | 8/8 | 61.9003 | 60.2017 | 62.0981 |
| fold_eight_rate_mean | 4 | mean | T_missing_seen | flat | 7/8 | 58.5633 | 55.4227 | 58.0837 |
| fold_eight_rate_mean | 4 | mean | T_missing_seen | normal | 7/8 | 59.9198 | 56.8625 | 59.9940 |
| fold_eight_rate_mean | 4 | mean | T_missing_seen | degree2_no_area | 7/8 | 59.9513 | 56.7972 | 60.0216 |
| fold_eight_rate_mean | 4 | mean | T_missing_seen | interior_shuffle_6601 | 7/8 | 59.9440 | 56.8424 | 60.0120 |
| fold_eight_rate_mean | 4 | mean | T_missing_seen | interior_shuffle_6602 | 7/8 | 60.0624 | 56.9402 | 60.1232 |
| fold_eight_rate_mean | 4 | mean | T_missing_seen | interior_shuffle_6603 | 7/8 | 60.0429 | 56.9620 | 60.1010 |
| fold_eight_rate_mean | 4 | mean | T_missing_never | flat | 7/8 | 38.7051 | 24.0011 | 31.2278 |
| fold_eight_rate_mean | 4 | mean | T_missing_never | normal | 7/8 | 39.4193 | 24.5302 | 33.0937 |
| fold_eight_rate_mean | 4 | mean | T_missing_never | degree2_no_area | 7/8 | 38.8902 | 24.3318 | 32.6012 |
| fold_eight_rate_mean | 4 | mean | T_missing_never | interior_shuffle_6601 | 7/8 | 39.4193 | 24.5302 | 33.0937 |
| fold_eight_rate_mean | 4 | mean | T_missing_never | interior_shuffle_6602 | 7/8 | 39.4193 | 24.5302 | 33.0937 |
| fold_eight_rate_mean | 4 | mean | T_missing_never | interior_shuffle_6603 | 7/8 | 39.4193 | 24.5302 | 33.0937 |
| fold_eight_rate_mean | 5 | mean | all | flat | 8/8 | 63.7015 | 61.9989 | 63.4092 |
| fold_eight_rate_mean | 5 | mean | all | normal | 8/8 | 64.5949 | 63.1788 | 64.6318 |
| fold_eight_rate_mean | 5 | mean | all | degree2_no_area | 8/8 | 64.5102 | 63.1180 | 64.5436 |
| fold_eight_rate_mean | 5 | mean | all | interior_shuffle_6601 | 8/8 | 64.4486 | 62.9924 | 64.4826 |
| fold_eight_rate_mean | 5 | mean | all | interior_shuffle_6602 | 8/8 | 64.3946 | 62.9368 | 64.4304 |
| fold_eight_rate_mean | 5 | mean | all | interior_shuffle_6603 | 8/8 | 64.4254 | 62.9963 | 64.4621 |
| fold_eight_rate_mean | 5 | mean | shuffle_eligible | flat | 8/8 | 63.8290 | 62.3306 | 63.4466 |
| fold_eight_rate_mean | 5 | mean | shuffle_eligible | normal | 8/8 | 64.6396 | 63.4000 | 64.5957 |
| fold_eight_rate_mean | 5 | mean | shuffle_eligible | degree2_no_area | 8/8 | 64.5008 | 63.2795 | 64.4457 |
| fold_eight_rate_mean | 5 | mean | shuffle_eligible | interior_shuffle_6601 | 8/8 | 64.4611 | 63.1884 | 64.4158 |
| fold_eight_rate_mean | 5 | mean | shuffle_eligible | interior_shuffle_6602 | 8/8 | 64.3934 | 63.1193 | 64.3493 |
| fold_eight_rate_mean | 5 | mean | shuffle_eligible | interior_shuffle_6603 | 8/8 | 64.4254 | 63.1756 | 64.3815 |
| fold_eight_rate_mean | 5 | mean | no_history | flat | 8/8 | 57.6613 | 34.8661 | 55.4888 |
| fold_eight_rate_mean | 5 | mean | no_history | normal | 8/8 | 57.6613 | 34.8661 | 55.4888 |
| fold_eight_rate_mean | 5 | mean | no_history | degree2_no_area | 8/8 | 57.6613 | 34.8661 | 55.4888 |
| fold_eight_rate_mean | 5 | mean | no_history | interior_shuffle_6601 | 8/8 | 57.6613 | 34.8661 | 55.4888 |
| fold_eight_rate_mean | 5 | mean | no_history | interior_shuffle_6602 | 8/8 | 57.6613 | 34.8661 | 55.4888 |
| fold_eight_rate_mean | 5 | mean | no_history | interior_shuffle_6603 | 8/8 | 57.6613 | 34.8661 | 55.4888 |
| fold_eight_rate_mean | 5 | mean | T_present | flat | 8/8 | 64.0996 | 62.6206 | 63.9113 |
| fold_eight_rate_mean | 5 | mean | T_present | normal | 8/8 | 64.9066 | 63.6833 | 65.0323 |
| fold_eight_rate_mean | 5 | mean | T_present | degree2_no_area | 8/8 | 64.6832 | 63.4377 | 64.8067 |
| fold_eight_rate_mean | 5 | mean | T_present | interior_shuffle_6601 | 8/8 | 64.6884 | 63.4356 | 64.8336 |
| fold_eight_rate_mean | 5 | mean | T_present | interior_shuffle_6602 | 8/8 | 64.6116 | 63.3714 | 64.7549 |
| fold_eight_rate_mean | 5 | mean | T_present | interior_shuffle_6603 | 8/8 | 64.6910 | 63.4905 | 64.8417 |
| fold_eight_rate_mean | 5 | mean | T_missing_seen | flat | 7/8 | 61.9501 | 60.0271 | 61.1143 |
| fold_eight_rate_mean | 5 | mean | T_missing_seen | normal | 7/8 | 63.3290 | 61.5655 | 63.0695 |
| fold_eight_rate_mean | 5 | mean | T_missing_seen | degree2_no_area | 7/8 | 63.3056 | 61.6839 | 62.9906 |
| fold_eight_rate_mean | 5 | mean | T_missing_seen | interior_shuffle_6601 | 7/8 | 63.1335 | 61.3227 | 62.8167 |
| fold_eight_rate_mean | 5 | mean | T_missing_seen | interior_shuffle_6602 | 7/8 | 63.2575 | 61.4371 | 62.9666 |
| fold_eight_rate_mean | 5 | mean | T_missing_seen | interior_shuffle_6603 | 7/8 | 63.0835 | 61.1607 | 62.7641 |
| fold_eight_rate_mean | 5 | mean | T_missing_never | flat | 7/8 | 56.8103 | 30.3526 | 52.2675 |
| fold_eight_rate_mean | 5 | mean | T_missing_never | normal | 7/8 | 58.5388 | 30.7232 | 53.7625 |
| fold_eight_rate_mean | 5 | mean | T_missing_never | degree2_no_area | 7/8 | 58.5388 | 30.7232 | 53.7625 |
| fold_eight_rate_mean | 5 | mean | T_missing_never | interior_shuffle_6601 | 7/8 | 58.5388 | 30.7232 | 53.7625 |
| fold_eight_rate_mean | 5 | mean | T_missing_never | interior_shuffle_6602 | 7/8 | 58.5388 | 30.7232 | 53.7625 |
| fold_eight_rate_mean | 5 | mean | T_missing_never | interior_shuffle_6603 | 7/8 | 58.5388 | 30.7232 | 53.7625 |
| fivefold_rate_mean | None | 0.0 | all | flat | 5/5 | 64.5483 | 63.4797 | 64.3468 |
| fivefold_rate_mean | None | 0.0 | all | normal | 5/5 | 65.2789 | 64.7409 | 65.3818 |
| fivefold_rate_mean | None | 0.0 | all | degree2_no_area | 5/5 | 64.9394 | 64.3323 | 65.0442 |
| fivefold_rate_mean | None | 0.0 | all | interior_shuffle_6601 | 5/5 | 65.0578 | 64.5006 | 65.1409 |
| fivefold_rate_mean | None | 0.0 | all | interior_shuffle_6602 | 5/5 | 64.9420 | 64.3527 | 65.0325 |
| fivefold_rate_mean | None | 0.0 | all | interior_shuffle_6603 | 5/5 | 64.9104 | 64.3760 | 64.9881 |
| fivefold_rate_mean | None | 0.0 | shuffle_eligible | flat | 5/5 | 64.8565 | 64.0126 | 64.6923 |
| fivefold_rate_mean | None | 0.0 | shuffle_eligible | normal | 5/5 | 65.4701 | 65.1314 | 65.6290 |
| fivefold_rate_mean | None | 0.0 | shuffle_eligible | degree2_no_area | 5/5 | 65.0828 | 64.6713 | 65.2497 |
| fivefold_rate_mean | None | 0.0 | shuffle_eligible | interior_shuffle_6601 | 5/5 | 65.2277 | 64.8700 | 65.3684 |
| fivefold_rate_mean | None | 0.0 | shuffle_eligible | interior_shuffle_6602 | 5/5 | 65.1021 | 64.6995 | 65.2503 |
| fivefold_rate_mean | None | 0.0 | shuffle_eligible | interior_shuffle_6603 | 5/5 | 65.0669 | 64.7380 | 65.1977 |
| fivefold_rate_mean | None | 0.0 | no_history | flat | 5/5 | 64.0090 | 47.1693 | 60.5966 |
| fivefold_rate_mean | None | 0.0 | no_history | normal | 5/5 | 64.0090 | 47.1693 | 60.5966 |
| fivefold_rate_mean | None | 0.0 | no_history | degree2_no_area | 5/5 | 64.0090 | 47.1693 | 60.5966 |
| fivefold_rate_mean | None | 0.0 | no_history | interior_shuffle_6601 | 5/5 | 64.0090 | 47.1693 | 60.5966 |
| fivefold_rate_mean | None | 0.0 | no_history | interior_shuffle_6602 | 5/5 | 64.0090 | 47.1693 | 60.5966 |
| fivefold_rate_mean | None | 0.0 | no_history | interior_shuffle_6603 | 5/5 | 64.0090 | 47.1693 | 60.5966 |
| fivefold_rate_mean | None | 0.0 | T_present | flat | 5/5 | 64.5483 | 63.4797 | 64.3468 |
| fivefold_rate_mean | None | 0.0 | T_present | normal | 5/5 | 65.2789 | 64.7409 | 65.3818 |
| fivefold_rate_mean | None | 0.0 | T_present | degree2_no_area | 5/5 | 64.9394 | 64.3323 | 65.0442 |
| fivefold_rate_mean | None | 0.0 | T_present | interior_shuffle_6601 | 5/5 | 65.0578 | 64.5006 | 65.1409 |
| fivefold_rate_mean | None | 0.0 | T_present | interior_shuffle_6602 | 5/5 | 64.9420 | 64.3527 | 65.0325 |
| fivefold_rate_mean | None | 0.0 | T_present | interior_shuffle_6603 | 5/5 | 64.9104 | 64.3760 | 64.9881 |
| fivefold_rate_mean | None | 0.0 | T_missing_seen | flat | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_seen | normal | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_seen | degree2_no_area | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_seen | interior_shuffle_6601 | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_seen | interior_shuffle_6602 | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_seen | interior_shuffle_6603 | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_never | flat | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_never | normal | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_never | degree2_no_area | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_never | interior_shuffle_6601 | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_never | interior_shuffle_6602 | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.0 | T_missing_never | interior_shuffle_6603 | 0/5 | N/A | N/A | N/A |
| fivefold_rate_mean | None | 0.1 | all | flat | 5/5 | 63.4380 | 62.1172 | 63.1002 |
| fivefold_rate_mean | None | 0.1 | all | normal | 5/5 | 64.1475 | 63.1250 | 64.1246 |
| fivefold_rate_mean | None | 0.1 | all | degree2_no_area | 5/5 | 64.1202 | 63.1263 | 64.1031 |
| fivefold_rate_mean | None | 0.1 | all | interior_shuffle_6601 | 5/5 | 64.1229 | 63.1100 | 64.0989 |
| fivefold_rate_mean | None | 0.1 | all | interior_shuffle_6602 | 5/5 | 64.1352 | 63.1331 | 64.1112 |
| fivefold_rate_mean | None | 0.1 | all | interior_shuffle_6603 | 5/5 | 64.0613 | 63.0471 | 64.0355 |
| fivefold_rate_mean | None | 0.1 | shuffle_eligible | flat | 5/5 | 63.5154 | 62.2607 | 63.2051 |
| fivefold_rate_mean | None | 0.1 | shuffle_eligible | normal | 5/5 | 64.2000 | 63.1826 | 64.2158 |
| fivefold_rate_mean | None | 0.1 | shuffle_eligible | degree2_no_area | 5/5 | 64.1727 | 63.1822 | 64.1947 |
| fivefold_rate_mean | None | 0.1 | shuffle_eligible | interior_shuffle_6601 | 5/5 | 64.1733 | 63.1671 | 64.1878 |
| fivefold_rate_mean | None | 0.1 | shuffle_eligible | interior_shuffle_6602 | 5/5 | 64.1866 | 63.1910 | 64.2009 |
| fivefold_rate_mean | None | 0.1 | shuffle_eligible | interior_shuffle_6603 | 5/5 | 64.1065 | 63.0990 | 64.1185 |
| fivefold_rate_mean | None | 0.1 | no_history | flat | 5/5 | 67.2792 | 51.1958 | 63.5593 |
| fivefold_rate_mean | None | 0.1 | no_history | normal | 5/5 | 67.2792 | 51.1958 | 63.5593 |
| fivefold_rate_mean | None | 0.1 | no_history | degree2_no_area | 5/5 | 67.2792 | 51.1958 | 63.5593 |
| fivefold_rate_mean | None | 0.1 | no_history | interior_shuffle_6601 | 5/5 | 67.2792 | 51.1958 | 63.5593 |
| fivefold_rate_mean | None | 0.1 | no_history | interior_shuffle_6602 | 5/5 | 67.2792 | 51.1958 | 63.5593 |
| fivefold_rate_mean | None | 0.1 | no_history | interior_shuffle_6603 | 5/5 | 67.2792 | 51.1958 | 63.5593 |
| fivefold_rate_mean | None | 0.1 | T_present | flat | 5/5 | 63.5170 | 62.2220 | 63.2217 |
| fivefold_rate_mean | None | 0.1 | T_present | normal | 5/5 | 64.3054 | 63.2977 | 64.3091 |
| fivefold_rate_mean | None | 0.1 | T_present | degree2_no_area | 5/5 | 64.2749 | 63.2839 | 64.2860 |
| fivefold_rate_mean | None | 0.1 | T_present | interior_shuffle_6601 | 5/5 | 64.2916 | 63.2909 | 64.2958 |
| fivefold_rate_mean | None | 0.1 | T_present | interior_shuffle_6602 | 5/5 | 64.2777 | 63.2811 | 64.2795 |
| fivefold_rate_mean | None | 0.1 | T_present | interior_shuffle_6603 | 5/5 | 64.2362 | 63.2466 | 64.2386 |
| fivefold_rate_mean | None | 0.1 | T_missing_seen | flat | 5/5 | 63.2115 | 61.6486 | 62.2513 |
| fivefold_rate_mean | None | 0.1 | T_missing_seen | normal | 5/5 | 63.2154 | 62.0513 | 62.8076 |
| fivefold_rate_mean | None | 0.1 | T_missing_seen | degree2_no_area | 5/5 | 63.2154 | 62.1434 | 62.7829 |
| fivefold_rate_mean | None | 0.1 | T_missing_seen | interior_shuffle_6601 | 5/5 | 63.0985 | 61.9680 | 62.6680 |
| fivefold_rate_mean | None | 0.1 | T_missing_seen | interior_shuffle_6602 | 5/5 | 63.3324 | 62.2267 | 62.9267 |
| fivefold_rate_mean | None | 0.1 | T_missing_seen | interior_shuffle_6603 | 5/5 | 62.9815 | 61.7925 | 62.5554 |
| fivefold_rate_mean | None | 0.1 | T_missing_never | flat | 5/5 | 24.9524 | 13.3333 | 16.7302 |
| fivefold_rate_mean | None | 0.1 | T_missing_never | normal | 5/5 | 24.9524 | 13.3333 | 16.7302 |
| fivefold_rate_mean | None | 0.1 | T_missing_never | degree2_no_area | 5/5 | 24.9524 | 13.3333 | 16.7302 |
| fivefold_rate_mean | None | 0.1 | T_missing_never | interior_shuffle_6601 | 5/5 | 24.9524 | 13.3333 | 16.7302 |
| fivefold_rate_mean | None | 0.1 | T_missing_never | interior_shuffle_6602 | 5/5 | 24.9524 | 13.3333 | 16.7302 |
| fivefold_rate_mean | None | 0.1 | T_missing_never | interior_shuffle_6603 | 5/5 | 24.9524 | 13.3333 | 16.7302 |
| fivefold_rate_mean | None | 0.2 | all | flat | 5/5 | 63.4364 | 61.6661 | 63.2661 |
| fivefold_rate_mean | None | 0.2 | all | normal | 5/5 | 63.9994 | 62.7048 | 64.0545 |
| fivefold_rate_mean | None | 0.2 | all | degree2_no_area | 5/5 | 63.9250 | 62.6797 | 63.9895 |
| fivefold_rate_mean | None | 0.2 | all | interior_shuffle_6601 | 5/5 | 63.9369 | 62.5899 | 63.9771 |
| fivefold_rate_mean | None | 0.2 | all | interior_shuffle_6602 | 5/5 | 63.8744 | 62.5316 | 63.9161 |
| fivefold_rate_mean | None | 0.2 | all | interior_shuffle_6603 | 5/5 | 63.9237 | 62.5964 | 63.9734 |
| fivefold_rate_mean | None | 0.2 | shuffle_eligible | flat | 5/5 | 64.0426 | 62.4682 | 63.8991 |
| fivefold_rate_mean | None | 0.2 | shuffle_eligible | normal | 5/5 | 64.5140 | 63.3877 | 64.6044 |
| fivefold_rate_mean | None | 0.2 | shuffle_eligible | degree2_no_area | 5/5 | 64.4191 | 63.3273 | 64.5200 |
| fivefold_rate_mean | None | 0.2 | shuffle_eligible | interior_shuffle_6601 | 5/5 | 64.4459 | 63.2726 | 64.5200 |
| fivefold_rate_mean | None | 0.2 | shuffle_eligible | interior_shuffle_6602 | 5/5 | 64.3778 | 63.2085 | 64.4536 |
| fivefold_rate_mean | None | 0.2 | shuffle_eligible | interior_shuffle_6603 | 5/5 | 64.4315 | 63.2741 | 64.5157 |
| fivefold_rate_mean | None | 0.2 | no_history | flat | 5/5 | 55.3043 | 38.1587 | 52.3165 |
| fivefold_rate_mean | None | 0.2 | no_history | normal | 5/5 | 55.3043 | 38.1587 | 52.3165 |
| fivefold_rate_mean | None | 0.2 | no_history | degree2_no_area | 5/5 | 55.3043 | 38.1587 | 52.3165 |
| fivefold_rate_mean | None | 0.2 | no_history | interior_shuffle_6601 | 5/5 | 55.3043 | 38.1587 | 52.3165 |
| fivefold_rate_mean | None | 0.2 | no_history | interior_shuffle_6602 | 5/5 | 55.3043 | 38.1587 | 52.3165 |
| fivefold_rate_mean | None | 0.2 | no_history | interior_shuffle_6603 | 5/5 | 55.3043 | 38.1587 | 52.3165 |
| fivefold_rate_mean | None | 0.2 | T_present | flat | 5/5 | 64.2022 | 62.4645 | 64.0027 |
| fivefold_rate_mean | None | 0.2 | T_present | normal | 5/5 | 64.6233 | 63.3566 | 64.6206 |
| fivefold_rate_mean | None | 0.2 | T_present | degree2_no_area | 5/5 | 64.5605 | 63.3156 | 64.5726 |
| fivefold_rate_mean | None | 0.2 | T_present | interior_shuffle_6601 | 5/5 | 64.5603 | 63.2442 | 64.5496 |
| fivefold_rate_mean | None | 0.2 | T_present | interior_shuffle_6602 | 5/5 | 64.4816 | 63.1775 | 64.4677 |
| fivefold_rate_mean | None | 0.2 | T_present | interior_shuffle_6603 | 5/5 | 64.5285 | 63.2622 | 64.5278 |
| fivefold_rate_mean | None | 0.2 | T_missing_seen | flat | 5/5 | 60.8173 | 58.8376 | 60.5943 |
| fivefold_rate_mean | None | 0.2 | T_missing_seen | normal | 5/5 | 61.9155 | 60.4017 | 62.2084 |
| fivefold_rate_mean | None | 0.2 | T_missing_seen | degree2_no_area | 5/5 | 61.7961 | 60.4020 | 62.0753 |
| fivefold_rate_mean | None | 0.2 | T_missing_seen | interior_shuffle_6601 | 5/5 | 61.8558 | 60.2657 | 62.1023 |
| fivefold_rate_mean | None | 0.2 | T_missing_seen | interior_shuffle_6602 | 5/5 | 61.8558 | 60.2465 | 62.1143 |
| fivefold_rate_mean | None | 0.2 | T_missing_seen | interior_shuffle_6603 | 5/5 | 61.9155 | 60.2065 | 62.1591 |
| fivefold_rate_mean | None | 0.2 | T_missing_never | flat | 5/5 | 54.7778 | 22.7778 | 51.4687 |
| fivefold_rate_mean | None | 0.2 | T_missing_never | normal | 5/5 | 54.7778 | 22.7778 | 51.4687 |
| fivefold_rate_mean | None | 0.2 | T_missing_never | degree2_no_area | 5/5 | 54.7778 | 22.7778 | 51.4687 |
| fivefold_rate_mean | None | 0.2 | T_missing_never | interior_shuffle_6601 | 5/5 | 54.7778 | 22.7778 | 51.4687 |
| fivefold_rate_mean | None | 0.2 | T_missing_never | interior_shuffle_6602 | 5/5 | 54.7778 | 22.7778 | 51.4687 |
| fivefold_rate_mean | None | 0.2 | T_missing_never | interior_shuffle_6603 | 5/5 | 54.7778 | 22.7778 | 51.4687 |
| fivefold_rate_mean | None | 0.3 | all | flat | 5/5 | 62.4583 | 61.2242 | 62.3363 |
| fivefold_rate_mean | None | 0.3 | all | normal | 5/5 | 62.9083 | 61.6834 | 62.7575 |
| fivefold_rate_mean | None | 0.3 | all | degree2_no_area | 5/5 | 62.9215 | 61.6914 | 62.7654 |
| fivefold_rate_mean | None | 0.3 | all | interior_shuffle_6601 | 5/5 | 62.8951 | 61.6840 | 62.7353 |
| fivefold_rate_mean | None | 0.3 | all | interior_shuffle_6602 | 5/5 | 62.9215 | 61.6859 | 62.7647 |
| fivefold_rate_mean | None | 0.3 | all | interior_shuffle_6603 | 5/5 | 62.9083 | 61.6839 | 62.7505 |
| fivefold_rate_mean | None | 0.3 | shuffle_eligible | flat | 5/5 | 62.6162 | 61.4285 | 62.5443 |
| fivefold_rate_mean | None | 0.3 | shuffle_eligible | normal | 5/5 | 63.0845 | 61.9280 | 62.9575 |
| fivefold_rate_mean | None | 0.3 | shuffle_eligible | degree2_no_area | 5/5 | 63.0845 | 61.9249 | 62.9526 |
| fivefold_rate_mean | None | 0.3 | shuffle_eligible | interior_shuffle_6601 | 5/5 | 63.0699 | 61.9298 | 62.9342 |
| fivefold_rate_mean | None | 0.3 | shuffle_eligible | interior_shuffle_6602 | 5/5 | 63.0990 | 61.9302 | 62.9667 |
| fivefold_rate_mean | None | 0.3 | shuffle_eligible | interior_shuffle_6603 | 5/5 | 63.0845 | 61.9291 | 62.9510 |
| fivefold_rate_mean | None | 0.3 | no_history | flat | 5/5 | 56.8404 | 43.0026 | 51.6847 |
| fivefold_rate_mean | None | 0.3 | no_history | normal | 5/5 | 56.8404 | 43.0026 | 51.6847 |
| fivefold_rate_mean | None | 0.3 | no_history | degree2_no_area | 5/5 | 56.8404 | 43.0026 | 51.6847 |
| fivefold_rate_mean | None | 0.3 | no_history | interior_shuffle_6601 | 5/5 | 56.8404 | 43.0026 | 51.6847 |
| fivefold_rate_mean | None | 0.3 | no_history | interior_shuffle_6602 | 5/5 | 56.8404 | 43.0026 | 51.6847 |
| fivefold_rate_mean | None | 0.3 | no_history | interior_shuffle_6603 | 5/5 | 56.8404 | 43.0026 | 51.6847 |
| fivefold_rate_mean | None | 0.3 | T_present | flat | 5/5 | 63.1636 | 61.9278 | 63.1396 |
| fivefold_rate_mean | None | 0.3 | T_present | normal | 5/5 | 63.4796 | 62.3032 | 63.4187 |
| fivefold_rate_mean | None | 0.3 | T_present | degree2_no_area | 5/5 | 63.4796 | 62.3459 | 63.4093 |
| fivefold_rate_mean | None | 0.3 | T_present | interior_shuffle_6601 | 5/5 | 63.4415 | 62.3098 | 63.3722 |
| fivefold_rate_mean | None | 0.3 | T_present | interior_shuffle_6602 | 5/5 | 63.4225 | 62.2604 | 63.3585 |
| fivefold_rate_mean | None | 0.3 | T_present | interior_shuffle_6603 | 5/5 | 63.4035 | 62.2331 | 63.3390 |
| fivefold_rate_mean | None | 0.3 | T_missing_seen | flat | 5/5 | 61.0699 | 60.1735 | 60.6248 |
| fivefold_rate_mean | None | 0.3 | T_missing_seen | normal | 5/5 | 61.8605 | 60.8015 | 61.3990 |
| fivefold_rate_mean | None | 0.3 | T_missing_seen | degree2_no_area | 5/5 | 61.9528 | 60.7804 | 61.4984 |
| fivefold_rate_mean | None | 0.3 | T_missing_seen | interior_shuffle_6601 | 5/5 | 61.9066 | 60.8045 | 61.4413 |
| fivefold_rate_mean | None | 0.3 | T_missing_seen | interior_shuffle_6602 | 5/5 | 62.0452 | 60.9274 | 61.5665 |
| fivefold_rate_mean | None | 0.3 | T_missing_seen | interior_shuffle_6603 | 5/5 | 62.0452 | 60.9720 | 61.5668 |
| fivefold_rate_mean | None | 0.3 | T_missing_never | flat | 5/5 | 52.9749 | 28.4444 | 44.5178 |
| fivefold_rate_mean | None | 0.3 | T_missing_never | normal | 5/5 | 52.9749 | 28.4444 | 44.5178 |
| fivefold_rate_mean | None | 0.3 | T_missing_never | degree2_no_area | 5/5 | 52.2342 | 28.1667 | 43.8284 |
| fivefold_rate_mean | None | 0.3 | T_missing_never | interior_shuffle_6601 | 5/5 | 52.9749 | 28.4444 | 44.5178 |
| fivefold_rate_mean | None | 0.3 | T_missing_never | interior_shuffle_6602 | 5/5 | 52.9749 | 28.4444 | 44.5178 |
| fivefold_rate_mean | None | 0.3 | T_missing_never | interior_shuffle_6603 | 5/5 | 52.9749 | 28.4444 | 44.5178 |
| fivefold_rate_mean | None | 0.4 | all | flat | 5/5 | 61.3066 | 60.1542 | 60.8166 |
| fivefold_rate_mean | None | 0.4 | all | normal | 5/5 | 62.0371 | 61.0324 | 62.1028 |
| fivefold_rate_mean | None | 0.4 | all | degree2_no_area | 5/5 | 61.9752 | 60.9936 | 62.0272 |
| fivefold_rate_mean | None | 0.4 | all | interior_shuffle_6601 | 5/5 | 62.0191 | 61.0067 | 62.0859 |
| fivefold_rate_mean | None | 0.4 | all | interior_shuffle_6602 | 5/5 | 61.9863 | 61.0087 | 62.0561 |
| fivefold_rate_mean | None | 0.4 | all | interior_shuffle_6603 | 5/5 | 62.0457 | 61.0846 | 62.1100 |
| fivefold_rate_mean | None | 0.4 | shuffle_eligible | flat | 5/5 | 61.8607 | 60.8593 | 61.3765 |
| fivefold_rate_mean | None | 0.4 | shuffle_eligible | normal | 5/5 | 62.6048 | 61.7897 | 62.7002 |
| fivefold_rate_mean | None | 0.4 | shuffle_eligible | degree2_no_area | 5/5 | 62.5378 | 61.7706 | 62.6170 |
| fivefold_rate_mean | None | 0.4 | shuffle_eligible | interior_shuffle_6601 | 5/5 | 62.5858 | 61.7625 | 62.6826 |
| fivefold_rate_mean | None | 0.4 | shuffle_eligible | interior_shuffle_6602 | 5/5 | 62.5491 | 61.7664 | 62.6514 |
| fivefold_rate_mean | None | 0.4 | shuffle_eligible | interior_shuffle_6603 | 5/5 | 62.6147 | 61.8443 | 62.7097 |
| fivefold_rate_mean | None | 0.4 | no_history | flat | 5/5 | 44.5336 | 32.9921 | 41.8744 |
| fivefold_rate_mean | None | 0.4 | no_history | normal | 5/5 | 44.5336 | 32.9921 | 41.8744 |
| fivefold_rate_mean | None | 0.4 | no_history | degree2_no_area | 5/5 | 44.5336 | 32.9921 | 41.8744 |
| fivefold_rate_mean | None | 0.4 | no_history | interior_shuffle_6601 | 5/5 | 44.5336 | 32.9921 | 41.8744 |
| fivefold_rate_mean | None | 0.4 | no_history | interior_shuffle_6602 | 5/5 | 44.5336 | 32.9921 | 41.8744 |
| fivefold_rate_mean | None | 0.4 | no_history | interior_shuffle_6603 | 5/5 | 44.5336 | 32.9921 | 41.8744 |
| fivefold_rate_mean | None | 0.4 | T_present | flat | 5/5 | 62.5945 | 61.2967 | 62.0904 |
| fivefold_rate_mean | None | 0.4 | T_present | normal | 5/5 | 63.1255 | 62.0037 | 63.1371 |
| fivefold_rate_mean | None | 0.4 | T_present | degree2_no_area | 5/5 | 62.9591 | 61.8599 | 62.9724 |
| fivefold_rate_mean | None | 0.4 | T_present | interior_shuffle_6601 | 5/5 | 63.0746 | 61.9397 | 63.0900 |
| fivefold_rate_mean | None | 0.4 | T_present | interior_shuffle_6602 | 5/5 | 63.0187 | 61.9350 | 63.0390 |
| fivefold_rate_mean | None | 0.4 | T_present | interior_shuffle_6603 | 5/5 | 63.0757 | 62.0152 | 63.0943 |
| fivefold_rate_mean | None | 0.4 | T_missing_seen | flat | 5/5 | 60.0524 | 59.1665 | 59.5190 |
| fivefold_rate_mean | None | 0.4 | T_missing_seen | normal | 5/5 | 61.0317 | 60.3910 | 61.1809 |
| fivefold_rate_mean | None | 0.4 | T_missing_seen | degree2_no_area | 5/5 | 61.1294 | 60.5194 | 61.2441 |
| fivefold_rate_mean | None | 0.4 | T_missing_seen | interior_shuffle_6601 | 5/5 | 61.0652 | 60.4503 | 61.2128 |
| fivefold_rate_mean | None | 0.4 | T_missing_seen | interior_shuffle_6602 | 5/5 | 61.0698 | 60.4414 | 61.2172 |
| fivefold_rate_mean | None | 0.4 | T_missing_seen | interior_shuffle_6603 | 5/5 | 61.1354 | 60.5384 | 61.2606 |
| fivefold_rate_mean | None | 0.4 | T_missing_never | flat | 5/5 | 40.9352 | 23.4293 | 38.9955 |
| fivefold_rate_mean | None | 0.4 | T_missing_never | normal | 5/5 | 42.8876 | 23.6092 | 40.3487 |
| fivefold_rate_mean | None | 0.4 | T_missing_never | degree2_no_area | 5/5 | 42.8876 | 23.6092 | 40.3487 |
| fivefold_rate_mean | None | 0.4 | T_missing_never | interior_shuffle_6601 | 5/5 | 42.8876 | 23.6092 | 40.3487 |
| fivefold_rate_mean | None | 0.4 | T_missing_never | interior_shuffle_6602 | 5/5 | 42.8876 | 23.6092 | 40.3487 |
| fivefold_rate_mean | None | 0.4 | T_missing_never | interior_shuffle_6603 | 5/5 | 42.8876 | 23.6092 | 40.3487 |
| fivefold_rate_mean | None | 0.5 | all | flat | 5/5 | 60.2277 | 58.6174 | 60.0129 |
| fivefold_rate_mean | None | 0.5 | all | normal | 5/5 | 61.3833 | 59.9908 | 61.3073 |
| fivefold_rate_mean | None | 0.5 | all | degree2_no_area | 5/5 | 61.1392 | 59.7470 | 61.0599 |
| fivefold_rate_mean | None | 0.5 | all | interior_shuffle_6601 | 5/5 | 61.2819 | 59.8570 | 61.1987 |
| fivefold_rate_mean | None | 0.5 | all | interior_shuffle_6602 | 5/5 | 61.2527 | 59.7957 | 61.1700 |
| fivefold_rate_mean | None | 0.5 | all | interior_shuffle_6603 | 5/5 | 61.3383 | 59.8901 | 61.2502 |
| fivefold_rate_mean | None | 0.5 | shuffle_eligible | flat | 5/5 | 61.1767 | 60.0725 | 61.0275 |
| fivefold_rate_mean | None | 0.5 | shuffle_eligible | normal | 5/5 | 62.2365 | 61.3193 | 62.2121 |
| fivefold_rate_mean | None | 0.5 | shuffle_eligible | degree2_no_area | 5/5 | 61.9920 | 61.0709 | 61.9615 |
| fivefold_rate_mean | None | 0.5 | shuffle_eligible | interior_shuffle_6601 | 5/5 | 62.1210 | 61.1686 | 62.0890 |
| fivefold_rate_mean | None | 0.5 | shuffle_eligible | interior_shuffle_6602 | 5/5 | 62.0878 | 61.1008 | 62.0578 |
| fivefold_rate_mean | None | 0.5 | shuffle_eligible | interior_shuffle_6603 | 5/5 | 62.1852 | 61.2118 | 62.1489 |
| fivefold_rate_mean | None | 0.5 | no_history | flat | 5/5 | 47.6010 | 33.9286 | 42.1262 |
| fivefold_rate_mean | None | 0.5 | no_history | normal | 5/5 | 47.6010 | 33.9286 | 42.1262 |
| fivefold_rate_mean | None | 0.5 | no_history | degree2_no_area | 5/5 | 47.6010 | 33.9286 | 42.1262 |
| fivefold_rate_mean | None | 0.5 | no_history | interior_shuffle_6601 | 5/5 | 47.6010 | 33.9286 | 42.1262 |
| fivefold_rate_mean | None | 0.5 | no_history | interior_shuffle_6602 | 5/5 | 47.6010 | 33.9286 | 42.1262 |
| fivefold_rate_mean | None | 0.5 | no_history | interior_shuffle_6603 | 5/5 | 47.6010 | 33.9286 | 42.1262 |
| fivefold_rate_mean | None | 0.5 | T_present | flat | 5/5 | 61.7201 | 59.8434 | 61.4419 |
| fivefold_rate_mean | None | 0.5 | T_present | normal | 5/5 | 62.6922 | 61.5845 | 62.6780 |
| fivefold_rate_mean | None | 0.5 | T_present | degree2_no_area | 5/5 | 62.5375 | 61.3418 | 62.5112 |
| fivefold_rate_mean | None | 0.5 | T_present | interior_shuffle_6601 | 5/5 | 62.6051 | 61.4043 | 62.5854 |
| fivefold_rate_mean | None | 0.5 | T_present | interior_shuffle_6602 | 5/5 | 62.5186 | 61.3030 | 62.5007 |
| fivefold_rate_mean | None | 0.5 | T_present | interior_shuffle_6603 | 5/5 | 62.6654 | 61.4697 | 62.6447 |
| fivefold_rate_mean | None | 0.5 | T_missing_seen | flat | 5/5 | 59.3546 | 57.9916 | 59.1851 |
| fivefold_rate_mean | None | 0.5 | T_missing_seen | normal | 5/5 | 60.6675 | 59.0157 | 60.5158 |
| fivefold_rate_mean | None | 0.5 | T_missing_seen | degree2_no_area | 5/5 | 60.3242 | 58.7500 | 60.1665 |
| fivefold_rate_mean | None | 0.5 | T_missing_seen | interior_shuffle_6601 | 5/5 | 60.5488 | 58.9187 | 60.3870 |
| fivefold_rate_mean | None | 0.5 | T_missing_seen | interior_shuffle_6602 | 5/5 | 60.5760 | 58.8968 | 60.4130 |
| fivefold_rate_mean | None | 0.5 | T_missing_seen | interior_shuffle_6603 | 5/5 | 60.6040 | 58.9149 | 60.4339 |
| fivefold_rate_mean | None | 0.5 | T_missing_never | flat | 5/5 | 44.4422 | 27.3690 | 38.6229 |
| fivefold_rate_mean | None | 0.5 | T_missing_never | normal | 5/5 | 46.8395 | 30.0304 | 42.5139 |
| fivefold_rate_mean | None | 0.5 | T_missing_never | degree2_no_area | 5/5 | 46.8395 | 30.0304 | 42.5139 |
| fivefold_rate_mean | None | 0.5 | T_missing_never | interior_shuffle_6601 | 5/5 | 46.8395 | 30.0304 | 42.5139 |
| fivefold_rate_mean | None | 0.5 | T_missing_never | interior_shuffle_6602 | 5/5 | 46.8395 | 30.0304 | 42.5139 |
| fivefold_rate_mean | None | 0.5 | T_missing_never | interior_shuffle_6603 | 5/5 | 46.8395 | 30.0304 | 42.5139 |
| fivefold_rate_mean | None | 0.6 | all | flat | 5/5 | 59.1887 | 58.3249 | 59.0434 |
| fivefold_rate_mean | None | 0.6 | all | normal | 5/5 | 60.1101 | 59.1893 | 60.1558 |
| fivefold_rate_mean | None | 0.6 | all | degree2_no_area | 5/5 | 59.8888 | 58.9422 | 59.9548 |
| fivefold_rate_mean | None | 0.6 | all | interior_shuffle_6601 | 5/5 | 59.9643 | 59.0453 | 60.0218 |
| fivefold_rate_mean | None | 0.6 | all | interior_shuffle_6602 | 5/5 | 59.9274 | 59.0223 | 59.9897 |
| fivefold_rate_mean | None | 0.6 | all | interior_shuffle_6603 | 5/5 | 59.9128 | 58.9886 | 59.9741 |
| fivefold_rate_mean | None | 0.6 | shuffle_eligible | flat | 5/5 | 59.9429 | 59.3861 | 59.8250 |
| fivefold_rate_mean | None | 0.6 | shuffle_eligible | normal | 5/5 | 60.8086 | 60.2148 | 60.9295 |
| fivefold_rate_mean | None | 0.6 | shuffle_eligible | degree2_no_area | 5/5 | 60.5534 | 59.9671 | 60.6989 |
| fivefold_rate_mean | None | 0.6 | shuffle_eligible | interior_shuffle_6601 | 5/5 | 60.6267 | 60.0438 | 60.7624 |
| fivefold_rate_mean | None | 0.6 | shuffle_eligible | interior_shuffle_6602 | 5/5 | 60.5812 | 60.0191 | 60.7226 |
| fivefold_rate_mean | None | 0.6 | shuffle_eligible | interior_shuffle_6603 | 5/5 | 60.5629 | 59.9791 | 60.7039 |
| fivefold_rate_mean | None | 0.6 | no_history | flat | 5/5 | 52.4238 | 42.3095 | 51.3950 |
| fivefold_rate_mean | None | 0.6 | no_history | normal | 5/5 | 52.4238 | 42.3095 | 51.3950 |
| fivefold_rate_mean | None | 0.6 | no_history | degree2_no_area | 5/5 | 52.4238 | 42.3095 | 51.3950 |
| fivefold_rate_mean | None | 0.6 | no_history | interior_shuffle_6601 | 5/5 | 52.4238 | 42.3095 | 51.3950 |
| fivefold_rate_mean | None | 0.6 | no_history | interior_shuffle_6602 | 5/5 | 52.4238 | 42.3095 | 51.3950 |
| fivefold_rate_mean | None | 0.6 | no_history | interior_shuffle_6603 | 5/5 | 52.4238 | 42.3095 | 51.3950 |
| fivefold_rate_mean | None | 0.6 | T_present | flat | 5/5 | 61.0617 | 60.4618 | 60.9385 |
| fivefold_rate_mean | None | 0.6 | T_present | normal | 5/5 | 62.5879 | 61.9433 | 62.6178 |
| fivefold_rate_mean | None | 0.6 | T_present | degree2_no_area | 5/5 | 62.1859 | 61.5800 | 62.2248 |
| fivefold_rate_mean | None | 0.6 | T_present | interior_shuffle_6601 | 5/5 | 62.4312 | 61.8149 | 62.4551 |
| fivefold_rate_mean | None | 0.6 | T_present | interior_shuffle_6602 | 5/5 | 62.2962 | 61.6962 | 62.3220 |
| fivefold_rate_mean | None | 0.6 | T_present | interior_shuffle_6603 | 5/5 | 62.3327 | 61.7318 | 62.3610 |
| fivefold_rate_mean | None | 0.6 | T_missing_seen | flat | 5/5 | 58.6394 | 57.9579 | 58.3568 |
| fivefold_rate_mean | None | 0.6 | T_missing_seen | normal | 5/5 | 59.1859 | 58.4408 | 59.1656 |
| fivefold_rate_mean | None | 0.6 | T_missing_seen | degree2_no_area | 5/5 | 59.1027 | 58.3385 | 59.1088 |
| fivefold_rate_mean | None | 0.6 | T_missing_seen | interior_shuffle_6601 | 5/5 | 59.0386 | 58.2623 | 59.0357 |
| fivefold_rate_mean | None | 0.6 | T_missing_seen | interior_shuffle_6602 | 5/5 | 59.0687 | 58.3117 | 59.0710 |
| fivefold_rate_mean | None | 0.6 | T_missing_seen | interior_shuffle_6603 | 5/5 | 59.0172 | 58.2223 | 59.0234 |
| fivefold_rate_mean | None | 0.6 | T_missing_never | flat | 5/5 | 45.7647 | 31.1865 | 43.1226 |
| fivefold_rate_mean | None | 0.6 | T_missing_never | normal | 5/5 | 46.1596 | 32.3355 | 43.0567 |
| fivefold_rate_mean | None | 0.6 | T_missing_never | degree2_no_area | 5/5 | 45.7515 | 32.1133 | 42.8137 |
| fivefold_rate_mean | None | 0.6 | T_missing_never | interior_shuffle_6601 | 5/5 | 46.1596 | 32.3355 | 43.0567 |
| fivefold_rate_mean | None | 0.6 | T_missing_never | interior_shuffle_6602 | 5/5 | 46.1596 | 32.3355 | 43.0567 |
| fivefold_rate_mean | None | 0.6 | T_missing_never | interior_shuffle_6603 | 5/5 | 46.1596 | 32.3355 | 43.0567 |
| fivefold_rate_mean | None | 0.7 | all | flat | 5/5 | 57.4535 | 56.8730 | 57.3903 |
| fivefold_rate_mean | None | 0.7 | all | normal | 5/5 | 58.3227 | 57.4231 | 58.2927 |
| fivefold_rate_mean | None | 0.7 | all | degree2_no_area | 5/5 | 58.2698 | 57.4128 | 58.2234 |
| fivefold_rate_mean | None | 0.7 | all | interior_shuffle_6601 | 5/5 | 58.1189 | 57.2494 | 58.0989 |
| fivefold_rate_mean | None | 0.7 | all | interior_shuffle_6602 | 5/5 | 58.1197 | 57.2067 | 58.1005 |
| fivefold_rate_mean | None | 0.7 | all | interior_shuffle_6603 | 5/5 | 58.0819 | 57.1856 | 58.0528 |
| fivefold_rate_mean | None | 0.7 | shuffle_eligible | flat | 5/5 | 58.8479 | 57.7129 | 58.6818 |
| fivefold_rate_mean | None | 0.7 | shuffle_eligible | normal | 5/5 | 59.7526 | 58.3437 | 59.6325 |
| fivefold_rate_mean | None | 0.7 | shuffle_eligible | degree2_no_area | 5/5 | 59.6249 | 58.2082 | 59.4657 |
| fivefold_rate_mean | None | 0.7 | shuffle_eligible | interior_shuffle_6601 | 5/5 | 59.4479 | 58.0692 | 59.3395 |
| fivefold_rate_mean | None | 0.7 | shuffle_eligible | interior_shuffle_6602 | 5/5 | 59.4505 | 58.0200 | 59.3425 |
| fivefold_rate_mean | None | 0.7 | shuffle_eligible | interior_shuffle_6603 | 5/5 | 59.3941 | 57.9755 | 59.2729 |
| fivefold_rate_mean | None | 0.7 | no_history | flat | 5/5 | 49.8619 | 34.9603 | 46.2922 |
| fivefold_rate_mean | None | 0.7 | no_history | normal | 5/5 | 49.8619 | 34.9603 | 46.2922 |
| fivefold_rate_mean | None | 0.7 | no_history | degree2_no_area | 5/5 | 49.8619 | 34.9603 | 46.2922 |
| fivefold_rate_mean | None | 0.7 | no_history | interior_shuffle_6601 | 5/5 | 49.8619 | 34.9603 | 46.2922 |
| fivefold_rate_mean | None | 0.7 | no_history | interior_shuffle_6602 | 5/5 | 49.8619 | 34.9603 | 46.2922 |
| fivefold_rate_mean | None | 0.7 | no_history | interior_shuffle_6603 | 5/5 | 49.8619 | 34.9603 | 46.2922 |
| fivefold_rate_mean | None | 0.7 | T_present | flat | 5/5 | 59.5195 | 59.1646 | 59.6073 |
| fivefold_rate_mean | None | 0.7 | T_present | normal | 5/5 | 59.6146 | 59.0492 | 59.7285 |
| fivefold_rate_mean | None | 0.7 | T_present | degree2_no_area | 5/5 | 59.5798 | 58.9793 | 59.6694 |
| fivefold_rate_mean | None | 0.7 | T_present | interior_shuffle_6601 | 5/5 | 59.3166 | 58.7845 | 59.4705 |
| fivefold_rate_mean | None | 0.7 | T_present | interior_shuffle_6602 | 5/5 | 59.3542 | 58.8165 | 59.5090 |
| fivefold_rate_mean | None | 0.7 | T_present | interior_shuffle_6603 | 5/5 | 59.3535 | 58.7843 | 59.4988 |
| fivefold_rate_mean | None | 0.7 | T_missing_seen | flat | 5/5 | 57.0327 | 56.5619 | 56.9108 |
| fivefold_rate_mean | None | 0.7 | T_missing_seen | normal | 5/5 | 58.2117 | 57.3164 | 58.1309 |
| fivefold_rate_mean | None | 0.7 | T_missing_seen | degree2_no_area | 5/5 | 58.1425 | 57.3181 | 58.0561 |
| fivefold_rate_mean | None | 0.7 | T_missing_seen | interior_shuffle_6601 | 5/5 | 58.0457 | 57.1652 | 57.9590 |
| fivefold_rate_mean | None | 0.7 | T_missing_seen | interior_shuffle_6602 | 5/5 | 58.0272 | 57.0905 | 57.9388 |
| fivefold_rate_mean | None | 0.7 | T_missing_seen | interior_shuffle_6603 | 5/5 | 57.9676 | 57.0665 | 57.8705 |
| fivefold_rate_mean | None | 0.7 | T_missing_never | flat | 5/5 | 45.7958 | 35.0483 | 43.1040 |
| fivefold_rate_mean | None | 0.7 | T_missing_never | normal | 5/5 | 48.2016 | 37.5653 | 45.9829 |
| fivefold_rate_mean | None | 0.7 | T_missing_never | degree2_no_area | 5/5 | 48.2016 | 37.3801 | 45.6879 |
| fivefold_rate_mean | None | 0.7 | T_missing_never | interior_shuffle_6601 | 5/5 | 48.2016 | 37.5653 | 45.9829 |
| fivefold_rate_mean | None | 0.7 | T_missing_never | interior_shuffle_6602 | 5/5 | 48.2016 | 37.5653 | 45.9829 |
| fivefold_rate_mean | None | 0.7 | T_missing_never | interior_shuffle_6603 | 5/5 | 48.2016 | 37.5653 | 45.9829 |
| fivefold_eight_rate_mean | None | mean | all | flat | 40/40 | 61.5072 | 60.3071 | 61.2891 |
| fivefold_eight_rate_mean | None | mean | all | normal | 40/40 | 62.2734 | 61.2362 | 62.2721 |
| fivefold_eight_rate_mean | None | mean | all | degree2_no_area | 40/40 | 62.1474 | 61.1157 | 62.1459 |
| fivefold_eight_rate_mean | None | mean | all | interior_shuffle_6601 | 40/40 | 62.1746 | 61.1304 | 62.1697 |
| fivefold_eight_rate_mean | None | mean | all | interior_shuffle_6602 | 40/40 | 62.1449 | 61.0921 | 62.1426 |
| fivefold_eight_rate_mean | None | mean | all | interior_shuffle_6603 | 40/40 | 62.1478 | 61.1065 | 62.1418 |
| fivefold_eight_rate_mean | None | mean | shuffle_eligible | flat | 40/40 | 62.1074 | 61.0251 | 61.9064 |
| fivefold_eight_rate_mean | None | mean | shuffle_eligible | normal | 40/40 | 62.8339 | 61.9121 | 62.8601 |
| fivefold_eight_rate_mean | None | mean | shuffle_eligible | degree2_no_area | 40/40 | 62.6834 | 61.7653 | 62.7075 |
| fivefold_eight_rate_mean | None | mean | shuffle_eligible | interior_shuffle_6601 | 40/40 | 62.7123 | 61.7854 | 62.7355 |
| fivefold_eight_rate_mean | None | mean | shuffle_eligible | interior_shuffle_6602 | 40/40 | 62.6793 | 61.7419 | 62.7057 |
| fivefold_eight_rate_mean | None | mean | shuffle_eligible | interior_shuffle_6603 | 40/40 | 62.6808 | 61.7563 | 62.7023 |
| fivefold_eight_rate_mean | None | mean | no_history | flat | 40/40 | 54.7317 | 40.4646 | 51.2306 |
| fivefold_eight_rate_mean | None | mean | no_history | normal | 40/40 | 54.7317 | 40.4646 | 51.2306 |
| fivefold_eight_rate_mean | None | mean | no_history | degree2_no_area | 40/40 | 54.7317 | 40.4646 | 51.2306 |
| fivefold_eight_rate_mean | None | mean | no_history | interior_shuffle_6601 | 40/40 | 54.7317 | 40.4646 | 51.2306 |
| fivefold_eight_rate_mean | None | mean | no_history | interior_shuffle_6602 | 40/40 | 54.7317 | 40.4646 | 51.2306 |
| fivefold_eight_rate_mean | None | mean | no_history | interior_shuffle_6603 | 40/40 | 54.7317 | 40.4646 | 51.2306 |
| fivefold_eight_rate_mean | None | mean | T_present | flat | 40/40 | 62.5409 | 61.3576 | 62.3486 |
| fivefold_eight_rate_mean | None | mean | T_present | normal | 40/40 | 63.2134 | 62.2849 | 63.2365 |
| fivefold_eight_rate_mean | None | mean | T_present | degree2_no_area | 40/40 | 63.0646 | 62.1299 | 63.0862 |
| fivefold_eight_rate_mean | None | mean | T_present | interior_shuffle_6601 | 40/40 | 63.0973 | 62.1611 | 63.1199 |
| fivefold_eight_rate_mean | None | mean | T_present | interior_shuffle_6602 | 40/40 | 63.0389 | 62.1028 | 63.0636 |
| fivefold_eight_rate_mean | None | mean | T_present | interior_shuffle_6603 | 40/40 | 63.0633 | 62.1399 | 63.0865 |
| fivefold_eight_rate_mean | None | mean | T_missing_seen | flat | 35/40 | 60.0254 | 58.9054 | 59.6346 |
| fivefold_eight_rate_mean | None | mean | T_missing_seen | normal | 35/40 | 60.8697 | 59.7741 | 60.7726 |
| fivefold_eight_rate_mean | None | mean | T_missing_seen | degree2_no_area | 35/40 | 60.8090 | 59.7503 | 60.7046 |
| fivefold_eight_rate_mean | None | mean | T_missing_seen | interior_shuffle_6601 | 35/40 | 60.7942 | 59.6907 | 60.6866 |
| fivefold_eight_rate_mean | None | mean | T_missing_seen | interior_shuffle_6602 | 35/40 | 60.8536 | 59.7344 | 60.7496 |
| fivefold_eight_rate_mean | None | mean | T_missing_seen | interior_shuffle_6603 | 35/40 | 60.8095 | 59.6733 | 60.6957 |
| fivefold_eight_rate_mean | None | mean | T_missing_never | flat | 35/40 | 44.2347 | 25.9413 | 39.5088 |
| fivefold_eight_rate_mean | None | mean | T_missing_never | normal | 35/40 | 45.2562 | 26.8709 | 40.6598 |
| fivefold_eight_rate_mean | None | mean | T_missing_never | degree2_no_area | 35/40 | 45.0921 | 26.7730 | 40.4845 |
| fivefold_eight_rate_mean | None | mean | T_missing_never | interior_shuffle_6601 | 35/40 | 45.2562 | 26.8709 | 40.6598 |
| fivefold_eight_rate_mean | None | mean | T_missing_never | interior_shuffle_6602 | 35/40 | 45.2562 | 26.8709 | 40.6598 |
| fivefold_eight_rate_mean | None | mean | T_missing_never | interior_shuffle_6603 | 35/40 | 45.2562 | 26.8709 | 40.6598 |
| fivefold_high_rate_mean | None | high | all | flat | 15/15 | 58.9567 | 57.9384 | 58.8155 |
| fivefold_high_rate_mean | None | high | all | normal | 15/15 | 59.9387 | 58.8677 | 59.9186 |
| fivefold_high_rate_mean | None | high | all | degree2_no_area | 15/15 | 59.7660 | 58.7007 | 59.7460 |
| fivefold_high_rate_mean | None | high | all | interior_shuffle_6601 | 15/15 | 59.7884 | 58.7172 | 59.7731 |
| fivefold_high_rate_mean | None | high | all | interior_shuffle_6602 | 15/15 | 59.7666 | 58.6749 | 59.7534 |
| fivefold_high_rate_mean | None | high | all | interior_shuffle_6603 | 15/15 | 59.7777 | 58.6881 | 59.7591 |
| fivefold_high_rate_mean | None | high | shuffle_eligible | flat | 15/15 | 59.9892 | 59.0572 | 59.8448 |
| fivefold_high_rate_mean | None | high | shuffle_eligible | normal | 15/15 | 60.9326 | 59.9593 | 60.9247 |
| fivefold_high_rate_mean | None | high | shuffle_eligible | degree2_no_area | 15/15 | 60.7234 | 59.7487 | 60.7087 |
| fivefold_high_rate_mean | None | high | shuffle_eligible | interior_shuffle_6601 | 15/15 | 60.7318 | 59.7605 | 60.7303 |
| fivefold_high_rate_mean | None | high | shuffle_eligible | interior_shuffle_6602 | 15/15 | 60.7065 | 59.7133 | 60.7076 |
| fivefold_high_rate_mean | None | high | shuffle_eligible | interior_shuffle_6603 | 15/15 | 60.7140 | 59.7221 | 60.7086 |
| fivefold_high_rate_mean | None | high | no_history | flat | 15/15 | 49.9622 | 37.0661 | 46.6045 |
| fivefold_high_rate_mean | None | high | no_history | normal | 15/15 | 49.9622 | 37.0661 | 46.6045 |
| fivefold_high_rate_mean | None | high | no_history | degree2_no_area | 15/15 | 49.9622 | 37.0661 | 46.6045 |
| fivefold_high_rate_mean | None | high | no_history | interior_shuffle_6601 | 15/15 | 49.9622 | 37.0661 | 46.6045 |
| fivefold_high_rate_mean | None | high | no_history | interior_shuffle_6602 | 15/15 | 49.9622 | 37.0661 | 46.6045 |
| fivefold_high_rate_mean | None | high | no_history | interior_shuffle_6603 | 15/15 | 49.9622 | 37.0661 | 46.6045 |
| fivefold_high_rate_mean | None | high | T_present | flat | 15/15 | 60.7671 | 59.8233 | 60.6626 |
| fivefold_high_rate_mean | None | high | T_present | normal | 15/15 | 61.6316 | 60.8590 | 61.6748 |
| fivefold_high_rate_mean | None | high | T_present | degree2_no_area | 15/15 | 61.4344 | 60.6337 | 61.4685 |
| fivefold_high_rate_mean | None | high | T_present | interior_shuffle_6601 | 15/15 | 61.4510 | 60.6679 | 61.5036 |
| fivefold_high_rate_mean | None | high | T_present | interior_shuffle_6602 | 15/15 | 61.3897 | 60.6053 | 61.4439 |
| fivefold_high_rate_mean | None | high | T_present | interior_shuffle_6603 | 15/15 | 61.4505 | 60.6620 | 61.5015 |
| fivefold_high_rate_mean | None | high | T_missing_seen | flat | 15/15 | 58.3422 | 57.5038 | 58.1509 |
| fivefold_high_rate_mean | None | high | T_missing_seen | normal | 15/15 | 59.3550 | 58.2576 | 59.2708 |
| fivefold_high_rate_mean | None | high | T_missing_seen | degree2_no_area | 15/15 | 59.1898 | 58.1355 | 59.1105 |
| fivefold_high_rate_mean | None | high | T_missing_seen | interior_shuffle_6601 | 15/15 | 59.2110 | 58.1154 | 59.1272 |
| fivefold_high_rate_mean | None | high | T_missing_seen | interior_shuffle_6602 | 15/15 | 59.2240 | 58.0997 | 59.1409 |
| fivefold_high_rate_mean | None | high | T_missing_seen | interior_shuffle_6603 | 15/15 | 59.1963 | 58.0679 | 59.1093 |
| fivefold_high_rate_mean | None | high | T_missing_never | flat | 15/15 | 45.3342 | 31.2013 | 41.6165 |
| fivefold_high_rate_mean | None | high | T_missing_never | normal | 15/15 | 47.0669 | 33.3104 | 43.8512 |
| fivefold_high_rate_mean | None | high | T_missing_never | degree2_no_area | 15/15 | 46.9309 | 33.1746 | 43.6719 |
| fivefold_high_rate_mean | None | high | T_missing_never | interior_shuffle_6601 | 15/15 | 47.0669 | 33.3104 | 43.8512 |
| fivefold_high_rate_mean | None | high | T_missing_never | interior_shuffle_6602 | 15/15 | 47.0669 | 33.3104 | 43.8512 |
| fivefold_high_rate_mean | None | high | T_missing_never | interior_shuffle_6603 | 15/15 | 47.0669 | 33.3104 | 43.8512 |

## Limits

- Endpoint-preserving shuffles test only interior content/time assignment, not all history order.
- Removing antisymmetric level-two area preserves the first level and symmetric terms; this is a frozen, potentially out-of-distribution intervention.
- CHM itself remains order-sensitive. This audit cannot prove Flat is order-blind.
- Six-class UA includes zero recall for classes absent in a cohort, consistent with the original fixed-six-class metric.
- Seed68 was selected for inspection after observing results; this is exploratory, not independent generalization evidence.
- Per-class recall/support and cohort rows are in JSON/CSV; no incomplete fivefold/eight-rate main summary is emitted.
