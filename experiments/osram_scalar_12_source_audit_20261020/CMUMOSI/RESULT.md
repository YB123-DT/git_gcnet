# OSRAM scalar 12-Source Evidence Audit

INTERNAL DIAGNOSTIC ONLY

Dataset CMUMOSI; seed 66; fold 1; source `1c5f29cb6c1592cdf0afe8874afc34a51ccd7938`.

Fixed original miss=0.3 checkpoint: `/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_66/flat/best_miss_0p3.pt`.

Checkpoint SHA256: `98640a71e7082751c98b66e61463fc9f1b81078e3b90f8c234df1be86d6c730e`.

No training; original Flat, Local Skip, Memory, Query and task head are unchanged. One fixed checkpoint is used for every rate/pattern, not the eight per-rate selected baseline models. The anchor was previously selected by Test-oracle; these are internal frozen interventions, not independent generalization results.

Rows are read destinations Base/Gap-A/Gap-T/Gap-V; columns are historical A/T/V Value-write sources. Each cell reports the final score after removing ONLY that one source contribution at the final Flat input. Local and all other evidence remain unchanged. Sources are algebraic joint-write Value-slot contributions, not pure modality semantics.

ACC/W-F1 use abs(gold)>0 and prediction>0; prediction=0 is nonpositive. MSE includes all valid labels. Each Gap row uses only utterances where that target modality is currently missing; its original score is recomputed on the identical cohort. Inactive/empty rows are N/A. Do not compare rows with different cohorts as if they were full-test scores.

Positive ACC/W-F1 Δ means removing the item improves that score; positive MSE Δ means removing it increases loss. Corrections/harms compare removed vs full on nonzero-label utterances only. No subgroup W-F1 values are averaged to reconstruct full-test W-F1.

## Full-model metrics by condition

| Condition | N | Polarity N | ACC | W-F1 | MSE |
|---|---:|---:|---:|---:|---:|
| random 0.0 | 686 | 656 | 86.2805 | 86.2270 | 0.9830 |
| random 0.1 | 686 | 656 | 83.3841 | 83.3347 | 1.1339 |
| random 0.2 | 686 | 656 | 81.8598 | 81.7469 | 1.3062 |
| random 0.3 | 686 | 656 | 82.1646 | 82.1220 | 1.2599 |
| random 0.4 | 686 | 656 | 76.9817 | 77.0180 | 1.5229 |
| random 0.5 | 686 | 656 | 74.0854 | 74.0977 | 1.5948 |
| random 0.6 | 686 | 656 | 72.4085 | 72.4280 | 1.7946 |
| random 0.7 | 686 | 656 | 66.9207 | 67.0113 | 2.0640 |
| persistent A | 686 | 656 | 58.9939 | 57.7876 | 2.5301 |
| persistent T | 686 | 656 | 85.9756 | 86.0006 | 1.0643 |
| persistent V | 686 | 656 | 55.0305 | 53.8849 | 2.8740 |
| persistent AT | 686 | 656 | 86.2805 | 86.1702 | 0.9489 |
| persistent AV | 686 | 656 | 60.0610 | 59.5079 | 2.4802 |
| persistent TV | 686 | 656 | 85.2134 | 85.2239 | 1.0150 |
| persistent ATV | 686 | 656 | 86.2805 | 86.2270 | 0.9830 |
| random8 | — | — | 78.0107 | 77.9982 | 1.4574 |
| high_missing | — | — | 71.1382 | 71.1790 | 1.8178 |
| persistent6 | — | — | 71.9258 | 71.4292 | 1.8187 |

Aggregates are equal-weight full-condition means; Persistent ATV is separate from persistent6.

## All 15 conditions: 4×3 source removal tables

### random 0.0

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 86.2805 | 86.2805 (+0.0000) | 86.2805 (+0.0000) | 85.8232 (-0.4573) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 86.2270 | 86.2516 (+0.0246) | 86.2437 (+0.0167) | 85.7893 (-0.4378) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 0.9830 | 0.9822 (-0.0008) | 1.0038 (+0.0208) | 0.9870 (+0.0040) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 7.7243 | 8.4349 | 8.1655 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

### random 0.1

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 83.3841 | 83.8415 (+0.4573) | 83.5366 (+0.1524) | 83.2317 (-0.1524) |
| Gap-A | 83 | 80 | 80.0000 | 80.0000 (+0.0000) | 80.0000 (+0.0000) | 80.0000 (+0.0000) |
| Gap-T | 65 | 62 | 61.2903 | 59.6774 (-1.6129) | 56.4516 (-4.8387) | 62.9032 (+1.6129) |
| Gap-V | 56 | 52 | 78.8462 | 80.7692 (+1.9231) | 78.8462 (+0.0000) | 82.6923 (+3.8462) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 83.3347 | 83.8335 (+0.4988) | 83.5111 (+0.1764) | 83.2147 (-0.1199) |
| Gap-A | 83 | 80 | 79.8869 | 79.8869 (+0.0000) | 79.8869 (+0.0000) | 79.8869 (+0.0000) |
| Gap-T | 65 | 62 | 61.2903 | 59.5348 (-1.7555) | 56.8532 (-4.4371) | 63.0128 (+1.7225) |
| Gap-V | 56 | 52 | 79.2301 | 81.1986 (+1.9685) | 79.2301 (+0.0000) | 83.0065 (+3.7763) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 1.1339 | 1.1367 (+0.0028) | 1.1611 (+0.0273) | 1.1402 (+0.0063) |
| Gap-A | 83 | 80 | 1.3662 | 1.3677 (+0.0015) | 1.3633 (-0.0028) | 1.3700 (+0.0038) |
| Gap-T | 65 | 62 | 2.3418 | 2.3268 (-0.0150) | 2.3922 (+0.0505) | 2.3645 (+0.0227) |
| Gap-V | 56 | 52 | 1.4517 | 1.4537 (+0.0020) | 1.4566 (+0.0050) | 1.4530 (+0.0014) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 7.5251 | 8.4561 | 8.1486 |
| Gap-A | 7.3992 | 3.1645 | 3.2462 |
| Gap-T | 2.5660 | 8.4947 | 3.4342 |
| Gap-V | 2.1238 | 3.3847 | 10.3341 |

### random 0.2

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 81.8598 | 81.7073 (-0.1524) | 81.8598 (+0.0000) | 81.4024 (-0.4573) |
| Gap-A | 133 | 127 | 84.2520 | 84.2520 (+0.0000) | 84.2520 (+0.0000) | 84.2520 (+0.0000) |
| Gap-T | 150 | 145 | 65.5172 | 64.1379 (-1.3793) | 66.2069 (+0.6897) | 63.4483 (-2.0690) |
| Gap-V | 136 | 132 | 81.8182 | 81.8182 (+0.0000) | 81.8182 (+0.0000) | 80.3030 (-1.5152) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 81.7469 | 81.6361 (-0.1108) | 81.7716 (+0.0247) | 81.3181 (-0.4288) |
| Gap-A | 133 | 127 | 84.5883 | 84.5883 (+0.0000) | 84.5883 (+0.0000) | 84.5883 (+0.0000) |
| Gap-T | 150 | 145 | 65.0218 | 63.4730 (-1.5488) | 65.5040 (+0.4822) | 62.8489 (-2.1729) |
| Gap-V | 136 | 132 | 81.6861 | 81.6861 (+0.0000) | 81.6861 (+0.0000) | 80.0724 (-1.6137) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 1.3062 | 1.3079 (+0.0016) | 1.3345 (+0.0283) | 1.3144 (+0.0082) |
| Gap-A | 133 | 127 | 1.0173 | 1.0130 (-0.0043) | 1.0270 (+0.0097) | 1.0180 (+0.0007) |
| Gap-T | 150 | 145 | 2.6024 | 2.6141 (+0.0118) | 2.7030 (+0.1006) | 2.5886 (-0.0138) |
| Gap-V | 136 | 132 | 1.3259 | 1.3302 (+0.0043) | 1.3292 (+0.0034) | 1.3281 (+0.0023) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 7.4057 | 8.3270 | 8.0980 |
| Gap-A | 7.4171 | 3.6236 | 3.5122 |
| Gap-T | 2.7270 | 8.2380 | 3.7717 |
| Gap-V | 2.3926 | 3.4494 | 9.7115 |

### random 0.3

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 82.1646 | 81.7073 (-0.4573) | 82.0122 (-0.1524) | 81.5549 (-0.6098) |
| Gap-A | 205 | 198 | 84.8485 | 84.3434 (-0.5051) | 84.3434 (-0.5051) | 84.8485 (+0.0000) |
| Gap-T | 221 | 214 | 66.8224 | 66.8224 (+0.0000) | 67.7570 (+0.9346) | 65.4206 (-1.4019) |
| Gap-V | 190 | 182 | 89.0110 | 89.5604 (+0.5495) | 90.1099 (+1.0989) | 86.8132 (-2.1978) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 82.1220 | 81.6982 (-0.4238) | 81.9743 (-0.1477) | 81.5504 (-0.5717) |
| Gap-A | 205 | 198 | 84.8485 | 84.3503 (-0.4982) | 84.3358 (-0.5127) | 84.8485 (+0.0000) |
| Gap-T | 221 | 214 | 67.0097 | 66.9399 (-0.0698) | 68.0004 (+0.9906) | 65.6495 (-1.3602) |
| Gap-V | 190 | 182 | 89.0110 | 89.5388 (+0.5278) | 90.0680 (+1.0570) | 86.7573 (-2.2536) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 1.2599 | 1.2682 (+0.0083) | 1.2896 (+0.0297) | 1.2772 (+0.0173) |
| Gap-A | 205 | 198 | 1.1268 | 1.1208 (-0.0060) | 1.1359 (+0.0091) | 1.1267 (-0.0001) |
| Gap-T | 221 | 214 | 2.0263 | 2.0340 (+0.0076) | 2.1122 (+0.0859) | 2.0575 (+0.0311) |
| Gap-V | 190 | 182 | 1.0463 | 1.0399 (-0.0065) | 1.0347 (-0.0117) | 1.0400 (-0.0064) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 7.0828 | 8.1170 | 8.1014 |
| Gap-A | 6.8500 | 3.7462 | 3.6195 |
| Gap-T | 2.9571 | 7.8137 | 4.0972 |
| Gap-V | 2.5411 | 3.7313 | 10.0584 |

### random 0.4

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 76.9817 | 77.7439 (+0.7622) | 76.9817 (+0.0000) | 77.2866 (+0.3049) |
| Gap-A | 260 | 251 | 76.8924 | 76.8924 (+0.0000) | 76.8924 (+0.0000) | 76.8924 (+0.0000) |
| Gap-T | 277 | 265 | 64.5283 | 64.5283 (+0.0000) | 61.8868 (-2.6415) | 64.9057 (+0.3774) |
| Gap-V | 261 | 252 | 78.9683 | 79.3651 (+0.3968) | 78.5714 (-0.3968) | 78.1746 (-0.7937) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 77.0180 | 77.8234 (+0.8054) | 77.0363 (+0.0183) | 77.3405 (+0.3224) |
| Gap-A | 260 | 251 | 77.0233 | 77.0233 (+0.0000) | 77.0233 (+0.0000) | 77.0233 (+0.0000) |
| Gap-T | 277 | 265 | 64.6871 | 64.7062 (+0.0192) | 62.0181 (-2.6689) | 65.0517 (+0.3647) |
| Gap-V | 261 | 252 | 78.8941 | 79.3174 (+0.4233) | 78.4127 (-0.4814) | 77.9829 (-0.9112) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 1.5229 | 1.5381 (+0.0152) | 1.5426 (+0.0197) | 1.5433 (+0.0204) |
| Gap-A | 260 | 251 | 1.5628 | 1.5668 (+0.0040) | 1.5680 (+0.0052) | 1.5668 (+0.0040) |
| Gap-T | 277 | 265 | 2.2439 | 2.2475 (+0.0036) | 2.2829 (+0.0390) | 2.2291 (-0.0148) |
| Gap-V | 261 | 252 | 1.4302 | 1.4340 (+0.0039) | 1.4277 (-0.0024) | 1.4068 (-0.0234) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 6.8579 | 8.1434 | 8.0182 |
| Gap-A | 6.6864 | 4.0653 | 4.0332 |
| Gap-T | 2.9446 | 7.9441 | 4.5502 |
| Gap-V | 2.9963 | 3.7764 | 9.9829 |

### random 0.5

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 74.0854 | 73.9329 (-0.1524) | 73.0183 (-1.0671) | 73.9329 (-0.1524) |
| Gap-A | 355 | 338 | 77.5148 | 77.8107 (+0.2959) | 77.5148 (+0.0000) | 77.5148 (+0.0000) |
| Gap-T | 344 | 327 | 62.6911 | 62.9969 (+0.3058) | 61.4679 (-1.2232) | 62.9969 (+0.3058) |
| Gap-V | 347 | 331 | 74.3202 | 74.3202 (+0.0000) | 74.6224 (+0.3021) | 74.3202 (+0.0000) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 74.0977 | 73.9948 (-0.1029) | 73.0494 (-1.0483) | 74.0134 (-0.0843) |
| Gap-A | 355 | 338 | 77.6365 | 77.9336 (+0.2971) | 77.6419 (+0.0054) | 77.6365 (+0.0000) |
| Gap-T | 344 | 327 | 62.6650 | 63.0305 (+0.3655) | 61.3778 (-1.2871) | 63.0305 (+0.3655) |
| Gap-V | 347 | 331 | 74.2235 | 74.2235 (+0.0000) | 74.5059 (+0.2825) | 73.9873 (-0.2361) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 1.5948 | 1.6107 (+0.0159) | 1.6315 (+0.0367) | 1.6096 (+0.0148) |
| Gap-A | 355 | 338 | 1.4243 | 1.4273 (+0.0029) | 1.4315 (+0.0072) | 1.4269 (+0.0026) |
| Gap-T | 344 | 327 | 2.2326 | 2.2347 (+0.0021) | 2.3043 (+0.0718) | 2.2428 (+0.0103) |
| Gap-V | 347 | 331 | 1.4319 | 1.4280 (-0.0039) | 1.4206 (-0.0113) | 1.4189 (-0.0130) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 6.4486 | 8.0662 | 7.9354 |
| Gap-A | 6.3729 | 4.2218 | 4.1611 |
| Gap-T | 3.3206 | 7.7147 | 5.3544 |
| Gap-V | 3.1519 | 4.1678 | 9.7494 |

### random 0.6

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 72.4085 | 72.2561 (-0.1524) | 71.9512 (-0.4573) | 72.2561 (-0.1524) |
| Gap-A | 406 | 390 | 75.1282 | 75.3846 (+0.2564) | 75.1282 (+0.0000) | 75.3846 (+0.2564) |
| Gap-T | 415 | 400 | 63.5000 | 64.0000 (+0.5000) | 62.5000 (-1.0000) | 63.5000 (+0.0000) |
| Gap-V | 419 | 399 | 77.1930 | 77.4436 (+0.2506) | 78.1955 (+1.0025) | 78.4461 (+1.2531) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 72.4280 | 72.3552 (-0.0728) | 72.0124 (-0.4156) | 72.3464 (-0.0817) |
| Gap-A | 406 | 390 | 75.2093 | 75.4691 (+0.2599) | 75.2144 (+0.0051) | 75.4666 (+0.2573) |
| Gap-T | 415 | 400 | 63.6390 | 64.1804 (+0.5413) | 62.5555 (-1.0835) | 63.6938 (+0.0547) |
| Gap-V | 419 | 399 | 76.8720 | 77.1072 (+0.2353) | 77.8134 (+0.9415) | 78.0875 (+1.2156) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 1.7946 | 1.8136 (+0.0190) | 1.8243 (+0.0296) | 1.8182 (+0.0235) |
| Gap-A | 406 | 390 | 1.6328 | 1.6414 (+0.0086) | 1.6434 (+0.0106) | 1.6348 (+0.0019) |
| Gap-T | 415 | 400 | 2.2916 | 2.3085 (+0.0168) | 2.3503 (+0.0587) | 2.3177 (+0.0260) |
| Gap-V | 419 | 399 | 1.5331 | 1.5282 (-0.0049) | 1.5267 (-0.0064) | 1.5513 (+0.0181) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 5.9085 | 7.5615 | 7.5341 |
| Gap-A | 5.8356 | 4.4458 | 4.0941 |
| Gap-T | 3.0832 | 7.2394 | 5.5730 |
| Gap-V | 3.0524 | 4.1360 | 9.1398 |

### random 0.7

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 66.9207 | 66.4634 (-0.4573) | 65.7012 (-1.2195) | 66.4634 (-0.4573) |
| Gap-A | 455 | 432 | 71.5278 | 71.2963 (-0.2315) | 70.8333 (-0.6944) | 71.7593 (+0.2315) |
| Gap-T | 465 | 447 | 59.2841 | 59.2841 (+0.0000) | 56.8233 (-2.4609) | 58.1655 (-1.1186) |
| Gap-V | 452 | 433 | 70.2079 | 70.2079 (+0.0000) | 70.9007 (+0.6928) | 70.9007 (+0.6928) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 67.0113 | 66.6455 (-0.3658) | 65.8184 (-1.1930) | 66.6338 (-0.3775) |
| Gap-A | 455 | 432 | 71.6944 | 71.4646 (-0.2298) | 71.0029 (-0.6915) | 71.9249 (+0.2305) |
| Gap-T | 465 | 447 | 59.3524 | 59.4433 (+0.0909) | 56.8365 (-2.5159) | 58.3237 (-1.0287) |
| Gap-V | 452 | 433 | 70.0474 | 70.0474 (+0.0000) | 70.6938 (+0.6464) | 70.5440 (+0.4966) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 2.0640 | 2.0900 (+0.0260) | 2.0742 (+0.0102) | 2.0908 (+0.0268) |
| Gap-A | 455 | 432 | 1.7840 | 1.7917 (+0.0077) | 1.7929 (+0.0090) | 1.7865 (+0.0025) |
| Gap-T | 465 | 447 | 2.5404 | 2.5592 (+0.0188) | 2.5475 (+0.0071) | 2.5599 (+0.0195) |
| Gap-V | 452 | 433 | 1.8558 | 1.8533 (-0.0025) | 1.8335 (-0.0223) | 1.8458 (-0.0100) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 5.6797 | 7.3881 | 7.4449 |
| Gap-A | 5.6348 | 4.6056 | 4.1763 |
| Gap-T | 3.3108 | 6.9314 | 5.3750 |
| Gap-V | 2.9061 | 4.1420 | 8.7807 |

### persistent A

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 58.9939 | 56.2500 (-2.7439) | 58.9939 (+0.0000) | 58.9939 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 686 | 656 | 58.9939 | 59.2988 (+0.3049) | 58.9939 (+0.0000) | 58.9939 (+0.0000) |
| Gap-V | 686 | 656 | 58.9939 | 59.1463 (+0.1524) | 58.9939 (+0.0000) | 58.9939 (+0.0000) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 57.7876 | 56.5155 (-1.2721) | 57.7876 (+0.0000) | 57.7876 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 686 | 656 | 57.7876 | 57.8349 (+0.0473) | 57.7876 (+0.0000) | 57.7876 (+0.0000) |
| Gap-V | 686 | 656 | 57.7876 | 57.9190 (+0.1313) | 57.7876 (+0.0000) | 57.7876 (+0.0000) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 2.5301 | 2.5973 (+0.0672) | 2.5301 (+0.0000) | 2.5301 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 686 | 656 | 2.5301 | 2.5165 (-0.0136) | 2.5301 (+0.0000) | 2.5301 (+0.0000) |
| Gap-V | 686 | 656 | 2.5301 | 2.5272 (-0.0029) | 2.5301 (+0.0000) | 2.5301 (+0.0000) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 7.6750 | 0.0000 | 0.0000 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | 0.4163 | 0.0000 | 0.0000 |
| Gap-V | 0.4457 | 0.0000 | 0.0000 |

### persistent T

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 85.9756 | 85.9756 (+0.0000) | 85.3659 (-0.6098) | 85.9756 (+0.0000) |
| Gap-A | 686 | 656 | 85.9756 | 85.9756 (+0.0000) | 86.1280 (+0.1524) | 85.9756 (+0.0000) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 686 | 656 | 85.9756 | 85.9756 (+0.0000) | 85.9756 (+0.0000) | 85.9756 (+0.0000) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 86.0006 | 86.0006 (+0.0000) | 85.4085 (-0.5921) | 86.0006 (+0.0000) |
| Gap-A | 686 | 656 | 86.0006 | 86.0006 (+0.0000) | 86.1499 (+0.1493) | 86.0006 (+0.0000) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 686 | 656 | 86.0006 | 86.0006 (+0.0000) | 85.9948 (-0.0058) | 86.0006 (+0.0000) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 1.0643 | 1.0643 (+0.0000) | 1.0973 (+0.0329) | 1.0643 (+0.0000) |
| Gap-A | 686 | 656 | 1.0643 | 1.0643 (+0.0000) | 1.0634 (-0.0009) | 1.0643 (+0.0000) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 686 | 656 | 1.0643 | 1.0643 (+0.0000) | 1.0466 (-0.0177) | 1.0643 (+0.0000) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 | 11.7721 | 0.0000 |
| Gap-A | 0.0000 | 5.9666 | 0.0000 |
| Gap-T | N/A | N/A | N/A |
| Gap-V | 0.0000 | 6.2534 | 0.0000 |

### persistent V

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 55.0305 | 55.0305 (+0.0000) | 55.0305 (+0.0000) | 53.2012 (-1.8293) |
| Gap-A | 686 | 656 | 55.0305 | 55.0305 (+0.0000) | 55.0305 (+0.0000) | 55.1829 (+0.1524) |
| Gap-T | 686 | 656 | 55.0305 | 55.0305 (+0.0000) | 55.0305 (+0.0000) | 54.7256 (-0.3049) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 53.8849 | 53.8849 (+0.0000) | 53.8849 (+0.0000) | 50.4013 (-3.4836) |
| Gap-A | 686 | 656 | 53.8849 | 53.8849 (+0.0000) | 53.8849 (+0.0000) | 54.0164 (+0.1315) |
| Gap-T | 686 | 656 | 53.8849 | 53.8849 (+0.0000) | 53.8849 (+0.0000) | 53.4704 (-0.4145) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 2.8740 | 2.8740 (+0.0000) | 2.8740 (+0.0000) | 3.0086 (+0.1346) |
| Gap-A | 686 | 656 | 2.8740 | 2.8740 (+0.0000) | 2.8740 (+0.0000) | 2.8785 (+0.0044) |
| Gap-T | 686 | 656 | 2.8740 | 2.8740 (+0.0000) | 2.8740 (+0.0000) | 2.9022 (+0.0282) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 | 0.0000 | 10.1478 |
| Gap-A | 0.0000 | 0.0000 | 1.9606 |
| Gap-T | 0.0000 | 0.0000 | 1.8718 |
| Gap-V | N/A | N/A | N/A |

### persistent AT

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 86.2805 | 86.1280 (-0.1524) | 86.2805 (+0.0000) | 86.2805 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 686 | 656 | 86.2805 | 86.1280 (-0.1524) | 85.9756 (-0.3049) | 86.2805 (+0.0000) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 86.1702 | 86.0319 (-0.1383) | 86.1804 (+0.0102) | 86.1702 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 686 | 656 | 86.1702 | 86.0113 (-0.1589) | 85.8411 (-0.3291) | 86.1702 (+0.0000) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 0.9489 | 0.9460 (-0.0029) | 0.9636 (+0.0147) | 0.9489 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 686 | 656 | 0.9489 | 0.9506 (+0.0017) | 0.9411 (-0.0077) | 0.9489 (+0.0000) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 7.1957 | 10.1740 | 0.0000 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | 1.1825 | 4.6485 | 0.0000 |

### persistent AV

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 60.0610 | 60.3659 (+0.3049) | 60.0610 (+0.0000) | 60.6707 (+0.6098) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 686 | 656 | 60.0610 | 60.0610 (+0.0000) | 60.0610 (+0.0000) | 60.2134 (+0.1524) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 59.5079 | 60.2115 (+0.7035) | 59.5079 (+0.0000) | 60.5175 (+1.0096) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 686 | 656 | 59.5079 | 59.4713 (-0.0367) | 59.5079 (+0.0000) | 59.7833 (+0.2754) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 2.4802 | 2.5127 (+0.0325) | 2.4802 (+0.0000) | 2.5155 (+0.0353) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 686 | 656 | 2.4802 | 2.4789 (-0.0013) | 2.4802 (+0.0000) | 2.4937 (+0.0135) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 8.6119 | 0.0000 | 10.7293 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | 0.2481 | 0.0000 | 1.5726 |
| Gap-V | N/A | N/A | N/A |

### persistent TV

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 85.2134 | 85.2134 (+0.0000) | 85.6707 (+0.4573) | 85.9756 (+0.7622) |
| Gap-A | 686 | 656 | 85.2134 | 85.2134 (+0.0000) | 85.2134 (+0.0000) | 85.2134 (+0.0000) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 85.2239 | 85.2239 (+0.0000) | 85.7074 (+0.4835) | 86.0006 (+0.7768) |
| Gap-A | 686 | 656 | 85.2239 | 85.2239 (+0.0000) | 85.2239 (+0.0000) | 85.2239 (+0.0000) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 1.0150 | 1.0150 (+0.0000) | 1.0493 (+0.0343) | 1.0336 (+0.0186) |
| Gap-A | 686 | 656 | 1.0150 | 1.0150 (+0.0000) | 1.0137 (-0.0013) | 1.0143 (-0.0007) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 | 10.8166 | 7.9610 |
| Gap-A | 0.0000 | 4.9226 | 2.2694 |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

### persistent ATV

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 86.2805 | 86.2805 (+0.0000) | 86.2805 (+0.0000) | 85.8232 (-0.4573) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 86.2270 | 86.2516 (+0.0246) | 86.2437 (+0.0167) | 85.7893 (-0.4378) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 0.9830 | 0.9822 (-0.0008) | 1.0038 (+0.0208) | 0.9870 (+0.0040) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 7.7243 | 8.4349 | 8.1655 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

## Integrity and limits

Source-sum reconstruction maxima:

| Check | Maximum |
|---|---:|
| base_sum_max_abs_error | 1.43051147e-06 |
| base_sum_max_relative_error | 1.59662378e-07 |
| gap_sum_max_abs_error | 1.43051147e-06 |
| gap_sum_max_relative_error | 5.98686142e-07 |
| memory_sum_max_abs_error | 1.43051147e-06 |
| memory_sum_max_relative_error | 2.86684156e-07 |
| original_readout_reproduction_max_abs_error | 0 |
| prewrite_memory_sum_max_abs_error | 1.43051147e-06 |
| prewrite_memory_sum_max_relative_error | 2.88746008e-07 |
| source_sum_max_abs_error | 1.43051147e-06 |

SLOTS.csv includes paired corrections/harms, active N, polarity N, norms and all metric deltas. SLOTS_GROUPED.csv additionally stratifies current availability, strictly-past ever-seen history and source coverage. SHAPLEY.csv uses exact three-source grouped Shapley with payoff=-per-utterance MSE (not ACC/W-F1). Independent slot effects are nonlinear and not additive; source-group Shapley credits satisfy efficiency for the fixed -MSE game. Source subtraction may be out-of-distribution. Norm/cancellation are not sentiment conflict or reliability. This single-seed diagnostic does not establish significant gains or a new method.
