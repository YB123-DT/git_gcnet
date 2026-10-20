# OSRAM scalar 12-Source Evidence Audit

INTERNAL DIAGNOSTIC ONLY

Dataset CMUMOSEI; seed 66; fold 1; source `1c5f29cb6c1592cdf0afe8874afc34a51ccd7938`.

Fixed original miss=0.3 checkpoint: `/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat/best_miss_0p3.pt`.

Checkpoint SHA256: `7905d518b4006d78f5f0d1139fc025ff190f9e1725da831d091292fd5575e0ab`.

No training; original Flat, Local Skip, Memory, Query and task head are unchanged. One fixed checkpoint is used for every rate/pattern, not the eight per-rate selected baseline models. The anchor was previously selected by Test-oracle; these are internal frozen interventions, not independent generalization results.

Rows are read destinations Base/Gap-A/Gap-T/Gap-V; columns are historical A/T/V Value-write sources. Each cell reports the final score after removing ONLY that one source contribution at the final Flat input. Local and all other evidence remain unchanged. Sources are algebraic joint-write Value-slot contributions, not pure modality semantics.

ACC/W-F1 use abs(gold)>0 and prediction>0; prediction=0 is nonpositive. MSE includes all valid labels. Each Gap row uses only utterances where that target modality is currently missing; its original score is recomputed on the identical cohort. Inactive/empty rows are N/A. Do not compare rows with different cohorts as if they were full-test scores.

Positive ACC/W-F1 Δ means removing the item improves that score; positive MSE Δ means removing it increases loss. Corrections/harms compare removed vs full on nonzero-label utterances only. No subgroup W-F1 values are averaged to reconstruct full-test W-F1.

## Full-model metrics by condition

| Condition | N | Polarity N | ACC | W-F1 | MSE |
|---|---:|---:|---:|---:|---:|
| random 0.0 | 4659 | 3634 | 87.0391 | 86.9468 | 0.4607 |
| random 0.1 | 4659 | 3634 | 85.9384 | 85.8408 | 0.4984 |
| random 0.2 | 4659 | 3634 | 85.4155 | 85.2688 | 0.5313 |
| random 0.3 | 4659 | 3634 | 84.7001 | 84.5401 | 0.5594 |
| random 0.4 | 4659 | 3634 | 82.5812 | 82.3938 | 0.6393 |
| random 0.5 | 4659 | 3634 | 81.7556 | 81.4720 | 0.6752 |
| random 0.6 | 4659 | 3634 | 79.2515 | 78.9221 | 0.7263 |
| random 0.7 | 4659 | 3634 | 78.1233 | 77.6210 | 0.7829 |
| persistent A | 4659 | 3634 | 66.0704 | 58.8533 | 1.1700 |
| persistent T | 4659 | 3634 | 86.7914 | 86.5730 | 0.4861 |
| persistent V | 4659 | 3634 | 63.8690 | 55.3595 | 1.1683 |
| persistent AT | 4659 | 3634 | 87.1216 | 86.9895 | 0.4729 |
| persistent AV | 4659 | 3634 | 67.5839 | 64.0978 | 1.1151 |
| persistent TV | 4659 | 3634 | 87.5069 | 87.4168 | 0.4638 |
| persistent ATV | 4659 | 3634 | 87.0391 | 86.9468 | 0.4607 |
| random8 | — | — | 83.1006 | 82.8757 | 0.6092 |
| high_missing | — | — | 79.7101 | 79.3384 | 0.7281 |
| persistent6 | — | — | 76.4906 | 73.2150 | 0.8127 |

Aggregates are equal-weight full-condition means; Persistent ATV is separate from persistent6.

## All 15 conditions: 4×3 source removal tables

### random 0.0

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 87.0391 | 87.1767 (+0.1376) | 86.3236 (-0.7155) | 87.0941 (+0.0550) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 86.9468 | 87.0674 (+0.1206) | 86.1402 (-0.8066) | 86.9829 (+0.0361) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.4607 | 0.4616 (+0.0009) | 0.4979 (+0.0371) | 0.4616 (+0.0009) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 5.9569 | 15.7138 | 6.1961 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

### random 0.1

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 85.9384 | 85.9108 (-0.0275) | 85.1954 (-0.7430) | 85.8833 (-0.0550) |
| Gap-A | 486 | 371 | 88.9488 | 88.6792 (-0.2695) | 88.6792 (-0.2695) | 88.9488 (+0.0000) |
| Gap-T | 463 | 369 | 72.8997 | 72.6287 (-0.2710) | 75.6098 (+2.7100) | 72.3577 (-0.5420) |
| Gap-V | 466 | 372 | 87.3656 | 87.3656 (+0.0000) | 86.5591 (-0.8065) | 86.2903 (-1.0753) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 85.8408 | 85.7935 (-0.0474) | 85.0046 (-0.8363) | 85.7724 (-0.0685) |
| Gap-A | 486 | 371 | 88.8230 | 88.5373 (-0.2857) | 88.5373 (-0.2857) | 88.8230 (+0.0000) |
| Gap-T | 463 | 369 | 72.7164 | 72.4224 (-0.2940) | 74.6106 (+1.8942) | 72.1275 (-0.5889) |
| Gap-V | 466 | 372 | 87.1845 | 87.1845 (+0.0000) | 86.3529 (-0.8316) | 85.9754 (-1.2091) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.4984 | 0.4992 (+0.0009) | 0.5333 (+0.0349) | 0.4993 (+0.0009) |
| Gap-A | 486 | 371 | 0.4499 | 0.4502 (+0.0003) | 0.4491 (-0.0009) | 0.4503 (+0.0003) |
| Gap-T | 463 | 369 | 0.8305 | 0.8308 (+0.0003) | 0.8864 (+0.0560) | 0.8306 (+0.0001) |
| Gap-V | 466 | 372 | 0.4836 | 0.4824 (-0.0012) | 0.4884 (+0.0048) | 0.4944 (+0.0108) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 5.8141 | 15.1962 | 6.0915 |
| Gap-A | 5.7185 | 8.6973 | 2.6843 |
| Gap-T | 1.5681 | 18.1367 | 1.9792 |
| Gap-V | 2.6940 | 8.3358 | 9.4705 |

### random 0.2

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 85.4155 | 85.4981 (+0.0826) | 84.8927 (-0.5228) | 85.4155 (+0.0000) |
| Gap-A | 917 | 713 | 86.5358 | 86.9565 (+0.4208) | 86.8163 (+0.2805) | 86.2553 (-0.2805) |
| Gap-T | 934 | 705 | 78.7234 | 78.7234 (+0.0000) | 77.7305 (-0.9929) | 78.7234 (+0.0000) |
| Gap-V | 906 | 701 | 85.0214 | 84.7361 (-0.2853) | 85.0214 (+0.0000) | 84.5934 (-0.4280) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 85.2688 | 85.3186 (+0.0499) | 84.6644 (-0.6044) | 85.2484 (-0.0203) |
| Gap-A | 917 | 713 | 86.2933 | 86.6969 (+0.4036) | 86.5952 (+0.3019) | 86.0247 (-0.2686) |
| Gap-T | 934 | 705 | 78.4826 | 78.4826 (+0.0000) | 76.6605 (-1.8221) | 78.4361 (-0.0465) |
| Gap-V | 906 | 701 | 84.9361 | 84.6634 (-0.2727) | 84.9219 (-0.0143) | 84.4208 (-0.5153) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.5313 | 0.5325 (+0.0011) | 0.5595 (+0.0282) | 0.5322 (+0.0009) |
| Gap-A | 917 | 713 | 0.5086 | 0.5114 (+0.0028) | 0.5117 (+0.0030) | 0.5090 (+0.0004) |
| Gap-T | 934 | 705 | 0.7824 | 0.7826 (+0.0002) | 0.8616 (+0.0791) | 0.7853 (+0.0029) |
| Gap-V | 906 | 701 | 0.5693 | 0.5690 (-0.0004) | 0.5722 (+0.0028) | 0.5780 (+0.0087) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 5.7087 | 14.5005 | 5.9489 |
| Gap-A | 5.8689 | 8.6444 | 2.8982 |
| Gap-T | 1.7839 | 17.0217 | 2.2943 |
| Gap-V | 2.8186 | 8.8626 | 8.9996 |

### random 0.3

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 84.7001 | 84.6450 (-0.0550) | 83.9296 (-0.7705) | 84.7276 (+0.0275) |
| Gap-A | 1448 | 1143 | 85.3893 | 84.8644 (-0.5249) | 85.0394 (-0.3500) | 85.2143 (-0.1750) |
| Gap-T | 1371 | 1061 | 77.7568 | 77.7568 (+0.0000) | 76.8143 (-0.9425) | 77.8511 (+0.0943) |
| Gap-V | 1391 | 1088 | 86.5809 | 86.7647 (+0.1838) | 86.5809 (+0.0000) | 86.1213 (-0.4596) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 84.5401 | 84.4471 (-0.0929) | 83.6744 (-0.8656) | 84.5386 (-0.0014) |
| Gap-A | 1448 | 1143 | 85.2141 | 84.6374 (-0.5767) | 84.8600 (-0.3542) | 85.0456 (-0.1685) |
| Gap-T | 1371 | 1061 | 77.4838 | 77.4680 (-0.0158) | 75.5450 (-1.9388) | 77.5714 (+0.0876) |
| Gap-V | 1391 | 1088 | 86.3719 | 86.5778 (+0.2059) | 86.3519 (-0.0200) | 85.8360 (-0.5359) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.5594 | 0.5606 (+0.0012) | 0.5869 (+0.0275) | 0.5607 (+0.0013) |
| Gap-A | 1448 | 1143 | 0.5450 | 0.5479 (+0.0029) | 0.5484 (+0.0035) | 0.5456 (+0.0007) |
| Gap-T | 1371 | 1061 | 0.7931 | 0.7928 (-0.0003) | 0.8692 (+0.0761) | 0.7946 (+0.0014) |
| Gap-V | 1391 | 1088 | 0.4920 | 0.4923 (+0.0003) | 0.4930 (+0.0010) | 0.5001 (+0.0080) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 5.4262 | 13.8131 | 5.6894 |
| Gap-A | 5.3673 | 8.4858 | 3.0653 |
| Gap-T | 2.0453 | 16.0130 | 2.5322 |
| Gap-V | 2.6351 | 8.2126 | 8.1529 |

### random 0.4

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 82.5812 | 82.5537 (-0.0275) | 81.7281 (-0.8531) | 82.3886 (-0.1926) |
| Gap-A | 1865 | 1437 | 84.1336 | 84.2728 (+0.1392) | 83.7857 (-0.3479) | 83.9944 (-0.1392) |
| Gap-T | 1866 | 1454 | 76.7538 | 76.5475 (-0.2063) | 74.8968 (-1.8569) | 76.8913 (+0.1376) |
| Gap-V | 1843 | 1464 | 84.0847 | 83.9481 (-0.1366) | 83.8798 (-0.2049) | 83.9481 (-0.1366) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 82.3938 | 82.3215 (-0.0723) | 81.4812 (-0.9127) | 82.1652 (-0.2286) |
| Gap-A | 1865 | 1437 | 84.0553 | 84.1811 (+0.1259) | 83.6948 (-0.3605) | 83.9224 (-0.1329) |
| Gap-T | 1866 | 1454 | 76.4398 | 76.2619 (-0.1779) | 73.7967 (-2.6431) | 76.5667 (+0.1269) |
| Gap-V | 1843 | 1464 | 83.9287 | 83.8140 (-0.1147) | 83.6855 (-0.2433) | 83.7338 (-0.1950) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.6393 | 0.6395 (+0.0002) | 0.6629 (+0.0237) | 0.6397 (+0.0004) |
| Gap-A | 1865 | 1437 | 0.5825 | 0.5833 (+0.0008) | 0.5844 (+0.0020) | 0.5832 (+0.0007) |
| Gap-T | 1866 | 1454 | 0.8635 | 0.8634 (-0.0000) | 0.9402 (+0.0767) | 0.8646 (+0.0012) |
| Gap-V | 1843 | 1464 | 0.6127 | 0.6130 (+0.0003) | 0.6149 (+0.0023) | 0.6180 (+0.0053) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 5.1968 | 12.8800 | 5.4784 |
| Gap-A | 5.1369 | 8.6197 | 3.1337 |
| Gap-T | 2.3986 | 15.1560 | 2.7900 |
| Gap-V | 2.6323 | 8.4843 | 7.7862 |

### random 0.5

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 81.7556 | 81.8107 (+0.0550) | 81.2603 (-0.4953) | 81.8382 (+0.0826) |
| Gap-A | 2330 | 1792 | 84.1518 | 84.4866 (+0.3348) | 84.0960 (-0.0558) | 83.9286 (-0.2232) |
| Gap-T | 2370 | 1851 | 76.4992 | 76.7693 (+0.2701) | 75.7428 (-0.7563) | 76.4992 (+0.0000) |
| Gap-V | 2332 | 1841 | 82.0750 | 82.0206 (-0.0543) | 81.8034 (-0.2716) | 81.5861 (-0.4889) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 81.4720 | 81.4625 (-0.0095) | 80.8929 (-0.5791) | 81.5134 (+0.0414) |
| Gap-A | 2330 | 1792 | 83.9733 | 84.2724 (+0.2991) | 83.9135 (-0.0597) | 83.7475 (-0.2257) |
| Gap-T | 2370 | 1851 | 76.0513 | 76.3100 (+0.2588) | 74.4515 (-1.5998) | 76.0290 (-0.0223) |
| Gap-V | 2332 | 1841 | 81.7708 | 81.7193 (-0.0515) | 81.4515 (-0.3193) | 81.1719 (-0.5988) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.6752 | 0.6770 (+0.0017) | 0.6922 (+0.0170) | 0.6758 (+0.0006) |
| Gap-A | 2330 | 1792 | 0.6176 | 0.6191 (+0.0015) | 0.6208 (+0.0032) | 0.6187 (+0.0011) |
| Gap-T | 2370 | 1851 | 0.8607 | 0.8603 (-0.0004) | 0.9166 (+0.0559) | 0.8627 (+0.0020) |
| Gap-V | 2332 | 1841 | 0.6300 | 0.6297 (-0.0004) | 0.6329 (+0.0028) | 0.6380 (+0.0080) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 4.9394 | 11.7451 | 5.1544 |
| Gap-A | 4.7339 | 8.4328 | 3.3804 |
| Gap-T | 2.4163 | 13.7540 | 2.7623 |
| Gap-V | 2.5681 | 7.9766 | 7.0962 |

### random 0.6

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 79.2515 | 79.3616 (+0.1101) | 78.9763 (-0.2752) | 79.3616 (+0.1101) |
| Gap-A | 2828 | 2222 | 80.6031 | 80.4680 (-0.1350) | 80.4680 (-0.1350) | 80.4680 (-0.1350) |
| Gap-T | 2779 | 2126 | 74.0357 | 73.8946 (-0.1411) | 73.5654 (-0.4704) | 73.9417 (-0.0941) |
| Gap-V | 2777 | 2177 | 82.3610 | 82.2232 (-0.1378) | 82.0854 (-0.2756) | 82.2232 (-0.1378) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 78.9221 | 78.9737 (+0.0515) | 78.5567 (-0.3654) | 78.9831 (+0.0610) |
| Gap-A | 2828 | 2222 | 80.3532 | 80.1520 (-0.2011) | 80.2131 (-0.1400) | 80.2131 (-0.1400) |
| Gap-T | 2779 | 2126 | 73.4362 | 73.2404 (-0.1958) | 72.0385 (-1.3977) | 73.3173 (-0.1189) |
| Gap-V | 2777 | 2177 | 82.1392 | 82.0027 (-0.1365) | 81.8222 (-0.3170) | 81.9328 (-0.2064) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.7263 | 0.7276 (+0.0013) | 0.7413 (+0.0150) | 0.7266 (+0.0003) |
| Gap-A | 2828 | 2222 | 0.6701 | 0.6732 (+0.0031) | 0.6725 (+0.0023) | 0.6709 (+0.0008) |
| Gap-T | 2779 | 2126 | 0.8756 | 0.8770 (+0.0014) | 0.9232 (+0.0476) | 0.8767 (+0.0011) |
| Gap-V | 2777 | 2177 | 0.6704 | 0.6707 (+0.0004) | 0.6718 (+0.0014) | 0.6749 (+0.0045) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 4.4583 | 11.1187 | 4.8020 |
| Gap-A | 4.1824 | 8.2406 | 3.2015 |
| Gap-T | 2.3434 | 12.9379 | 2.6015 |
| Gap-V | 2.4113 | 7.8817 | 6.2216 |

### random 0.7

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 78.1233 | 77.6555 (-0.4678) | 77.5729 (-0.5504) | 77.9857 (-0.1376) |
| Gap-A | 3083 | 2408 | 79.4850 | 79.0282 (-0.4568) | 79.2774 (-0.2076) | 79.4850 (+0.0000) |
| Gap-T | 3155 | 2447 | 74.1725 | 73.8455 (-0.3269) | 72.3743 (-1.7981) | 74.1316 (-0.0409) |
| Gap-V | 3080 | 2413 | 80.7708 | 80.6465 (-0.1243) | 80.5636 (-0.2072) | 80.2321 (-0.5387) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 77.6210 | 77.0081 (-0.6130) | 76.9557 (-0.6653) | 77.4169 (-0.2042) |
| Gap-A | 3083 | 2408 | 79.1627 | 78.6181 (-0.5446) | 78.9281 (-0.2346) | 79.1560 (-0.0067) |
| Gap-T | 3155 | 2447 | 73.4781 | 73.0917 (-0.3863) | 70.6737 (-2.8044) | 73.4012 (-0.0769) |
| Gap-V | 3080 | 2413 | 80.2511 | 80.1196 (-0.1316) | 79.9624 (-0.2887) | 79.5612 (-0.6899) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.7829 | 0.7872 (+0.0043) | 0.7959 (+0.0130) | 0.7856 (+0.0027) |
| Gap-A | 3083 | 2408 | 0.7195 | 0.7236 (+0.0040) | 0.7226 (+0.0030) | 0.7203 (+0.0008) |
| Gap-T | 3155 | 2447 | 0.9146 | 0.9162 (+0.0016) | 0.9688 (+0.0542) | 0.9171 (+0.0025) |
| Gap-V | 3080 | 2413 | 0.7114 | 0.7114 (+0.0000) | 0.7135 (+0.0021) | 0.7207 (+0.0093) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 4.3101 | 9.1031 | 4.4731 |
| Gap-A | 4.1218 | 7.1506 | 2.9238 |
| Gap-T | 2.2675 | 10.5132 | 2.4941 |
| Gap-V | 2.3251 | 7.0263 | 5.5707 |

### persistent A

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 66.0704 | 62.8509 (-3.2196) | 66.0704 (+0.0000) | 66.0704 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 4659 | 3634 | 66.0704 | 66.0980 (+0.0275) | 66.0704 (+0.0000) | 66.0704 (+0.0000) |
| Gap-V | 4659 | 3634 | 66.0704 | 66.0429 (-0.0275) | 66.0704 (+0.0000) | 66.0704 (+0.0000) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 58.8533 | 48.5135 (-10.3398) | 58.8533 (+0.0000) | 58.8533 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 4659 | 3634 | 58.8533 | 58.9004 (+0.0471) | 58.8533 (+0.0000) | 58.8533 (+0.0000) |
| Gap-V | 4659 | 3634 | 58.8533 | 58.8061 (-0.0472) | 58.8533 (+0.0000) | 58.8533 (+0.0000) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 1.1700 | 1.2031 (+0.0331) | 1.1700 (+0.0000) | 1.1700 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 4659 | 3634 | 1.1700 | 1.1708 (+0.0008) | 1.1700 (+0.0000) | 1.1700 (+0.0000) |
| Gap-V | 4659 | 3634 | 1.1700 | 1.1716 (+0.0016) | 1.1700 (+0.0000) | 1.1700 (+0.0000) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 6.8235 | 0.0000 | 0.0000 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | 0.4229 | 0.0000 | 0.0000 |
| Gap-V | 0.4240 | 0.0000 | 0.0000 |

### persistent T

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 86.7914 | 86.7914 (+0.0000) | 85.7182 (-1.0732) | 86.7914 (+0.0000) |
| Gap-A | 4659 | 3634 | 86.7914 | 86.7914 (+0.0000) | 86.7639 (-0.0275) | 86.7914 (+0.0000) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 4659 | 3634 | 86.7914 | 86.7914 (+0.0000) | 86.6538 (-0.1376) | 86.7914 (+0.0000) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 86.5730 | 86.5730 (+0.0000) | 85.3762 (-1.1968) | 86.5730 (+0.0000) |
| Gap-A | 4659 | 3634 | 86.5730 | 86.5730 (+0.0000) | 86.5347 (-0.0383) | 86.5730 (+0.0000) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 4659 | 3634 | 86.5730 | 86.5730 (+0.0000) | 86.4197 (-0.1533) | 86.5730 (+0.0000) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.4861 | 0.4861 (+0.0000) | 0.5184 (+0.0322) | 0.4861 (+0.0000) |
| Gap-A | 4659 | 3634 | 0.4861 | 0.4861 (+0.0000) | 0.4886 (+0.0025) | 0.4861 (+0.0000) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 4659 | 3634 | 0.4861 | 0.4861 (+0.0000) | 0.4894 (+0.0032) | 0.4861 (+0.0000) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 | 15.2968 | 0.0000 |
| Gap-A | 0.0000 | 8.6575 | 0.0000 |
| Gap-T | N/A | N/A | N/A |
| Gap-V | 0.0000 | 8.6126 | 0.0000 |

### persistent V

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 63.8690 | 63.8690 (+0.0000) | 63.8690 (+0.0000) | 63.1811 (-0.6879) |
| Gap-A | 4659 | 3634 | 63.8690 | 63.8690 (+0.0000) | 63.8690 (+0.0000) | 63.8965 (+0.0275) |
| Gap-T | 4659 | 3634 | 63.8690 | 63.8690 (+0.0000) | 63.8690 (+0.0000) | 63.9241 (+0.0550) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 55.3595 | 55.3595 (+0.0000) | 55.3595 (+0.0000) | 51.1386 (-4.2209) |
| Gap-A | 4659 | 3634 | 55.3595 | 55.3595 (+0.0000) | 55.3595 (+0.0000) | 55.5040 (+0.1445) |
| Gap-T | 4659 | 3634 | 55.3595 | 55.3595 (+0.0000) | 55.3595 (+0.0000) | 55.7399 (+0.3804) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 1.1683 | 1.1683 (+0.0000) | 1.1683 (+0.0000) | 1.1835 (+0.0153) |
| Gap-A | 4659 | 3634 | 1.1683 | 1.1683 (+0.0000) | 1.1683 (+0.0000) | 1.1686 (+0.0003) |
| Gap-T | 4659 | 3634 | 1.1683 | 1.1683 (+0.0000) | 1.1683 (+0.0000) | 1.1686 (+0.0003) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 | 0.0000 | 7.3688 |
| Gap-A | 0.0000 | 0.0000 | 0.8079 |
| Gap-T | 0.0000 | 0.0000 | 0.7946 |
| Gap-V | N/A | N/A | N/A |

### persistent AT

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 87.1216 | 87.2592 (+0.1376) | 86.0759 (-1.0457) | 87.1216 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 4659 | 3634 | 87.1216 | 87.1216 (+0.0000) | 87.1767 (+0.0550) | 87.1216 (+0.0000) |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 86.9895 | 87.1042 (+0.1147) | 85.8457 (-1.1438) | 86.9895 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 4659 | 3634 | 86.9895 | 86.9844 (-0.0051) | 87.0298 (+0.0403) | 86.9895 (+0.0000) |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.4729 | 0.4750 (+0.0021) | 0.5060 (+0.0332) | 0.4729 (+0.0000) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 4659 | 3634 | 0.4729 | 0.4730 (+0.0001) | 0.4755 (+0.0026) | 0.4729 (+0.0000) |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 6.5529 | 15.4469 | 0.0000 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | 0.5829 | 8.6428 | 0.0000 |

### persistent AV

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 67.5839 | 66.0980 (-1.4860) | 67.5839 (+0.0000) | 66.9785 (-0.6054) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 4659 | 3634 | 67.5839 | 67.4188 (-0.1651) | 67.5839 (+0.0000) | 67.3363 (-0.2477) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 64.0978 | 59.5371 (-4.5607) | 64.0978 (+0.0000) | 62.0037 (-2.0941) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 4659 | 3634 | 64.0978 | 63.8651 (-0.2327) | 64.0978 (+0.0000) | 63.9302 (-0.1676) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 1.1151 | 1.1404 (+0.0254) | 1.1151 (+0.0000) | 1.1240 (+0.0089) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 4659 | 3634 | 1.1151 | 1.1158 (+0.0008) | 1.1151 (+0.0000) | 1.1150 (-0.0001) |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 6.3303 | 0.0000 | 5.6708 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | 0.3923 | 0.0000 | 0.7108 |
| Gap-V | N/A | N/A | N/A |

### persistent TV

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 87.5069 | 87.5069 (+0.0000) | 86.3786 (-1.1282) | 87.5344 (+0.0275) |
| Gap-A | 4659 | 3634 | 87.5069 | 87.5069 (+0.0000) | 87.3968 (-0.1101) | 87.4794 (-0.0275) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 87.4168 | 87.4168 (+0.0000) | 86.1786 (-1.2382) | 87.4317 (+0.0149) |
| Gap-A | 4659 | 3634 | 87.4168 | 87.4168 (+0.0000) | 87.2942 (-0.1226) | 87.3879 (-0.0289) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.4638 | 0.4638 (+0.0000) | 0.4965 (+0.0326) | 0.4649 (+0.0010) |
| Gap-A | 4659 | 3634 | 0.4638 | 0.4638 (+0.0000) | 0.4658 (+0.0019) | 0.4638 (-0.0001) |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 0.0000 | 15.5143 | 7.7010 |
| Gap-A | 0.0000 | 8.0725 | 1.2397 |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

### persistent ATV

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 87.0391 | 87.1767 (+0.1376) | 86.3236 (-0.7155) | 87.0941 (+0.0550) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 86.9468 | 87.0674 (+0.1206) | 86.1402 (-0.8066) | 86.9829 (+0.0361) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

MSE: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 0.4607 | 0.4616 (+0.0009) | 0.4979 (+0.0371) | 0.4616 (+0.0009) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

Mean source-read L2 norm; active cohorts only.

| Read destination | History A | History T | History V |
|---|---:|---:|---:|
| Base | 5.9569 | 15.7138 | 6.1961 |
| Gap-A | N/A | N/A | N/A |
| Gap-T | N/A | N/A | N/A |
| Gap-V | N/A | N/A | N/A |

## Integrity and limits

Source-sum reconstruction maxima:

| Check | Maximum |
|---|---:|
| base_sum_max_abs_error | 3.33786011e-06 |
| base_sum_max_relative_error | 3.40138115e-07 |
| gap_sum_max_abs_error | 3.81469727e-06 |
| gap_sum_max_relative_error | 6.4765004e-06 |
| memory_sum_max_abs_error | 4.29153442e-06 |
| memory_sum_max_relative_error | 3.57627073e-07 |
| original_readout_reproduction_max_abs_error | 0 |
| prewrite_memory_sum_max_abs_error | 4.76837158e-06 |
| prewrite_memory_sum_max_relative_error | 3.59168098e-07 |
| source_sum_max_abs_error | 3.81469727e-06 |

SLOTS.csv includes paired corrections/harms, active N, polarity N, norms and all metric deltas. SLOTS_GROUPED.csv additionally stratifies current availability, strictly-past ever-seen history and source coverage. SHAPLEY.csv uses exact three-source grouped Shapley with payoff=-per-utterance MSE (not ACC/W-F1). Independent slot effects are nonlinear and not additive; source-group Shapley credits satisfy efficiency for the fixed -MSE game. Source subtraction may be out-of-distribution. Norm/cancellation are not sentiment conflict or reliability. This single-seed diagnostic does not establish significant gains or a new method.
