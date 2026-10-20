# Random eight-rate WF1 4×3 source-removal tables

INTERNAL DIAGNOSTIC ONLY

Dataset CMUMOSEI; seed 66; fold 1; source `1c5f29cb6c1592cdf0afe8874afc34a51ccd7938`.

Fixed original miss=0.3 checkpoint: `/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat/best_miss_0p3.pt`.

Checkpoint SHA256: `7905d518b4006d78f5f0d1139fc025ff190f9e1725da831d091292fd5575e0ab`.

No training; original Flat, Local Skip, Memory, Query and task head are unchanged. One fixed checkpoint is used for every rate/pattern, not the eight per-rate selected baseline models. The anchor was previously selected by Test-oracle; these are internal frozen interventions, not independent generalization results.

Rows are read destinations Base/Gap-A/Gap-T/Gap-V; columns are historical A/T/V Value-write sources. Each cell reports the final score after removing ONLY that one source contribution at the final Flat input. Local and all other evidence remain unchanged. Sources are algebraic joint-write Value-slot contributions, not pure modality semantics.

ACC/W-F1 use abs(gold)>0 and prediction>0; prediction=0 is nonpositive. MSE includes all valid labels. Each Gap row uses only utterances where that target modality is currently missing; its original score is recomputed on the identical cohort. Inactive/empty rows are N/A. Do not compare rows with different cohorts as if they were full-test scores.

Positive ACC/W-F1 Δ means removing the item improves that score; positive MSE Δ means removing it increases loss. Corrections/harms compare removed vs full on nonzero-label utterances only. No subgroup W-F1 values are averaged to reconstruct full-test W-F1.

## random 0.0

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 86.9468 | 87.0674 (+0.1206) | 86.1402 (-0.8066) | 86.9829 (+0.0361) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

## random 0.1

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 85.8408 | 85.7935 (-0.0474) | 85.0046 (-0.8363) | 85.7724 (-0.0685) |
| Gap-A | 486 | 371 | 88.8230 | 88.5373 (-0.2857) | 88.5373 (-0.2857) | 88.8230 (+0.0000) |
| Gap-T | 463 | 369 | 72.7164 | 72.4224 (-0.2940) | 74.6106 (+1.8942) | 72.1275 (-0.5889) |
| Gap-V | 466 | 372 | 87.1845 | 87.1845 (+0.0000) | 86.3529 (-0.8316) | 85.9754 (-1.2091) |

## random 0.2

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 85.2688 | 85.3186 (+0.0499) | 84.6644 (-0.6044) | 85.2484 (-0.0203) |
| Gap-A | 917 | 713 | 86.2933 | 86.6969 (+0.4036) | 86.5952 (+0.3019) | 86.0247 (-0.2686) |
| Gap-T | 934 | 705 | 78.4826 | 78.4826 (+0.0000) | 76.6605 (-1.8221) | 78.4361 (-0.0465) |
| Gap-V | 906 | 701 | 84.9361 | 84.6634 (-0.2727) | 84.9219 (-0.0143) | 84.4208 (-0.5153) |

## random 0.3

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 84.5401 | 84.4471 (-0.0929) | 83.6744 (-0.8656) | 84.5386 (-0.0014) |
| Gap-A | 1448 | 1143 | 85.2141 | 84.6374 (-0.5767) | 84.8600 (-0.3542) | 85.0456 (-0.1685) |
| Gap-T | 1371 | 1061 | 77.4838 | 77.4680 (-0.0158) | 75.5450 (-1.9388) | 77.5714 (+0.0876) |
| Gap-V | 1391 | 1088 | 86.3719 | 86.5778 (+0.2059) | 86.3519 (-0.0200) | 85.8360 (-0.5359) |

## random 0.4

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 82.3938 | 82.3215 (-0.0723) | 81.4812 (-0.9127) | 82.1652 (-0.2286) |
| Gap-A | 1865 | 1437 | 84.0553 | 84.1811 (+0.1259) | 83.6948 (-0.3605) | 83.9224 (-0.1329) |
| Gap-T | 1866 | 1454 | 76.4398 | 76.2619 (-0.1779) | 73.7967 (-2.6431) | 76.5667 (+0.1269) |
| Gap-V | 1843 | 1464 | 83.9287 | 83.8140 (-0.1147) | 83.6855 (-0.2433) | 83.7338 (-0.1950) |

## random 0.5

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 81.4720 | 81.4625 (-0.0095) | 80.8929 (-0.5791) | 81.5134 (+0.0414) |
| Gap-A | 2330 | 1792 | 83.9733 | 84.2724 (+0.2991) | 83.9135 (-0.0597) | 83.7475 (-0.2257) |
| Gap-T | 2370 | 1851 | 76.0513 | 76.3100 (+0.2588) | 74.4515 (-1.5998) | 76.0290 (-0.0223) |
| Gap-V | 2332 | 1841 | 81.7708 | 81.7193 (-0.0515) | 81.4515 (-0.3193) | 81.1719 (-0.5988) |

## random 0.6

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 78.9221 | 78.9737 (+0.0515) | 78.5567 (-0.3654) | 78.9831 (+0.0610) |
| Gap-A | 2828 | 2222 | 80.3532 | 80.1520 (-0.2011) | 80.2131 (-0.1400) | 80.2131 (-0.1400) |
| Gap-T | 2779 | 2126 | 73.4362 | 73.2404 (-0.1958) | 72.0385 (-1.3977) | 73.3173 (-0.1189) |
| Gap-V | 2777 | 2177 | 82.1392 | 82.0027 (-0.1365) | 81.8222 (-0.3170) | 81.9328 (-0.2064) |

## random 0.7

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 77.6210 | 77.0081 (-0.6130) | 76.9557 (-0.6653) | 77.4169 (-0.2042) |
| Gap-A | 3083 | 2408 | 79.1627 | 78.6181 (-0.5446) | 78.9281 (-0.2346) | 79.1560 (-0.0067) |
| Gap-T | 3155 | 2447 | 73.4781 | 73.0917 (-0.3863) | 70.6737 (-2.8044) | 73.4012 (-0.0769) |
| Gap-V | 3080 | 2413 | 80.2511 | 80.1196 (-0.1316) | 79.9624 (-0.2887) | 79.5612 (-0.6899) |
