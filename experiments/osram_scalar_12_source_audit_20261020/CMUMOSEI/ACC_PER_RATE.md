# Random eight-rate ACC 4×3 source-removal tables

INTERNAL DIAGNOSTIC ONLY

Dataset CMUMOSEI; seed 66; fold 1; source `1c5f29cb6c1592cdf0afe8874afc34a51ccd7938`.

Fixed original miss=0.3 checkpoint: `/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat/best_miss_0p3.pt`.

Checkpoint SHA256: `7905d518b4006d78f5f0d1139fc025ff190f9e1725da831d091292fd5575e0ab`.

No training; original Flat, Local Skip, Memory, Query and task head are unchanged. One fixed checkpoint is used for every rate/pattern, not the eight per-rate selected baseline models. The anchor was previously selected by Test-oracle; these are internal frozen interventions, not independent generalization results.

Rows are read destinations Base/Gap-A/Gap-T/Gap-V; columns are historical A/T/V Value-write sources. Each cell reports the final score after removing ONLY that one source contribution at the final Flat input. Local and all other evidence remain unchanged. Sources are algebraic joint-write Value-slot contributions, not pure modality semantics.

ACC/W-F1 use abs(gold)>0 and prediction>0; prediction=0 is nonpositive. MSE includes all valid labels. Each Gap row uses only utterances where that target modality is currently missing; its original score is recomputed on the identical cohort. Inactive/empty rows are N/A. Do not compare rows with different cohorts as if they were full-test scores.

Positive ACC/W-F1 Δ means removing the item improves that score; positive MSE Δ means removing it increases loss. Corrections/harms compare removed vs full on nonzero-label utterances only. No subgroup W-F1 values are averaged to reconstruct full-test W-F1.

## random 0.0

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 87.0391 | 87.1767 (+0.1376) | 86.3236 (-0.7155) | 87.0941 (+0.0550) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

## random 0.1

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 85.9384 | 85.9108 (-0.0275) | 85.1954 (-0.7430) | 85.8833 (-0.0550) |
| Gap-A | 486 | 371 | 88.9488 | 88.6792 (-0.2695) | 88.6792 (-0.2695) | 88.9488 (+0.0000) |
| Gap-T | 463 | 369 | 72.8997 | 72.6287 (-0.2710) | 75.6098 (+2.7100) | 72.3577 (-0.5420) |
| Gap-V | 466 | 372 | 87.3656 | 87.3656 (+0.0000) | 86.5591 (-0.8065) | 86.2903 (-1.0753) |

## random 0.2

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 85.4155 | 85.4981 (+0.0826) | 84.8927 (-0.5228) | 85.4155 (+0.0000) |
| Gap-A | 917 | 713 | 86.5358 | 86.9565 (+0.4208) | 86.8163 (+0.2805) | 86.2553 (-0.2805) |
| Gap-T | 934 | 705 | 78.7234 | 78.7234 (+0.0000) | 77.7305 (-0.9929) | 78.7234 (+0.0000) |
| Gap-V | 906 | 701 | 85.0214 | 84.7361 (-0.2853) | 85.0214 (+0.0000) | 84.5934 (-0.4280) |

## random 0.3

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 84.7001 | 84.6450 (-0.0550) | 83.9296 (-0.7705) | 84.7276 (+0.0275) |
| Gap-A | 1448 | 1143 | 85.3893 | 84.8644 (-0.5249) | 85.0394 (-0.3500) | 85.2143 (-0.1750) |
| Gap-T | 1371 | 1061 | 77.7568 | 77.7568 (+0.0000) | 76.8143 (-0.9425) | 77.8511 (+0.0943) |
| Gap-V | 1391 | 1088 | 86.5809 | 86.7647 (+0.1838) | 86.5809 (+0.0000) | 86.1213 (-0.4596) |

## random 0.4

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 82.5812 | 82.5537 (-0.0275) | 81.7281 (-0.8531) | 82.3886 (-0.1926) |
| Gap-A | 1865 | 1437 | 84.1336 | 84.2728 (+0.1392) | 83.7857 (-0.3479) | 83.9944 (-0.1392) |
| Gap-T | 1866 | 1454 | 76.7538 | 76.5475 (-0.2063) | 74.8968 (-1.8569) | 76.8913 (+0.1376) |
| Gap-V | 1843 | 1464 | 84.0847 | 83.9481 (-0.1366) | 83.8798 (-0.2049) | 83.9481 (-0.1366) |

## random 0.5

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 81.7556 | 81.8107 (+0.0550) | 81.2603 (-0.4953) | 81.8382 (+0.0826) |
| Gap-A | 2330 | 1792 | 84.1518 | 84.4866 (+0.3348) | 84.0960 (-0.0558) | 83.9286 (-0.2232) |
| Gap-T | 2370 | 1851 | 76.4992 | 76.7693 (+0.2701) | 75.7428 (-0.7563) | 76.4992 (+0.0000) |
| Gap-V | 2332 | 1841 | 82.0750 | 82.0206 (-0.0543) | 81.8034 (-0.2716) | 81.5861 (-0.4889) |

## random 0.6

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 79.2515 | 79.3616 (+0.1101) | 78.9763 (-0.2752) | 79.3616 (+0.1101) |
| Gap-A | 2828 | 2222 | 80.6031 | 80.4680 (-0.1350) | 80.4680 (-0.1350) | 80.4680 (-0.1350) |
| Gap-T | 2779 | 2126 | 74.0357 | 73.8946 (-0.1411) | 73.5654 (-0.4704) | 73.9417 (-0.0941) |
| Gap-V | 2777 | 2177 | 82.3610 | 82.2232 (-0.1378) | 82.0854 (-0.2756) | 82.2232 (-0.1378) |

## random 0.7

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4659 | 3634 | 78.1233 | 77.6555 (-0.4678) | 77.5729 (-0.5504) | 77.9857 (-0.1376) |
| Gap-A | 3083 | 2408 | 79.4850 | 79.0282 (-0.4568) | 79.2774 (-0.2076) | 79.4850 (+0.0000) |
| Gap-T | 3155 | 2447 | 74.1725 | 73.8455 (-0.3269) | 72.3743 (-1.7981) | 74.1316 (-0.0409) |
| Gap-V | 3080 | 2413 | 80.7708 | 80.6465 (-0.1243) | 80.5636 (-0.2072) | 80.2321 (-0.5387) |
