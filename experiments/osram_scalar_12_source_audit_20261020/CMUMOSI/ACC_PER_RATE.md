# Random eight-rate ACC 4×3 source-removal tables

INTERNAL DIAGNOSTIC ONLY

Dataset CMUMOSI; seed 66; fold 1; source `1c5f29cb6c1592cdf0afe8874afc34a51ccd7938`.

Fixed original miss=0.3 checkpoint: `/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_66/flat/best_miss_0p3.pt`.

Checkpoint SHA256: `98640a71e7082751c98b66e61463fc9f1b81078e3b90f8c234df1be86d6c730e`.

No training; original Flat, Local Skip, Memory, Query and task head are unchanged. One fixed checkpoint is used for every rate/pattern, not the eight per-rate selected baseline models. The anchor was previously selected by Test-oracle; these are internal frozen interventions, not independent generalization results.

Rows are read destinations Base/Gap-A/Gap-T/Gap-V; columns are historical A/T/V Value-write sources. Each cell reports the final score after removing ONLY that one source contribution at the final Flat input. Local and all other evidence remain unchanged. Sources are algebraic joint-write Value-slot contributions, not pure modality semantics.

ACC/W-F1 use abs(gold)>0 and prediction>0; prediction=0 is nonpositive. MSE includes all valid labels. Each Gap row uses only utterances where that target modality is currently missing; its original score is recomputed on the identical cohort. Inactive/empty rows are N/A. Do not compare rows with different cohorts as if they were full-test scores.

Positive ACC/W-F1 Δ means removing the item improves that score; positive MSE Δ means removing it increases loss. Corrections/harms compare removed vs full on nonzero-label utterances only. No subgroup W-F1 values are averaged to reconstruct full-test W-F1.

## random 0.0

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 86.2805 | 86.2805 (+0.0000) | 86.2805 (+0.0000) | 85.8232 (-0.4573) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

## random 0.1

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 83.3841 | 83.8415 (+0.4573) | 83.5366 (+0.1524) | 83.2317 (-0.1524) |
| Gap-A | 83 | 80 | 80.0000 | 80.0000 (+0.0000) | 80.0000 (+0.0000) | 80.0000 (+0.0000) |
| Gap-T | 65 | 62 | 61.2903 | 59.6774 (-1.6129) | 56.4516 (-4.8387) | 62.9032 (+1.6129) |
| Gap-V | 56 | 52 | 78.8462 | 80.7692 (+1.9231) | 78.8462 (+0.0000) | 82.6923 (+3.8462) |

## random 0.2

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 81.8598 | 81.7073 (-0.1524) | 81.8598 (+0.0000) | 81.4024 (-0.4573) |
| Gap-A | 133 | 127 | 84.2520 | 84.2520 (+0.0000) | 84.2520 (+0.0000) | 84.2520 (+0.0000) |
| Gap-T | 150 | 145 | 65.5172 | 64.1379 (-1.3793) | 66.2069 (+0.6897) | 63.4483 (-2.0690) |
| Gap-V | 136 | 132 | 81.8182 | 81.8182 (+0.0000) | 81.8182 (+0.0000) | 80.3030 (-1.5152) |

## random 0.3

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 82.1646 | 81.7073 (-0.4573) | 82.0122 (-0.1524) | 81.5549 (-0.6098) |
| Gap-A | 205 | 198 | 84.8485 | 84.3434 (-0.5051) | 84.3434 (-0.5051) | 84.8485 (+0.0000) |
| Gap-T | 221 | 214 | 66.8224 | 66.8224 (+0.0000) | 67.7570 (+0.9346) | 65.4206 (-1.4019) |
| Gap-V | 190 | 182 | 89.0110 | 89.5604 (+0.5495) | 90.1099 (+1.0989) | 86.8132 (-2.1978) |

## random 0.4

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 76.9817 | 77.7439 (+0.7622) | 76.9817 (+0.0000) | 77.2866 (+0.3049) |
| Gap-A | 260 | 251 | 76.8924 | 76.8924 (+0.0000) | 76.8924 (+0.0000) | 76.8924 (+0.0000) |
| Gap-T | 277 | 265 | 64.5283 | 64.5283 (+0.0000) | 61.8868 (-2.6415) | 64.9057 (+0.3774) |
| Gap-V | 261 | 252 | 78.9683 | 79.3651 (+0.3968) | 78.5714 (-0.3968) | 78.1746 (-0.7937) |

## random 0.5

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 74.0854 | 73.9329 (-0.1524) | 73.0183 (-1.0671) | 73.9329 (-0.1524) |
| Gap-A | 355 | 338 | 77.5148 | 77.8107 (+0.2959) | 77.5148 (+0.0000) | 77.5148 (+0.0000) |
| Gap-T | 344 | 327 | 62.6911 | 62.9969 (+0.3058) | 61.4679 (-1.2232) | 62.9969 (+0.3058) |
| Gap-V | 347 | 331 | 74.3202 | 74.3202 (+0.0000) | 74.6224 (+0.3021) | 74.3202 (+0.0000) |

## random 0.6

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 72.4085 | 72.2561 (-0.1524) | 71.9512 (-0.4573) | 72.2561 (-0.1524) |
| Gap-A | 406 | 390 | 75.1282 | 75.3846 (+0.2564) | 75.1282 (+0.0000) | 75.3846 (+0.2564) |
| Gap-T | 415 | 400 | 63.5000 | 64.0000 (+0.5000) | 62.5000 (-1.0000) | 63.5000 (+0.0000) |
| Gap-V | 419 | 399 | 77.1930 | 77.4436 (+0.2506) | 78.1955 (+1.0025) | 78.4461 (+1.2531) |

## random 0.7

ACC: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 66.9207 | 66.4634 (-0.4573) | 65.7012 (-1.2195) | 66.4634 (-0.4573) |
| Gap-A | 455 | 432 | 71.5278 | 71.2963 (-0.2315) | 70.8333 (-0.6944) | 71.7593 (+0.2315) |
| Gap-T | 465 | 447 | 59.2841 | 59.2841 (+0.0000) | 56.8233 (-2.4609) | 58.1655 (-1.1186) |
| Gap-V | 452 | 433 | 70.2079 | 70.2079 (+0.0000) | 70.9007 (+0.6928) | 70.9007 (+0.6928) |
