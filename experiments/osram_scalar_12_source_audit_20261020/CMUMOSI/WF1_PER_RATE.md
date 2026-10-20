# Random eight-rate WF1 4×3 source-removal tables

INTERNAL DIAGNOSTIC ONLY

Dataset CMUMOSI; seed 66; fold 1; source `1c5f29cb6c1592cdf0afe8874afc34a51ccd7938`.

Fixed original miss=0.3 checkpoint: `/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_66/flat/best_miss_0p3.pt`.

Checkpoint SHA256: `98640a71e7082751c98b66e61463fc9f1b81078e3b90f8c234df1be86d6c730e`.

No training; original Flat, Local Skip, Memory, Query and task head are unchanged. One fixed checkpoint is used for every rate/pattern, not the eight per-rate selected baseline models. The anchor was previously selected by Test-oracle; these are internal frozen interventions, not independent generalization results.

Rows are read destinations Base/Gap-A/Gap-T/Gap-V; columns are historical A/T/V Value-write sources. Each cell reports the final score after removing ONLY that one source contribution at the final Flat input. Local and all other evidence remain unchanged. Sources are algebraic joint-write Value-slot contributions, not pure modality semantics.

ACC/W-F1 use abs(gold)>0 and prediction>0; prediction=0 is nonpositive. MSE includes all valid labels. Each Gap row uses only utterances where that target modality is currently missing; its original score is recomputed on the identical cohort. Inactive/empty rows are N/A. Do not compare rows with different cohorts as if they were full-test scores.

Positive ACC/W-F1 Δ means removing the item improves that score; positive MSE Δ means removing it increases loss. Corrections/harms compare removed vs full on nonzero-label utterances only. No subgroup W-F1 values are averaged to reconstruct full-test W-F1.

## random 0.0

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 86.2270 | 86.2516 (+0.0246) | 86.2437 (+0.0167) | 85.7893 (-0.4378) |
| Gap-A | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-T | 0 | 0 | N/A | N/A | N/A | N/A |
| Gap-V | 0 | 0 | N/A | N/A | N/A | N/A |

## random 0.1

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 83.3347 | 83.8335 (+0.4988) | 83.5111 (+0.1764) | 83.2147 (-0.1199) |
| Gap-A | 83 | 80 | 79.8869 | 79.8869 (+0.0000) | 79.8869 (+0.0000) | 79.8869 (+0.0000) |
| Gap-T | 65 | 62 | 61.2903 | 59.5348 (-1.7555) | 56.8532 (-4.4371) | 63.0128 (+1.7225) |
| Gap-V | 56 | 52 | 79.2301 | 81.1986 (+1.9685) | 79.2301 (+0.0000) | 83.0065 (+3.7763) |

## random 0.2

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 81.7469 | 81.6361 (-0.1108) | 81.7716 (+0.0247) | 81.3181 (-0.4288) |
| Gap-A | 133 | 127 | 84.5883 | 84.5883 (+0.0000) | 84.5883 (+0.0000) | 84.5883 (+0.0000) |
| Gap-T | 150 | 145 | 65.0218 | 63.4730 (-1.5488) | 65.5040 (+0.4822) | 62.8489 (-2.1729) |
| Gap-V | 136 | 132 | 81.6861 | 81.6861 (+0.0000) | 81.6861 (+0.0000) | 80.0724 (-1.6137) |

## random 0.3

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 82.1220 | 81.6982 (-0.4238) | 81.9743 (-0.1477) | 81.5504 (-0.5717) |
| Gap-A | 205 | 198 | 84.8485 | 84.3503 (-0.4982) | 84.3358 (-0.5127) | 84.8485 (+0.0000) |
| Gap-T | 221 | 214 | 67.0097 | 66.9399 (-0.0698) | 68.0004 (+0.9906) | 65.6495 (-1.3602) |
| Gap-V | 190 | 182 | 89.0110 | 89.5388 (+0.5278) | 90.0680 (+1.0570) | 86.7573 (-2.2536) |

## random 0.4

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 77.0180 | 77.8234 (+0.8054) | 77.0363 (+0.0183) | 77.3405 (+0.3224) |
| Gap-A | 260 | 251 | 77.0233 | 77.0233 (+0.0000) | 77.0233 (+0.0000) | 77.0233 (+0.0000) |
| Gap-T | 277 | 265 | 64.6871 | 64.7062 (+0.0192) | 62.0181 (-2.6689) | 65.0517 (+0.3647) |
| Gap-V | 261 | 252 | 78.8941 | 79.3174 (+0.4233) | 78.4127 (-0.4814) | 77.9829 (-0.9112) |

## random 0.5

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 74.0977 | 73.9948 (-0.1029) | 73.0494 (-1.0483) | 74.0134 (-0.0843) |
| Gap-A | 355 | 338 | 77.6365 | 77.9336 (+0.2971) | 77.6419 (+0.0054) | 77.6365 (+0.0000) |
| Gap-T | 344 | 327 | 62.6650 | 63.0305 (+0.3655) | 61.3778 (-1.2871) | 63.0305 (+0.3655) |
| Gap-V | 347 | 331 | 74.2235 | 74.2235 (+0.0000) | 74.5059 (+0.2825) | 73.9873 (-0.2361) |

## random 0.6

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 72.4280 | 72.3552 (-0.0728) | 72.0124 (-0.4156) | 72.3464 (-0.0817) |
| Gap-A | 406 | 390 | 75.2093 | 75.4691 (+0.2599) | 75.2144 (+0.0051) | 75.4666 (+0.2573) |
| Gap-T | 415 | 400 | 63.6390 | 64.1804 (+0.5413) | 62.5555 (-1.0835) | 63.6938 (+0.0547) |
| Gap-V | 419 | 399 | 76.8720 | 77.1072 (+0.2353) | 77.8134 (+0.9415) | 78.0875 (+1.2156) |

## random 0.7

W-F1: removal score (Δ = removed − original, percentage points for ACC/W-F1).

| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |
|---|---:|---:|---:|---:|---:|---:|
| Base | 686 | 656 | 67.0113 | 66.6455 (-0.3658) | 65.8184 (-1.1930) | 66.6338 (-0.3775) |
| Gap-A | 455 | 432 | 71.6944 | 71.4646 (-0.2298) | 71.0029 (-0.6915) | 71.9249 (+0.2305) |
| Gap-T | 465 | 447 | 59.3524 | 59.4433 (+0.0909) | 56.8365 (-2.5159) | 58.3237 (-1.0287) |
| Gap-V | 452 | 433 | 70.0474 | 70.0474 (+0.0000) | 70.6938 (+0.6464) | 70.5440 (+0.4966) |
