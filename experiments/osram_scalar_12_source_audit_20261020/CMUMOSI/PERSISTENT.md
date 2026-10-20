# Persistent whole-conversation source-removal tables

INTERNAL DIAGNOSTIC ONLY

Dataset CMUMOSI; seed 66; fold 1; source `1c5f29cb6c1592cdf0afe8874afc34a51ccd7938`.

Fixed original miss=0.3 checkpoint: `/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_66/flat/best_miss_0p3.pt`.

Checkpoint SHA256: `98640a71e7082751c98b66e61463fc9f1b81078e3b90f8c234df1be86d6c730e`.

No training; original Flat, Local Skip, Memory, Query and task head are unchanged. One fixed checkpoint is used for every rate/pattern, not the eight per-rate selected baseline models. The anchor was previously selected by Test-oracle; these are internal frozen interventions, not independent generalization results.

Rows are read destinations Base/Gap-A/Gap-T/Gap-V; columns are historical A/T/V Value-write sources. Each cell reports the final score after removing ONLY that one source contribution at the final Flat input. Local and all other evidence remain unchanged. Sources are algebraic joint-write Value-slot contributions, not pure modality semantics.

ACC/W-F1 use abs(gold)>0 and prediction>0; prediction=0 is nonpositive. MSE includes all valid labels. Each Gap row uses only utterances where that target modality is currently missing; its original score is recomputed on the identical cohort. Inactive/empty rows are N/A. Do not compare rows with different cohorts as if they were full-test scores.

Positive ACC/W-F1 Δ means removing the item improves that score; positive MSE Δ means removing it increases loss. Corrections/harms compare removed vs full on nonzero-label utterances only. No subgroup W-F1 values are averaged to reconstruct full-test W-F1.

## persistent A

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

## persistent T

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

## persistent V

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

## persistent AT

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

## persistent AV

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

## persistent TV

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

## persistent ATV

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
