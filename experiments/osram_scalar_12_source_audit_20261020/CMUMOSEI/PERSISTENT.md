# Persistent whole-conversation source-removal tables

INTERNAL DIAGNOSTIC ONLY

Dataset CMUMOSEI; seed 66; fold 1; source `1c5f29cb6c1592cdf0afe8874afc34a51ccd7938`.

Fixed original miss=0.3 checkpoint: `/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat/best_miss_0p3.pt`.

Checkpoint SHA256: `7905d518b4006d78f5f0d1139fc025ff190f9e1725da831d091292fd5575e0ab`.

No training; original Flat, Local Skip, Memory, Query and task head are unchanged. One fixed checkpoint is used for every rate/pattern, not the eight per-rate selected baseline models. The anchor was previously selected by Test-oracle; these are internal frozen interventions, not independent generalization results.

Rows are read destinations Base/Gap-A/Gap-T/Gap-V; columns are historical A/T/V Value-write sources. Each cell reports the final score after removing ONLY that one source contribution at the final Flat input. Local and all other evidence remain unchanged. Sources are algebraic joint-write Value-slot contributions, not pure modality semantics.

ACC/W-F1 use abs(gold)>0 and prediction>0; prediction=0 is nonpositive. MSE includes all valid labels. Each Gap row uses only utterances where that target modality is currently missing; its original score is recomputed on the identical cohort. Inactive/empty rows are N/A. Do not compare rows with different cohorts as if they were full-test scores.

Positive ACC/W-F1 Δ means removing the item improves that score; positive MSE Δ means removing it increases loss. Corrections/harms compare removed vs full on nonzero-label utterances only. No subgroup W-F1 values are averaged to reconstruct full-test W-F1.

## persistent A

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

## persistent T

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

## persistent V

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

## persistent AT

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

## persistent AV

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

## persistent TV

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

## persistent ATV

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
