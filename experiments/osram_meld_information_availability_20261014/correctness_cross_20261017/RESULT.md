# Legacy MELD Probe correctness cross tables

INTERNAL DIAGNOSTIC ONLY.

No training or inference. Read saved predictions only. Backbone seed66; probe seeds66/67/68.
Missing protocol: gcnet-legacy-fixed-v1, NOT current gcnet-random-mask-v1.
Original probe environment was Torch2.2.2+cu121; this is not the current cmerc-repro baseline.

## Seed66, missing0.7: exact paired counts

| Other probe | Both correct | Local correct / other wrong | Local wrong / other correct | Both wrong | N |
|---|---:|---:|---:|---:|---:|
| Base | 1108 | 293 | 161 | 1048 | 2610 |
| Memory | 1251 | 150 | 100 | 1109 | 2610 |
| LocalMemory | 1277 | 124 | 117 | 1092 | 2610 |

## Eight-rate averages over the three probe initializations

Counts below are average counts per rate/initialization, not distinct utterances or three backbone seeds.

| Other | Local wrong / other correct | Local correct / other wrong | ΔACC pp | ΔUA pp | ΔWF1 pp |
|---|---:|---:|---:|---:|---:|
| Base | 190.708 | 428.792 | -9.123 | -12.790 | -13.453 |
| Memory | 168.125 | 345.875 | -6.811 | -10.025 | -9.479 |
| LocalMemory | 94.667 | 110.000 | -0.588 | -0.397 | -0.341 |

## All rates: seed66 counts

| Rate | Other | Both correct | Local correct / other wrong | Local wrong / other correct | Both wrong | N | ΔACC pp |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.0 | Base | 1376 | 347 | 166 | 721 | 2610 | -6.935 |
| 0.0 | Memory | 1372 | 351 | 174 | 713 | 2610 | -6.782 |
| 0.0 | LocalMemory | 1649 | 74 | 57 | 830 | 2610 | -0.651 |
| 0.1 | Base | 1218 | 460 | 208 | 722 | 2608 | -9.663 |
| 0.1 | Memory | 1229 | 449 | 190 | 740 | 2608 | -9.931 |
| 0.1 | LocalMemory | 1548 | 130 | 97 | 833 | 2608 | -1.265 |
| 0.2 | Base | 1110 | 553 | 202 | 745 | 2610 | -13.448 |
| 0.2 | Memory | 1172 | 491 | 180 | 767 | 2610 | -11.916 |
| 0.2 | LocalMemory | 1564 | 99 | 98 | 849 | 2610 | -0.038 |
| 0.3 | Base | 1059 | 529 | 215 | 807 | 2610 | -12.031 |
| 0.3 | Memory | 1192 | 396 | 182 | 840 | 2610 | -8.199 |
| 0.3 | LocalMemory | 1480 | 108 | 105 | 917 | 2610 | -0.115 |
| 0.4 | Base | 1050 | 497 | 244 | 819 | 2610 | -9.693 |
| 0.4 | Memory | 1176 | 371 | 194 | 869 | 2610 | -6.782 |
| 0.4 | LocalMemory | 1431 | 116 | 102 | 961 | 2610 | -0.536 |
| 0.5 | Base | 1072 | 412 | 196 | 930 | 2610 | -8.276 |
| 0.5 | Memory | 1175 | 309 | 176 | 950 | 2610 | -5.096 |
| 0.5 | LocalMemory | 1362 | 122 | 120 | 1006 | 2610 | -0.077 |
| 0.6 | Base | 1077 | 377 | 184 | 972 | 2610 | -7.395 |
| 0.6 | Memory | 1211 | 243 | 158 | 998 | 2610 | -3.257 |
| 0.6 | LocalMemory | 1321 | 133 | 120 | 1036 | 2610 | -0.498 |
| 0.7 | Base | 1108 | 293 | 161 | 1048 | 2610 | -5.057 |
| 0.7 | Memory | 1251 | 150 | 100 | 1109 | 2610 | -1.916 |
| 0.7 | LocalMemory | 1277 | 124 | 117 | 1092 | 2610 | -0.268 |

## Alignment and interpretation limits

- Local is evaluated on the full cache; all three other probes use donor-included rows. Local is filtered with that exact boolean mask before comparison.
- Check cached conversation/utterance identities, donor target order and labels; check saved predictions equal argmax(logits) and reproduce original RESULT.json metrics.
- Predictions.npz does not itself contain IDs. Its row identity relies on the checked original extraction/fit construction and cache/donor order, not an independent ID field in that file.
- Base and Memory probes include availability and history-presence/count metadata, whereas Local includes availability. These are separately fitted probe heads, not original Flat readout interventions.
- Local-wrong/other-correct demonstrates complementary predictions, not a deployable routing rule. Gold correctness must not be used as a switch.
- Base versus Memory differences are not an isolated Gap contribution: both the inputs and fitted heads differ.
- Only Local versus LocalMemory compares the learned combined probe with Local; compare its corrections AND harms. Test-oracle probe selection remains optimistic.
- Three probe initializations share the same frozen backbone seed66. These tables do not establish three-seed current-model gains.
- No statistical significance or storage/query/decision bottleneck conclusion is established by these counts alone.
