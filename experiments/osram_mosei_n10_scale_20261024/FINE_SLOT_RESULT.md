# MOSEI N10 — finer one-slot correction strengths

INTERNAL DIAGNOSTIC ONLY

Gamma0.1/0.2 refinement, not missing rates0.1/0.2. Each correction is adjusted separately; other four stay at1. Same completed N10 seed66, eight fixed per-rate Test-best weights and exact GCNet masks. No training or combined-coefficient selection. Only additional Nested corrections change; original LocalSkip and original Base/Gap remain intact. Gamma0 is not separately trained Flat.

## Eight-rate W-F1

| Slot | gamma0 | gamma0.1 | gamma0.2 | gamma0.5 | gamma1 |
|---|---:|---:|---:|---:|---:|
| Local | 83.386364 | 83.386364 | 83.383562 | 83.384585 | 83.377462 |
| Base | 83.377089 | 83.367984 | 83.370985 | 83.369292 | 83.377462 |
| Gap-A | 83.374788 | 83.378993 | 83.378993 | 83.390281 | 83.377462 |
| Gap-T | 83.433518 | 83.434026 | 83.422168 | 83.395763 | 83.377462 |
| Gap-V | 83.391069 | 83.387796 | 83.384597 | 83.386082 | 83.377462 |

## High-missing (.5/.6/.7) W-F1

| Slot | gamma0 | gamma0.1 | gamma0.2 | gamma0.5 | gamma1 |
|---|---:|---:|---:|---:|---:|
| Local | 80.004888 | 80.004888 | 80.014902 | 80.007710 | 79.956560 |
| Base | 80.044396 | 80.020116 | 80.026206 | 79.990502 | 79.956560 |
| Gap-A | 79.969074 | 79.970560 | 79.970560 | 79.990745 | 79.956560 |
| Gap-T | 80.094499 | 80.095855 | 80.081840 | 80.027757 | 79.956560 |
| Gap-V | 79.994870 | 79.994870 | 79.986340 | 79.980573 | 79.956560 |

## New cases and paired decisions

| Slot | gamma | ACC | W-F1 | Corrections vs full | Harms vs full |
|---|---:|---:|---:|---:|---:|
| Local | 0.1 | 83.606219 | 83.386364 | 15 | 11 |
| Local | 0.2 | 83.602779 | 83.383562 | 13 | 10 |
| Base | 0.1 | 83.599340 | 83.367984 | 36 | 34 |
| Base | 0.2 | 83.599340 | 83.370985 | 30 | 28 |
| Gap-A | 0.1 | 83.595900 | 83.378993 | 4 | 3 |
| Gap-A | 0.2 | 83.595900 | 83.378993 | 4 | 3 |
| Gap-T | 0.1 | 83.661255 | 83.434026 | 36 | 16 |
| Gap-T | 0.2 | 83.647496 | 83.422168 | 30 | 14 |
| Gap-V | 0.1 | 83.606219 | 83.387796 | 9 | 5 |
| Gap-V | 0.2 | 83.602779 | 83.384597 | 8 | 5 |

Counts sum repeated rate-utterance records (29072 nonzero-gold records), not unique utterances or W-F1 gains. MSE includes neutral gold; polarity metrics exclude it.

## Gap-T by missing rate

| Rate | gamma0 WF1 | gamma0.1 WF1 | gamma0.2 WF1 | gamma1 WF1 |
|---|---:|---:|---:|---:|
| 0.0 | 87.369885 | 87.369885 | 87.369885 | 87.369885 |
| 0.1 | 86.725780 | 86.725780 | 86.699260 | 86.699260 |
| 0.2 | 85.344328 | 85.344328 | 85.344328 | 85.402157 |
| 0.3 | 84.633277 | 84.633277 | 84.606979 | 84.561335 |
| 0.4 | 83.111373 | 83.111373 | 83.111373 | 83.117375 |
| 0.5 | 81.956901 | 81.930872 | 81.930872 | 81.778345 |
| 0.6 | 79.825348 | 79.855446 | 79.833957 | 79.739347 |
| 0.7 | 78.501248 | 78.501248 | 78.480691 | 78.351989 |

## Narrow result

Gap-T gamma0.1 is numerically best:ACC 83.661255, W-F1 83.434026, high W-F1 80.095855. Versus full, W-F1 changes by +0.056565pp overall/+0.139295pp high-missing. Versus gamma0, the changes are only +0.000509pp/+0.001356pp:effectively tied with switching off the extra Gap-T correction.

At gamma0.1 there are36corrections/16harms versus full; gamma0.2 gives30/14. A-only positive records gain14/0 atgamma0.1; A-only negatives have0/4. Effects are not uniform across conditions. Do not combine independently preferred slot coefficients and assume additive gains.

Separately-trained Flat128 seed66 is W-F1 83.387044/high 80.011427. The small numerical advantage is not evidence of stable multi-seed improvement. Weights and coefficients are compared using test data; this is exploratory, not independently validated deployment performance.

## Verification and provenance

-35 focused tests passed locally and remotely;11 expected red tests preceded the fine-grid implementation. The default joint and old one-slot grids remain unchanged.
-88 prediction hashes/metrics independently verified, including labels, IDs, masks, corrections/harms and aggregation. Full-strength predictions bitwise reproduce the previous replay.
-First-turn and inactive Gap interventions preserve predictions within1e-6 and exact polarity; source checkpoint hashes and frozen parameters remain unchanged.
-176 Memory scans,11 readouts per batch, no optimizer or parameter updates. Readouts preserve CUDA RNG;131 numerical source files match the protected original.
-Wall time:83.104seconds including setup. biggpu physicalGPU1, UUID GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049, cmerc-repro.
-Immutable source:4cd1ee2e76aeb4bdd47b2c0a37b9e75c1ae7a9f2.
-Remote source:/data2/yb/remote_experiments/osram_mosei_n10_scale_20261024/source_v4
-Remote output:/data2/yb/remote_experiments/osram_mosei_n10_scale_20261024/fine_slot_replay_v1
-tmux:mosei_n10_fine_slot_20261025 (completed).
-Full remote SUMMARY SHA256:fcd10b417121de71e16c7a978696858084d408c9eb68385610c7b10734504c5a.
-New machine records:fine_slot_results/SUMMARY.json and PER_RATE.csv. Gamma0/.5 references:preserved slot_results/SUMMARY.json. Historical outputs/checkpoints and other active jobs were not changed.

