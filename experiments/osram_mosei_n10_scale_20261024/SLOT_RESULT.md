# MOSEI N10 — one correction slot at a time

INTERNAL DIAGNOSTIC ONLY

The user explicitly requested separate strengths, not a shared global coefficient.
This completed replay tests exactly one Local/Base/Gap-A/Gap-T/Gap-V correction
at0/.5/1 with all four others fixed at1. Eleven unique readouts per batch:one
full-strength common reference plus ten non-full interventions. No training,
new checkpoints, seeds, coefficients or combined-coefficient search.

## Eight-rate mean W-F1

| Only adjusted correction slot | gamma0 | gamma.5 | gamma1 |
|---|---:|---:|---:|
| Local | 83.386364 | 83.384585 | 83.377462 |
| Base | 83.377089 | 83.369292 | 83.377462 |
| Gap-A | 83.374788 | 83.390281 | 83.377462 |
| Gap-T | 83.433518 | 83.395763 | 83.377462 |
| Gap-V | 83.391069 | 83.386082 | 83.377462 |

## High-missing mean W-F1 (.5/.6/.7)

| Only adjusted correction slot | gamma0 | gamma.5 | gamma1 |
|---|---:|---:|---:|
| Local | 80.004888 | 80.007710 | 79.956560 |
| Base | 80.044396 | 79.990502 | 79.956560 |
| Gap-A | 79.969074 | 79.990745 | 79.956560 |
| Gap-T | 80.094499 | 80.027757 | 79.956560 |
| Gap-V | 79.994870 | 79.980573 | 79.956560 |

## All conditions, including ACC

| Condition | ACC | W-F1 | High W-F1 | ΔW-F1 vs full (pp) |
|---|---:|---:|---:|---:|
| 1.0 | 83.592460 | 83.377462 | 79.956560 | 0.000000 |
| Base_0.0 | 83.609659 | 83.377089 | 80.044396 | -0.000373 |
| Base_0.5 | 83.592460 | 83.369292 | 79.990502 | -0.008170 |
| Gap-A_0.0 | 83.592460 | 83.374788 | 79.969074 | -0.002673 |
| Gap-A_0.5 | 83.606219 | 83.390281 | 79.990745 | 0.012819 |
| Gap-T_0.0 | 83.661255 | 83.433518 | 80.094499 | 0.056056 |
| Gap-T_0.5 | 83.616538 | 83.395763 | 80.027757 | 0.018301 |
| Gap-V_0.0 | 83.609659 | 83.391069 | 79.994870 | 0.013607 |
| Gap-V_0.5 | 83.602779 | 83.386082 | 79.980573 | 0.008620 |
| Local_0.0 | 83.606219 | 83.386364 | 80.004888 | 0.008903 |
| Local_0.5 | 83.602779 | 83.384585 | 80.007710 | 0.007124 |

## Narrow conclusion

Among these fixed test-diagnostic settings, disabling only Gap-T correction is
best:ACC83.661255/W-F1 83.433518/high W-F1 80.094499. Relative to the original
N10 it changes W-F1 by+0.056056pp and high W-F1 by+0.137939pp. Relative to the
independently trained Flat128 seed66 (83.387044/high80.011427), the corresponding
changes are+0.046474pp/+0.083072pp. These are small single-seed changes, not
evidence of a stable learned method or an independently validated advantage.

Very importantly this does NOT delete original Gap-T. It retains original
Memory/Gap-T and only removes the extra Nested input correction for that slot.
LocalSkip always uses original Local. Base, other Gap corrections, backbone,
head and queries/Memory writes remain the same as full N10.

Disabling Gap-T correction versus full N10 produces37corrections/17harms across
all eight repeated rate evaluations. A-only positives gain14correct decisions
without new harms; A-only negatives receive0corrections/4harms. Turning off
Gap-A correction cannot affect A-only examples because Gap-A is inactive there.
Counts are repeated rate-utterance records, NOT unique samples or W-F1 points.

Do not add these slot-wise gains or combine the preferred coefficients and
assume their benefits sum:the following Flat/LN/head remain nonlinear. No such
combination has been tested or automatically launched.

## Execution and independent checks

- Same completed N10 seed66 eight per-rate Test-best weights and exact GCNet masks.
- cmerc-repro, no feature/model/sampler/environment change;131 numerical files
  byte-identical to the immutable N10 source.
-23 focused tests passed locally and remotely after observed red tests.
-176 Memory scans total (22batches per rate);11 Flat/head replays per scan.
- Full-strength prediction and metrics reproduce protected original predictions,
  within1e-6 and exact polarity. First-turn and observed-target Gap interventions
  preserve predictions within float tolerance and exact polarity.
-88 saved prediction files:hashes, labels/IDs/indices/masks and ACC/W-F1/MSE
  independently verified; all paired-count partitions and aggregates reconcile.
- Source checkpoints and frozen model state unchanged; readouts do not advance CUDA RNG.
- Wall time:78.348seconds including setup, physicalGPU1.
- Source commit:`b6764fd09b0a7be61a9e0b216d99c3579a7a3b9c`.
- Remote output:`/data2/yb/remote_experiments/osram_mosei_n10_scale_20261024/slot_replay_v1`.
- Original remote SUMMARY SHA256:`4f1ec7dad5fd2b50eef33f0cd111197d3b7b49a7bfcf4b5f570e2e5a6fe5ed7f`.
- Source_v3/slot_replay_v1 are separate from both prior global-strength runs,
  including failed replay_v1 and completed replay_v2. Historical data and any
  active training jobs remain untouched.

All88 per-rate rows are in `slot_results/PER_RATE.csv`; the compact machine
record, provenance hashes, metrics and paired totals are in `slot_results/SUMMARY.json`.
Full original cohort details and predictions remain in the remote output.
