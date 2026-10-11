# Flat Base–Gap output interaction

INTERNAL DIAGNOSTIC ONLY

MOSI seed66; eight original per-rate Test-oracle BEST checkpoints. No training.
Code snapshot: `8945717`; server biggpu, physical GPU 6 (`GPU-e4cafb17-818e-216a-b94a-7440063a9153`).
Source: `/data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/flat_seed66`.

One causal Memory scan per batch; replay original Flat, Local Skip, emotion_norm and task head.
All parameters and checkpoint files unchanged. Original prediction archives, labels and availability masks verified.
Only forward512 read values captured; backward512 reconstructed as exact zeros. Inactive Gap and padding safely masked.

`I_BG=f11-f10-f01+f00`; diagnostic additive prediction `fadd=f10+f01-f00`.
Full−additive differences measure removing this output interaction, not replacing Memory or evaluating a trained additive architecture.
W-F1 excludes neutral labels y=0 and uses prediction>0; interaction statistics use all valid utterances.
Macro means equally weight rates; subgroup means use nonempty rates. Counts sum repeated rate/sample evaluations, not unique utterances.

| Rate | Full W-F1 | Additive W-F1 | Full−additive pp | Rescue | Harm | Mean abs I | p95 abs I |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 88.205 | 88.205 | +0.000 | 0 | 0 | 0.00000 | 0.00000 |
| 0.1 | 86.507 | 86.497 | +0.009 | 1 | 1 | 0.01931 | 0.12991 |
| 0.2 | 83.187 | 83.335 | -0.148 | 0 | 1 | 0.03837 | 0.21706 |
| 0.3 | 80.763 | 80.763 | +0.000 | 0 | 0 | 0.04885 | 0.23831 |
| 0.4 | 80.827 | 80.827 | +0.000 | 0 | 0 | 0.06709 | 0.30120 |
| 0.5 | 77.494 | 77.147 | +0.347 | 4 | 2 | 0.07173 | 0.23603 |
| 0.6 | 75.790 | 75.576 | +0.213 | 4 | 3 | 0.08690 | 0.32711 |
| 0.7 | 75.773 | 74.036 | +1.738 | 18 | 8 | 0.14378 | 0.50280 |

## Rate-macro and availability/adjacent-polarity groups

| Group | Scope | Rates | Full | Additive | Difference pp | Rescue | Harm | Mean abs I |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | eight_rates | 8 | 81.068 | 80.798 | +0.270 | 27 | 15 | 0.05950 |
| all | high_missing | 3 | 76.352 | 75.586 | +0.766 | 26 | 13 | 0.10080 |
| no_Text | eight_rates | 7 | 68.784 | 68.476 | +0.307 | 22 | 12 | 0.13534 |
| no_Text | high_missing | 3 | 66.515 | 65.102 | +1.413 | 22 | 10 | 0.15238 |
| Text_present | eight_rates | 8 | 86.947 | 86.883 | +0.064 | 5 | 3 | 0.02568 |
| Text_present | high_missing | 3 | 86.593 | 86.480 | +0.113 | 4 | 3 | 0.04530 |
| A | eight_rates | 7 | 77.028 | 75.877 | +1.151 | 15 | 6 | 0.18922 |
| A | high_missing | 3 | 68.091 | 65.405 | +2.686 | 15 | 6 | 0.20871 |
| T | eight_rates | 7 | 83.729 | 83.538 | +0.191 | 4 | 2 | 0.06696 |
| T | high_missing | 3 | 88.316 | 87.870 | +0.447 | 4 | 2 | 0.06600 |
| V | eight_rates | 7 | 68.291 | 68.030 | +0.261 | 7 | 4 | 0.11746 |
| V | high_missing | 3 | 66.061 | 65.452 | +0.608 | 7 | 4 | 0.12098 |
| AT | eight_rates | 7 | 85.799 | 85.679 | +0.120 | 1 | 1 | 0.05517 |
| AT | high_missing | 3 | 86.970 | 87.395 | -0.425 | 0 | 1 | 0.05362 |
| AV | eight_rates | 7 | 66.118 | 66.514 | -0.396 | 0 | 2 | 0.10550 |
| AV | high_missing | 3 | 61.448 | 61.448 | +0.000 | 0 | 0 | 0.10037 |
| TV | eight_rates | 7 | 88.224 | 88.224 | +0.000 | 0 | 0 | 0.01356 |
| TV | high_missing | 3 | 86.295 | 86.295 | +0.000 | 0 | 0 | 0.01119 |
| ATV | eight_rates | 8 | 84.896 | 84.896 | +0.000 | 0 | 0 | 0.00000 |
| ATV | high_missing | 3 | 80.451 | 80.451 | +0.000 | 0 | 0 | 0.00000 |
| same | eight_rates | 8 | 88.744 | 88.521 | +0.223 | 14 | 8 | 0.06678 |
| same | high_missing | 3 | 85.085 | 84.492 | +0.594 | 14 | 8 | 0.11198 |
| A/same | eight_rates | 7 | 89.213 | 88.655 | +0.558 | 7 | 4 | 0.22144 |
| A/same | high_missing | 3 | 81.037 | 79.736 | +1.301 | 7 | 4 | 0.22840 |
| T/same | eight_rates | 7 | 90.808 | 90.687 | +0.121 | 1 | 0 | 0.07952 |
| T/same | high_missing | 3 | 92.753 | 92.470 | +0.283 | 1 | 0 | 0.06998 |
| V/same | eight_rates | 7 | 82.013 | 81.588 | +0.425 | 6 | 3 | 0.12385 |
| V/same | high_missing | 3 | 79.563 | 78.571 | +0.993 | 6 | 3 | 0.13463 |
| AT/same | eight_rates | 7 | 88.940 | 89.302 | -0.362 | 0 | 1 | 0.06119 |
| AT/same | high_missing | 3 | 91.163 | 92.009 | -0.846 | 0 | 1 | 0.06004 |
| AV/same | eight_rates | 7 | 83.367 | 83.367 | +0.000 | 0 | 0 | 0.12309 |
| AV/same | high_missing | 3 | 77.775 | 77.775 | +0.000 | 0 | 0 | 0.11149 |
| TV/same | eight_rates | 7 | 93.802 | 93.802 | +0.000 | 0 | 0 | 0.01510 |
| TV/same | high_missing | 3 | 93.322 | 93.322 | +0.000 | 0 | 0 | 0.01193 |
| ATV/same | eight_rates | 8 | 88.812 | 88.812 | +0.000 | 0 | 0 | 0.00000 |
| ATV/same | high_missing | 3 | 82.824 | 82.824 | +0.000 | 0 | 0 | 0.00000 |
| opposite | eight_rates | 8 | 67.381 | 67.067 | +0.315 | 11 | 7 | 0.05555 |
| opposite | high_missing | 3 | 60.553 | 59.562 | +0.991 | 10 | 5 | 0.09604 |
| A/opposite | eight_rates | 7 | 54.861 | 52.007 | +2.854 | 7 | 2 | 0.16941 |
| A/opposite | high_missing | 3 | 38.575 | 31.916 | +6.659 | 7 | 2 | 0.20565 |
| T/opposite | eight_rates | 7 | 64.179 | 64.179 | +0.000 | 2 | 2 | 0.05682 |
| T/opposite | high_missing | 3 | 80.969 | 80.969 | +0.000 | 2 | 2 | 0.07123 |
| V/opposite | eight_rates | 7 | 35.850 | 35.832 | +0.018 | 1 | 1 | 0.12040 |
| V/opposite | high_missing | 3 | 42.361 | 42.318 | +0.043 | 1 | 1 | 0.10937 |
| AT/opposite | eight_rates | 7 | 80.312 | 79.359 | +0.954 | 1 | 0 | 0.04768 |
| AT/opposite | high_missing | 3 | 81.804 | 81.804 | +0.000 | 0 | 0 | 0.04573 |
| AV/opposite | eight_rates | 7 | 36.956 | 38.035 | -1.079 | 0 | 2 | 0.09139 |
| AV/opposite | high_missing | 3 | 37.528 | 37.528 | +0.000 | 0 | 0 | 0.10325 |
| TV/opposite | eight_rates | 7 | 78.711 | 78.711 | +0.000 | 0 | 0 | 0.01312 |
| TV/opposite | high_missing | 3 | 73.246 | 73.246 | +0.000 | 0 | 0 | 0.01129 |
| ATV/opposite | eight_rates | 8 | 78.949 | 78.949 | +0.000 | 0 | 0 | 0.00000 |
| ATV/opposite | high_missing | 3 | 77.326 | 77.326 | +0.000 | 0 | 0 | 0.00000 |

## Descriptive finding

Retaining the output interaction changes rate-macro W-F1 by +0.270 pp overall and +0.766 pp at high missing. Across repeated rate/sample evaluations: 27 corrections, 15 harms.
The output is not exactly additive on all samples; its classification effect is concentrated in higher missing rates. This does not support a claim that original Flat entirely fails to use Base–Gap joint effects.
Both adjacent-polarity groups contain corrections and harms; opposite polarity alone is not evidence for suppressing the interaction.
Availability groups are current masks within random-missing evaluations, NOT separately trained single-modality models or whole-conversation persistent missing. Small subgroup cohorts can have unstable W-F1; per-rate N is supplied in per_rate.csv.


## Boundaries and interpretation

- Same/opposite means exactly adjacent nonneutral utterance polarity; first turns/neutral pairs excluded, no cross-conversation pairing or backward skipping.
- Gold labels define analysis groups only; no gold input, learned gate or test-label routing.
- Adjacent polarity is not the full Memory sentiment state or a confirmed emotion transition. Same-availability rows are supplied to control current support.
- Nonzero output interaction can arise in the adapter, normalization or head; this diagnostic does not localize it to a specific operation.
- Zeroing active reads can be out-of-distribution. Synthetic additive prediction is not a deployable score or proof interaction is indispensable.
- One backbone seed, descriptive paired analysis only; rates are not independent seeds. No significance or cross-seed stability claim.
- No threshold tuning, training, new model, loss, query, memory write or second missing view.

Complete distributions, all groups, per-utterance values and checkpoint hashes are in `results/`.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=GPU-e4cafb17-818e-216a-b94a-7440063a9153 python -m experiments.osram_flat_base_gap_interaction_20261011.run \
  --source /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/flat_seed66 \
  --output <new-empty-output-directory> --code-commit 8945717 --gpu 6
```
