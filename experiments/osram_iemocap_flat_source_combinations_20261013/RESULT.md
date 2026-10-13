# IEMOCAP Flat Local/Base/Gap source-combination diagnostic

**INTERNAL DIAGNOSTIC ONLY.** Frozen interventions on retained per-rate
Test-oracle cfg84 no-JEPA Flat checkpoints; these are not separately
trained source models. Five LOSO folds are pooled within each seed, then
the three seed metrics are averaged.

`L` keeps both Local paths, `B` keeps Base, and `G` keeps all three
already-masked Gap slots. Memory/query/write and the task head are fixed.

## IEMOCAP 4-class

| Sources | ACC mean8 | ACC high | UA mean8 | UA high | W-F1 mean8 | W-F1 high |
|---|---:|---:|---:|---:|---:|---:|
| L | 61.386 | 56.299 | 59.399 | 53.817 | 61.187 | 55.846 |
| B | 69.464 | 68.175 | 70.719 | 69.491 | 68.983 | 67.697 |
| G | 47.664 | 58.818 | 48.991 | 60.309 | 46.272 | 58.231 |
| L+B | 77.638 | 74.355 | 77.561 | 73.945 | 77.688 | 74.438 |
| L+G | 71.259 | 71.846 | 70.408 | 71.184 | 71.350 | 71.938 |
| B+G | 69.674 | 68.563 | 71.082 | 70.104 | 69.133 | 67.994 |
| L+B+G | 79.286 | 77.127 | 79.618 | 77.422 | 79.238 | 77.066 |

### Per-rate ACC: principal nested comparison

| Sources | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L | 68.734 | 67.179 | 64.485 | 62.243 | 59.549 | 57.898 | 56.548 | 54.451 |
| L+B | 82.125 | 81.293 | 79.847 | 77.804 | 76.972 | 75.134 | 74.640 | 73.290 |
| L+B+G | 82.125 | 81.619 | 80.739 | 79.467 | 78.955 | 77.810 | 77.442 | 76.128 |

On 8-rate ACC, `L+B - L` is `+16.252` pp and `L+B+G - L+B` is `+1.648` pp. The total frozen Memory increment `L+B+G - L` is `+17.900` pp.

## IEMOCAP 6-class

| Sources | ACC mean8 | ACC high | UA mean8 | UA high | W-F1 mean8 | W-F1 high |
|---|---:|---:|---:|---:|---:|---:|
| L | 43.593 | 39.242 | 40.800 | 35.998 | 41.874 | 36.926 |
| B | 50.244 | 49.274 | 51.452 | 50.329 | 49.662 | 48.694 |
| G | 32.940 | 41.126 | 33.817 | 42.318 | 31.334 | 40.062 |
| L+B | 60.157 | 57.173 | 58.525 | 54.925 | 60.030 | 56.971 |
| L+G | 52.901 | 53.895 | 50.849 | 51.813 | 52.194 | 53.422 |
| B+G | 50.135 | 49.226 | 51.572 | 50.603 | 49.477 | 48.565 |
| L+B+G | 61.226 | 58.980 | 60.250 | 57.813 | 61.158 | 58.909 |

### Per-rate ACC: principal nested comparison

| Sources | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L | 49.348 | 48.191 | 46.473 | 43.975 | 43.029 | 40.522 | 39.419 | 37.786 |
| L+B | 64.169 | 63.223 | 61.891 | 61.101 | 59.352 | 58.254 | 57.348 | 55.917 |
| L+B+G | 64.169 | 63.465 | 62.379 | 62.236 | 60.617 | 59.855 | 59.101 | 57.985 |

On 8-rate ACC, `L+B - L` is `+16.564` pp and `L+B+G - L+B` is `+1.069` pp. The total frozen Memory increment `L+B+G - L` is `+17.633` pp.

## Relation to the MELD diagnostic

Under the same frozen readout-intervention definition, IEMOCAP's
8-rate ACC increment from `L` to `L+B+G` is +17.900 pp (4-class) and
+17.633 pp (6-class). The retained MELD diagnostic reports only
+0.208 pp. This is a qualitative cross-dataset contrast: label spaces,
features, checkpoints, and missing-mask protocols differ, so it does
not isolate conversation length as the cause.

## Validity boundary

Every untouched full forward exactly matches the saved source prediction,
label, and evaluation availability. Recomputing the numerically identical
Flat path changed one argmax among all 311,136 evaluated
seed/fold/rate utterance instances because of a near-tied floating-point
boundary (maximum logit error 2.384e-5); full metrics therefore use the
untouched forward as the authoritative anchor. Zeroing trained inputs is distribution
shifting, so isolated `B`/`G` scores are readout-response diagnostics,
not standalone modality-model performance or causal decomposition.
