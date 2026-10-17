# MELD Memory write-parameter screen — final

INTERNAL TEST-ORACLE SCREEN. All five authorized runs completed50/50 epochs; no failures.

Source/data/training protocol: see README.md and LAUNCH.md. Numerical source is
`b4a313841f098487a6482d2712bdac94aa8e319e`, driver is
`568daacb37a97f6d52dc52318e9e5537f470ad88`. MELD ECERC, seed66, batch32,
Adam LR.001 constant, dropout.5, weight decay1e-5, original GCNet random-mask-v1,
cyclic0.0–0.7, causal Flat1600 with Local Skip, no-JEPA. Only the named write
scalar changes; no additional modules or losses.

## Eight-rate ACC-selected results

| Run | Config | ACC | Corresponding UA | WF1 | ΔACC pp | ΔUA pp | ΔWF1 pp |
|---|---|---:|---:|---:|---:|---:|---:|
| W0 | 原参数对照 | 60.838 | 35.633 | 56.789 | 0.000 | 0.000 | 0.000 |
| W1 | write_step=0.3 | 60.948 | 35.653 | 57.054 | 0.110 | 0.021 | 0.265 |
| W2 | write_step=0.9 | 60.580 | 36.111 | 56.566 | -0.259 | 0.478 | -0.223 |
| W3 | write_ridge=1e-4 | 60.709 | 35.999 | 56.819 | -0.129 | 0.366 | 0.030 |
| W4 | write_ridge=1e-2 | 60.560 | 36.717 | 56.711 | -0.278 | 1.084 | -0.078 |

## High-missing mean (0.5/0.6/0.7)

| Run | ACC | UA | WF1 |
|---|---:|---:|---:|
| W0 | 56.718 | 29.025 | 50.812 |
| W1 | 56.654 | 29.784 | 51.170 |
| W2 | 56.679 | 29.569 | 50.917 |
| W3 | 56.526 | 28.986 | 50.860 |
| W4 | 56.411 | 29.564 | 50.608 |

## Per-rate ACC, all from ACC-selected forwards

| Miss | W0 | W1 | W2 | W3 | W4 |
|---|---:|---:|---:|---:|---:|
| 0.0 | 66.513 | 66.743 | 66.322 | 66.513 | 66.322 |
| 0.1 | 64.904 | 65.249 | 64.636 | 65.134 | 65.019 |
| 0.2 | 63.448 | 63.410 | 63.065 | 63.065 | 63.180 |
| 0.3 | 61.648 | 61.954 | 61.341 | 61.648 | 61.111 |
| 0.4 | 60.038 | 60.268 | 59.234 | 59.732 | 59.617 |
| 0.5 | 58.506 | 58.697 | 58.123 | 58.391 | 57.931 |
| 0.6 | 56.360 | 56.360 | 56.130 | 55.977 | 56.015 |
| 0.7 | 55.287 | 54.904 | 55.785 | 55.211 | 55.287 |

## Per-rate UA, all from ACC-selected forwards

| Miss | W0 | W1 | W2 | W3 | W4 |
|---|---:|---:|---:|---:|---:|
| 0.0 | 48.084 | 43.622 | 45.131 | 39.853 | 47.543 |
| 0.1 | 41.469 | 41.053 | 43.232 | 43.595 | 44.776 |
| 0.2 | 38.797 | 39.203 | 42.035 | 41.853 | 41.438 |
| 0.3 | 36.463 | 35.947 | 36.803 | 39.623 | 36.942 |
| 0.4 | 33.174 | 36.050 | 32.980 | 36.108 | 34.346 |
| 0.5 | 29.734 | 29.531 | 31.704 | 28.808 | 33.515 |
| 0.6 | 27.905 | 33.052 | 29.852 | 28.622 | 28.918 |
| 0.7 | 29.437 | 26.769 | 27.152 | 29.527 | 26.260 |

## Per-rate WF1, all from ACC-selected forwards

| Miss | W0 | W1 | W2 | W3 | W4 |
|---|---:|---:|---:|---:|---:|
| 0.0 | 65.217 | 64.804 | 64.209 | 63.720 | 64.802 |
| 0.1 | 62.670 | 62.780 | 62.397 | 62.879 | 63.023 |
| 0.2 | 60.636 | 60.285 | 60.567 | 60.673 | 60.593 |
| 0.3 | 57.639 | 58.605 | 57.547 | 58.719 | 57.200 |
| 0.4 | 55.713 | 56.446 | 55.060 | 55.983 | 56.248 |
| 0.5 | 52.509 | 52.465 | 52.940 | 51.989 | 52.499 |
| 0.6 | 50.646 | 52.692 | 50.299 | 50.191 | 50.813 |
| 0.7 | 49.280 | 48.354 | 49.512 | 50.399 | 48.510 |

## Integrity and interpretation

- All five initialization hashes and full50-epoch training/selection mask streams matched; final controller checks passed.
- Model parameter count13,290,795 throughout. All runs retain eight ACC-best checkpoints, exact selected predictions/masks and full last_training.pt on biggpu.
- Final selected predictions were checked against selected checkpoint metrics, not resampled final evaluation masks. UA/WF1 are from the ACC-selected epoch's same forward.
- W1 has the highest ACC in this five-run screen, but its +0.110pp ACC and +0.021pp UA are small single-seed changes, not proof of a robust improvement.
- W4 has the highest UA, with lower ACC and WF1 than W0. Stronger ridge is not an overall performance win.
- No automatic next search or three-seed expansion. These results do not replace the accepted historical three-seed baseline.
- The per-rate oracle means summarize different selected checkpoints; they are not one checkpoint's deployable score. There is no independent common-mask final replay in this task.
- Torch peak allocated564.897MiB each. Worker wall times vary with colocated workloads, so no clean runtime comparison is claimed.
- Machine-readable configs, per-rate metrics, winning epochs, mask/artifact hashes and provenance: FINAL_SUMMARY.json. Outputs remain at the recorded biggpu paths; weights are not uploaded.
