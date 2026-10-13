# MELD Flat Local/Base/Gap source-combination diagnostic

**INTERNAL DIAGNOSTIC ONLY.** These are frozen readout interventions on the
retained per-rate Test-oracle checkpoints. They are not seven separately
trained models and are not formal paper results.

## Protocol

- Dataset: MELD, official ECERC pickle and split.
- Model: causal cfg84 no-JEPA Flat OSRAM.
- Seeds: 66, 67, 68.
- Missing rates: 0.0--0.7 using the exact `gcnet-legacy-fixed-v1` masks that
  selected the source checkpoints.
- Memory, Local, Base and Gap are computed once per batch. Only the captured
  inputs to the original Flat readout are zeroed.
- `L` keeps Local in both `local_skip` and the adapter Local slice; `B` keeps
  Base; `G` keeps all three already-masked Gap slots.
- `LBG` replays the untouched original Flat path.

## Three-seed aggregate

All values are percentages. `High` averages missing rates 0.5/0.6/0.7.

| Sources | ACC 8-rate | ACC High | UA 8-rate | UA High | W-F1 8-rate | W-F1 High |
|---|---:|---:|---:|---:|---:|---:|
| L | 59.294 | 55.045 | 31.021 | 24.947 | 54.046 | 47.626 |
| B | 17.534 | 13.661 | 15.610 | 15.647 | 15.995 | 13.286 |
| G | 16.186 | 17.195 | 14.346 | 14.431 | 12.569 | 15.915 |
| L+B | 59.491 | 55.185 | 31.346 | 25.119 | 54.360 | 47.862 |
| L+G | 59.379 | 55.147 | 31.154 | 25.073 | 54.182 | 47.784 |
| B+G | 20.128 | 18.250 | 15.247 | 15.228 | 18.446 | 17.678 |
| L+B+G (original Flat) | **59.502** | **55.198** | **31.398** | **25.162** | **54.408** | **47.923** |

## Per-rate W-F1

| Sources | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L | 62.486 | 60.331 | 58.064 | 55.913 | 52.698 | 50.250 | 47.627 | 45.001 |
| B | 20.526 | 20.536 | 16.610 | 16.236 | 14.195 | 14.199 | 13.173 | 12.486 |
| G | 3.097 | 9.933 | 11.536 | 13.620 | 14.623 | 16.594 | 15.989 | 15.163 |
| L+B | 63.087 | 60.890 | 58.334 | 56.030 | 52.954 | 50.528 | 47.936 | 45.122 |
| L+G | 62.486 | 60.389 | 58.128 | 56.198 | 52.903 | 50.507 | 47.917 | 44.929 |
| B+G | 20.526 | 19.950 | 18.040 | 18.155 | 17.862 | 17.997 | 18.233 | 16.804 |
| L+B+G | **63.087** | **60.897** | **58.335** | **56.111** | **53.061** | **50.686** | **48.038** | **45.045** |

## Per-rate ACC

| Sources | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L | 65.428 | 63.563 | 61.826 | 60.128 | 58.276 | 56.769 | 54.930 | 53.436 |
| B | 23.946 | 23.985 | 18.557 | 17.880 | 14.917 | 14.623 | 13.461 | 12.899 |
| G | 13.129 | 15.300 | 15.722 | 16.628 | 17.126 | 18.238 | 17.267 | 16.079 |
| L+B | 65.875 | 63.985 | 61.928 | 60.128 | 58.455 | 56.960 | 55.109 | 53.487 |
| L+G | 65.428 | 63.589 | 61.801 | 60.358 | 58.416 | 56.909 | 55.147 | 53.384 |
| B+G | 23.946 | 23.410 | 20.153 | 20.192 | 18.570 | 18.672 | 18.825 | 17.254 |
| L+B+G | **65.875** | **63.946** | **61.903** | **60.179** | **58.519** | **57.011** | **55.185** | **53.397** |

## Per-rate UA

| Sources | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L | 39.432 | 37.372 | 34.623 | 32.593 | 29.305 | 26.991 | 25.076 | 22.775 |
| B | 15.426 | 15.744 | 15.433 | 15.682 | 15.652 | 15.801 | 15.266 | 15.873 |
| G | 14.286 | 14.014 | 14.473 | 14.206 | 14.500 | 15.183 | 14.165 | 13.944 |
| L+B | 40.129 | 37.981 | 35.019 | 32.759 | 29.521 | 27.235 | 25.248 | 22.875 |
| L+G | 39.432 | 37.473 | 34.762 | 32.861 | 29.481 | 27.240 | 25.291 | 22.687 |
| B+G | 15.426 | 15.692 | 15.309 | 15.196 | 14.671 | 15.317 | 15.809 | 14.559 |
| L+B+G | **40.129** | **38.033** | **35.074** | **32.853** | **29.611** | **27.391** | **25.321** | **22.773** |

## Seed-level W-F1 summary

| Seed | L | B | G | L+B | L+G | B+G | L+B+G |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 66, 8-rate | 54.456 | 17.939 | 14.443 | 54.484 | 54.487 | 25.463 | 54.530 |
| 67, 8-rate | 53.869 | 20.668 | 16.915 | 54.396 | 54.091 | 21.435 | 54.433 |
| 68, 8-rate | 53.814 | 9.379 | 6.350 | 54.201 | 53.969 | 8.440 | 54.260 |
| 66, High | 48.011 | 10.561 | 16.883 | 47.990 | 47.995 | 23.205 | 48.068 |
| 67, High | 46.652 | 17.590 | 22.329 | 47.156 | 46.936 | 20.149 | 47.140 |
| 68, High | 48.215 | 11.706 | 8.533 | 48.441 | 48.421 | 9.680 | 48.562 |

## Checks and interpretation boundary

- Across all rates and seeds, `LBG` classes exactly matched the original
  forward. Maximum absolute logit replay error was `1.765e-5`.
- Each seed used 72 batch scans over all eight rates; all seven combinations
  reused those scans.
- Local is the dominant readout source. Relative to `L`, adding both Base and
  Gap changes the three-seed 8-rate W-F1 by only `+0.362` pp and High W-F1 by
  `+0.297` pp in this frozen intervention.
- The isolated B/G results are not standalone modality models. Zeroing trained
  inputs is distribution-shifting, and adapter/classifier biases remain. The
  scores therefore describe this checkpoint's readout response, not additive
  or causal source contributions.

Raw per-seed summaries are stored under `results/`.
