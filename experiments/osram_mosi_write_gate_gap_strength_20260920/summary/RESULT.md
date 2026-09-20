# MOSI write-gate / Gap residual-strength follow-up

INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT

Protocol: cfg84 causal OSRAM eta=.6, per-seed × per-rate Test-oracle selection.

| Variant | Rate | Mean W-F1 | SD | Δ previous Full |
|---|---:|---:|---:|---:|
| external-beta-r1 | 0.0 | 87.320 | 0.519 | -0.739 |
| external-beta-r1 | 0.1 | 85.001 | 0.350 | -0.385 |
| external-beta-r1 | 0.2 | 82.281 | 0.838 | -0.519 |
| external-beta-r1 | 0.3 | 81.328 | 1.019 | -0.002 |
| external-beta-r1 | 0.4 | 77.712 | 1.849 | -1.324 |
| external-beta-r1 | 0.5 | 76.279 | 1.786 | -0.471 |
| external-beta-r1 | 0.6 | 75.474 | 0.768 | -0.197 |
| external-beta-r1 | 0.7 | 74.089 | 3.059 | -0.440 |
| embedded-r05 | 0.0 | 87.412 | 0.668 | -0.647 |
| embedded-r05 | 0.1 | 84.674 | 1.258 | -0.713 |
| embedded-r05 | 0.2 | 81.922 | 1.370 | -0.878 |
| embedded-r05 | 0.3 | 80.750 | 0.516 | -0.579 |
| embedded-r05 | 0.4 | 78.008 | 1.367 | -1.028 |
| embedded-r05 | 0.5 | 75.987 | 0.809 | -0.763 |
| embedded-r05 | 0.6 | 75.531 | 0.589 | -0.140 |
| embedded-r05 | 0.7 | 74.040 | 2.718 | -0.490 |
| external-beta-r05 | 0.0 | 87.309 | 0.253 | -0.750 |
| external-beta-r05 | 0.1 | 85.123 | 0.646 | -0.263 |
| external-beta-r05 | 0.2 | 82.325 | 1.534 | -0.476 |
| external-beta-r05 | 0.3 | 81.116 | 1.061 | -0.214 |
| external-beta-r05 | 0.4 | 77.520 | 1.589 | -1.516 |
| external-beta-r05 | 0.5 | 76.051 | 1.791 | -0.698 |
| external-beta-r05 | 0.6 | 75.418 | 0.752 | -0.253 |
| external-beta-r05 | 0.7 | 74.202 | 2.869 | -0.327 |

## Overall

| Variant | Rates | Mean ± SD | Δ Full | Positive seeds |
|---|---|---:|---:|---:|
| external-beta-r1 | all8 | 79.935 ± 0.476 | -0.510 | 0/5 |
| external-beta-r1 | high | 75.281 ± 0.933 | -0.369 | 2/5 |
| embedded-r05 | all8 | 79.790 ± 0.719 | -0.655 | 0/5 |
| embedded-r05 | high | 75.186 ± 1.065 | -0.464 | 0/5 |
| external-beta-r05 | all8 | 79.883 ± 0.654 | -0.562 | 0/5 |
| external-beta-r05 | high | 75.224 ± 0.905 | -0.426 | 1/5 |
