# MOSI causal readout ablation

**INTERNAL DIAGNOSTIC ONLY / NOT A FORMAL PAPER RESULT**

Status: **complete**.

Protocol: independently maximize test-oracle weighted F1 across all 100 epochs for each seed/rate; earliest epoch wins ties.
Full is inherited, not rerun. Checkpoint best.pt uses an 8-rate mean and is not used for this table.
Scores and paired deltas are in percentage points; std is sample std across five seeds.

| Variant | Missing rate | n | F1 mean ± std | Δ Full |
| --- | --- | --- | --- | --- |
| full | 0.0 | 5 | 87.417 ± 0.224 | +0.000 |
| full | 0.1 | 5 | 85.398 ± 0.719 | +0.000 |
| full | 0.2 | 5 | 81.935 ± 1.195 | +0.000 |
| full | 0.3 | 5 | 80.907 ± 0.890 | +0.000 |
| full | 0.4 | 5 | 78.088 ± 1.033 | +0.000 |
| full | 0.5 | 5 | 76.630 ± 1.064 | +0.000 |
| full | 0.6 | 5 | 75.090 ± 0.835 | +0.000 |
| full | 0.7 | 5 | 74.553 ± 2.172 | +0.000 |
| local-only | 0.0 | 5 | 87.179 ± 0.173 | -0.237 |
| local-only | 0.1 | 5 | 84.200 ± 0.500 | -1.198 |
| local-only | 0.2 | 5 | 80.768 ± 1.350 | -1.168 |
| local-only | 0.3 | 5 | 79.449 ± 1.362 | -1.458 |
| local-only | 0.4 | 5 | 75.909 ± 1.135 | -2.179 |
| local-only | 0.5 | 5 | 72.907 ± 2.014 | -3.723 |
| local-only | 0.6 | 5 | 72.698 ± 1.321 | -2.392 |
| local-only | 0.7 | 5 | 70.934 ± 2.087 | -3.618 |
| local-base | 0.0 | 5 | 87.061 ± 0.254 | -0.356 |
| local-base | 0.1 | 5 | 84.645 ± 0.708 | -0.752 |
| local-base | 0.2 | 5 | 81.678 ± 0.817 | -0.257 |
| local-base | 0.3 | 5 | 80.688 ± 0.805 | -0.219 |
| local-base | 0.4 | 5 | 77.739 ± 1.485 | -0.349 |
| local-base | 0.5 | 5 | 76.189 ± 1.510 | -0.441 |
| local-base | 0.6 | 5 | 75.655 ± 0.873 | +0.564 |
| local-base | 0.7 | 5 | 73.709 ± 2.768 | -0.843 |
| local-gap | 0.0 | 5 | 87.356 ± 0.382 | -0.061 |
| local-gap | 0.1 | 5 | 84.854 ± 0.590 | -0.544 |
| local-gap | 0.2 | 5 | 81.918 ± 1.412 | -0.017 |
| local-gap | 0.3 | 5 | 81.326 ± 0.630 | +0.418 |
| local-gap | 0.4 | 5 | 77.864 ± 1.252 | -0.224 |
| local-gap | 0.5 | 5 | 76.389 ± 1.477 | -0.241 |
| local-gap | 0.6 | 5 | 75.519 ± 1.134 | +0.428 |
| local-gap | 0.7 | 5 | 73.923 ± 2.699 | -0.630 |

## Overall (seed-first averages)

all8 averages rates 0.0–0.7; high averages rates 0.5, 0.6, 0.7 within each seed first. Positive seeds have a strictly positive paired average delta versus Full.

| Variant | Rates | n | F1 mean ± std | Δ Full | Positive seeds |
| --- | --- | --- | --- | --- | --- |
| full | all8 | 5 | 80.002 ± 0.468 | +0.000 | 0/5 |
| full | high | 5 | 75.424 ± 0.877 | +0.000 | 0/5 |
| local-only | all8 | 5 | 78.006 ± 0.320 | -1.997 | 0/5 |
| local-only | high | 5 | 72.180 ± 0.250 | -3.244 | 0/5 |
| local-base | all8 | 5 | 79.671 ± 0.311 | -0.332 | 1/5 |
| local-base | high | 5 | 75.184 ± 0.735 | -0.240 | 2/5 |
| local-gap | all8 | 5 | 79.893 ± 0.205 | -0.109 | 1/5 |
| local-gap | high | 5 | 75.277 ± 0.436 | -0.148 | 2/5 |
## Selected epochs (completed, validated records only)

| Variant | Seed | Rate | Epoch | F1 |
| --- | --- | --- | --- | --- |
| full | 66 | 0.0 | 30 | 87.512 |
| full | 66 | 0.1 | 30 | 86.280 |
| full | 66 | 0.2 | 51 | 81.912 |
| full | 66 | 0.3 | 30 | 80.414 |
| full | 66 | 0.4 | 47 | 79.394 |
| full | 66 | 0.5 | 48 | 77.099 |
| full | 66 | 0.6 | 30 | 74.541 |
| full | 66 | 0.7 | 29 | 74.828 |
| local-only | 66 | 0.0 | 36 | 87.012 |
| local-only | 66 | 0.1 | 36 | 84.854 |
| local-only | 66 | 0.2 | 36 | 80.538 |
| local-only | 66 | 0.3 | 50 | 78.678 |
| local-only | 66 | 0.4 | 50 | 76.861 |
| local-only | 66 | 0.5 | 95 | 72.986 |
| local-only | 66 | 0.6 | 38 | 70.759 |
| local-only | 66 | 0.7 | 35 | 71.665 |
| local-base | 66 | 0.0 | 45 | 87.251 |
| local-base | 66 | 0.1 | 72 | 85.243 |
| local-base | 66 | 0.2 | 49 | 82.066 |
| local-base | 66 | 0.3 | 47 | 80.373 |
| local-base | 66 | 0.4 | 39 | 79.542 |
| local-base | 66 | 0.5 | 75 | 77.718 |
| local-base | 66 | 0.6 | 45 | 74.481 |
| local-base | 66 | 0.7 | 54 | 74.853 |
| local-gap | 66 | 0.0 | 34 | 86.714 |
| local-gap | 66 | 0.1 | 47 | 84.952 |
| local-gap | 66 | 0.2 | 39 | 80.693 |
| local-gap | 66 | 0.3 | 61 | 80.959 |
| local-gap | 66 | 0.4 | 48 | 79.297 |
| local-gap | 66 | 0.5 | 48 | 76.967 |
| local-gap | 66 | 0.6 | 48 | 74.555 |
| local-gap | 66 | 0.7 | 48 | 74.391 |
| full | 67 | 0.0 | 41 | 87.576 |
| full | 67 | 0.1 | 41 | 85.930 |
| full | 67 | 0.2 | 41 | 83.857 |
| full | 67 | 0.3 | 54 | 80.477 |
| full | 67 | 0.4 | 41 | 76.634 |
| full | 67 | 0.5 | 54 | 76.257 |
| full | 67 | 0.6 | 93 | 75.160 |
| full | 67 | 0.7 | 59 | 76.363 |
| local-only | 67 | 0.0 | 27 | 87.111 |
| local-only | 67 | 0.1 | 45 | 84.088 |
| local-only | 67 | 0.2 | 47 | 82.733 |
| local-only | 67 | 0.3 | 47 | 78.637 |
| local-only | 67 | 0.4 | 25 | 74.178 |
| local-only | 67 | 0.5 | 62 | 70.723 |
| local-only | 67 | 0.6 | 26 | 74.191 |
| local-only | 67 | 0.7 | 40 | 72.337 |
| local-base | 67 | 0.0 | 47 | 86.863 |
| local-base | 67 | 0.1 | 47 | 85.038 |
| local-base | 67 | 0.2 | 47 | 82.761 |
| local-base | 67 | 0.3 | 91 | 79.730 |
| local-base | 67 | 0.4 | 47 | 75.496 |
| local-base | 67 | 0.5 | 96 | 73.927 |
| local-base | 67 | 0.6 | 59 | 76.015 |
| local-base | 67 | 0.7 | 84 | 76.045 |
| local-gap | 67 | 0.0 | 40 | 87.630 |
| local-gap | 67 | 0.1 | 40 | 85.664 |
| local-gap | 67 | 0.2 | 40 | 84.270 |
| local-gap | 67 | 0.3 | 40 | 80.730 |
| local-gap | 67 | 0.4 | 40 | 75.866 |
| local-gap | 67 | 0.5 | 41 | 75.077 |
| local-gap | 67 | 0.6 | 40 | 75.610 |
| local-gap | 67 | 0.7 | 38 | 76.389 |
| full | 68 | 0.0 | 39 | 87.637 |
| full | 68 | 0.1 | 28 | 85.186 |
| full | 68 | 0.2 | 47 | 82.012 |
| full | 68 | 0.3 | 47 | 81.716 |
| full | 68 | 0.4 | 47 | 78.601 |
| full | 68 | 0.5 | 26 | 77.857 |
| full | 68 | 0.6 | 31 | 75.387 |
| full | 68 | 0.7 | 48 | 71.171 |
| local-only | 68 | 0.0 | 42 | 87.378 |
| local-only | 68 | 0.1 | 42 | 84.216 |
| local-only | 68 | 0.2 | 76 | 81.443 |
| local-only | 68 | 0.3 | 47 | 81.624 |
| local-only | 68 | 0.4 | 34 | 76.658 |
| local-only | 68 | 0.5 | 38 | 76.156 |
| local-only | 68 | 0.6 | 38 | 73.570 |
| local-only | 68 | 0.7 | 38 | 67.247 |
| local-base | 68 | 0.0 | 38 | 87.360 |
| local-base | 68 | 0.1 | 43 | 84.840 |
| local-base | 68 | 0.2 | 67 | 81.792 |
| local-base | 68 | 0.3 | 64 | 81.942 |
| local-base | 68 | 0.4 | 42 | 78.374 |
| local-base | 68 | 0.5 | 42 | 76.976 |
| local-base | 68 | 0.6 | 42 | 76.622 |
| local-base | 68 | 0.7 | 61 | 69.004 |
| local-gap | 68 | 0.0 | 27 | 87.466 |
| local-gap | 68 | 0.1 | 27 | 84.864 |
| local-gap | 68 | 0.2 | 27 | 81.994 |
| local-gap | 68 | 0.3 | 43 | 80.987 |
| local-gap | 68 | 0.4 | 76 | 78.018 |
| local-gap | 68 | 0.5 | 45 | 78.179 |
| local-gap | 68 | 0.6 | 29 | 77.400 |
| local-gap | 68 | 0.7 | 46 | 69.352 |
| full | 69 | 0.0 | 46 | 87.231 |
| full | 69 | 0.1 | 92 | 84.451 |
| full | 69 | 0.2 | 46 | 81.133 |
| full | 69 | 0.3 | 36 | 81.984 |
| full | 69 | 0.4 | 37 | 78.151 |
| full | 69 | 0.5 | 58 | 76.913 |
| full | 69 | 0.6 | 91 | 76.271 |
| full | 69 | 0.7 | 56 | 76.475 |
| local-only | 69 | 0.0 | 42 | 87.046 |
| local-only | 69 | 0.1 | 41 | 83.472 |
| local-only | 69 | 0.2 | 41 | 79.472 |
| local-only | 69 | 0.3 | 41 | 79.946 |
| local-only | 69 | 0.4 | 41 | 75.338 |
| local-only | 69 | 0.5 | 77 | 71.990 |
| local-only | 69 | 0.6 | 68 | 72.216 |
| local-only | 69 | 0.7 | 34 | 71.951 |
| local-base | 69 | 0.0 | 45 | 86.754 |
| local-base | 69 | 0.1 | 43 | 83.440 |
| local-base | 69 | 0.2 | 40 | 81.078 |
| local-base | 69 | 0.3 | 35 | 80.679 |
| local-base | 69 | 0.4 | 38 | 77.403 |
| local-base | 69 | 0.5 | 40 | 76.885 |
| local-base | 69 | 0.6 | 71 | 76.124 |
| local-base | 69 | 0.7 | 38 | 75.023 |
| local-gap | 69 | 0.0 | 50 | 87.643 |
| local-gap | 69 | 0.1 | 54 | 84.005 |
| local-gap | 69 | 0.2 | 33 | 81.648 |
| local-gap | 69 | 0.3 | 33 | 82.237 |
| local-gap | 69 | 0.4 | 47 | 77.823 |
| local-gap | 69 | 0.5 | 32 | 77.067 |
| local-gap | 69 | 0.6 | 26 | 74.735 |
| local-gap | 69 | 0.7 | 57 | 75.292 |
| full | 70 | 0.0 | 46 | 87.129 |
| full | 70 | 0.1 | 53 | 85.142 |
| full | 70 | 0.2 | 51 | 80.763 |
| full | 70 | 0.3 | 56 | 79.944 |
| full | 70 | 0.4 | 78 | 77.657 |
| full | 70 | 0.5 | 46 | 75.023 |
| full | 70 | 0.6 | 65 | 74.091 |
| full | 70 | 0.7 | 53 | 73.927 |
| local-only | 70 | 0.0 | 38 | 87.351 |
| local-only | 70 | 0.1 | 48 | 84.370 |
| local-only | 70 | 0.2 | 66 | 79.653 |
| local-only | 70 | 0.3 | 62 | 78.357 |
| local-only | 70 | 0.4 | 55 | 76.508 |
| local-only | 70 | 0.5 | 43 | 72.681 |
| local-only | 70 | 0.6 | 49 | 72.755 |
| local-only | 70 | 0.7 | 43 | 71.472 |
| local-base | 70 | 0.0 | 46 | 87.078 |
| local-base | 70 | 0.1 | 42 | 84.667 |
| local-base | 70 | 0.2 | 44 | 80.693 |
| local-base | 70 | 0.3 | 46 | 80.718 |
| local-base | 70 | 0.4 | 42 | 77.880 |
| local-base | 70 | 0.5 | 29 | 75.439 |
| local-base | 70 | 0.6 | 33 | 75.032 |
| local-base | 70 | 0.7 | 65 | 73.622 |
| local-gap | 70 | 0.0 | 58 | 87.324 |
| local-gap | 70 | 0.1 | 52 | 84.783 |
| local-gap | 70 | 0.2 | 58 | 80.987 |
| local-gap | 70 | 0.3 | 52 | 81.716 |
| local-gap | 70 | 0.4 | 77 | 78.314 |
| local-gap | 70 | 0.5 | 51 | 74.654 |
| local-gap | 70 | 0.6 | 51 | 75.294 |
| local-gap | 70 | 0.7 | 51 | 74.191 |
