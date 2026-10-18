# MELD small-LR: frozen Local / Memory / Full

INTERNAL DIAGNOSTIC ONLY. Readout interventions, not separately trained classifiers.

All three seeds completed frozen evaluation at all8 rates; no failures. All24
original Full logit tensors matched the exact saved selection logits bit-for-bit
(maximum absolute error0), with matching IDs/indices/speakers/labels and masks.
Checkpoint and availability hashes also match the corresponding training records.
No new masks, optimizer steps, model fitting or weights changes were introduced.

## Three-seed averages

| Readout | ACC | UA | W-F1 |
|---|---:|---:|---:|
| Local-only |60.704|34.990|56.564|
| Memory-only |38.515|24.871|37.428|
| Full |61.545|37.225|58.390|

Full minus Local: ACC +0.841315pp, UA +2.234917pp, W-F1 +1.825875pp.
All three seeds have positive Full-versus-Local ACC and UA changes.
Across8 repeated rate-specific evaluations per seed, corrections/harms are:
seed66 781/532; seed67 708/535; seed68 699/594. Counts are not unique utterances.
The low Memory-only scores do not establish poor historical information quality:
this disables both Local paths in a jointly trained readout, not a separate
Memory classifier, and queries remain current-input-conditioned.
Same per-rate ACC-best checkpoint and exact selected masks. One CHM scan per batch.
L: Local in adapter + original skip; BG: Base/active Gap only, both Local paths off;
LBG: Full. Norm, adapter biases and task head remain unchanged.

| Seed | Readout | ACC | UA | W-F1 | CE |
|---|---|---:|---:|---:|---:|
| 66 | L | 60.594 | 35.746 | 56.604 | 1.249 |
| 66 | BG | 41.504 | 26.000 | 41.420 | 6.228 |
| 66 | LBG | 61.786 | 38.435 | 58.918 | 1.194 |
| 67 | L | 60.632 | 34.884 | 56.550 | 1.243 |
| 67 | BG | 44.411 | 25.474 | 42.777 | 5.787 |
| 67 | LBG | 61.461 | 36.991 | 58.267 | 1.194 |
| 68 | L | 60.886 | 34.340 | 56.537 | 1.238 |
| 68 | BG | 29.631 | 23.140 | 28.088 | 8.346 |
| 68 | LBG | 61.389 | 36.248 | 57.985 | 1.189 |

## Per-rate results

| Seed | Rate | Readout | ACC | UA | W-F1 |
|---|---:|---|---:|---:|---:|
| 66 | 0.0 | L | 66.552 | 42.008 | 64.136 |
| 66 | 0.0 | BG | 47.816 | 29.481 | 47.603 |
| 66 | 0.0 | LBG | 67.663 | 44.495 | 65.861 |
| 66 | 0.1 | L | 64.828 | 43.736 | 62.927 |
| 66 | 0.1 | BG | 42.069 | 28.909 | 42.844 |
| 66 | 0.1 | LBG | 65.556 | 45.739 | 64.316 |
| 66 | 0.2 | L | 62.644 | 38.960 | 59.848 |
| 66 | 0.2 | BG | 42.682 | 27.334 | 43.116 |
| 66 | 0.2 | LBG | 64.176 | 40.984 | 61.960 |
| 66 | 0.3 | L | 61.264 | 37.668 | 58.004 |
| 66 | 0.3 | BG | 41.188 | 25.999 | 41.144 |
| 66 | 0.3 | LBG | 62.414 | 40.244 | 60.124 |
| 66 | 0.4 | L | 59.502 | 34.923 | 55.474 |
| 66 | 0.4 | BG | 41.762 | 25.164 | 41.282 |
| 66 | 0.4 | LBG | 60.843 | 38.784 | 58.146 |
| 66 | 0.5 | L | 58.697 | 31.235 | 53.338 |
| 66 | 0.5 | BG | 40.843 | 24.278 | 39.797 |
| 66 | 0.5 | LBG | 59.732 | 33.210 | 55.657 |
| 66 | 0.6 | L | 56.475 | 29.353 | 50.377 |
| 66 | 0.6 | BG | 37.088 | 23.490 | 37.116 |
| 66 | 0.6 | LBG | 57.433 | 32.168 | 53.087 |
| 66 | 0.7 | L | 54.789 | 28.084 | 48.729 |
| 66 | 0.7 | BG | 38.582 | 23.345 | 38.456 |
| 66 | 0.7 | LBG | 56.475 | 31.856 | 52.191 |
| 67 | 0.0 | L | 66.322 | 42.667 | 64.155 |
| 67 | 0.0 | BG | 50.920 | 27.463 | 47.522 |
| 67 | 0.0 | LBG | 66.935 | 43.991 | 65.213 |
| 67 | 0.1 | L | 65.326 | 40.944 | 62.697 |
| 67 | 0.1 | BG | 49.464 | 27.096 | 46.394 |
| 67 | 0.1 | LBG | 65.862 | 42.277 | 63.746 |
| 67 | 0.2 | L | 63.525 | 36.934 | 59.940 |
| 67 | 0.2 | BG | 44.751 | 25.610 | 43.111 |
| 67 | 0.2 | LBG | 63.985 | 38.877 | 61.301 |
| 67 | 0.3 | L | 60.421 | 33.780 | 56.375 |
| 67 | 0.3 | BG | 46.284 | 26.976 | 44.823 |
| 67 | 0.3 | LBG | 62.222 | 36.838 | 59.176 |
| 67 | 0.4 | L | 59.770 | 34.517 | 55.733 |
| 67 | 0.4 | BG | 45.211 | 26.316 | 43.703 |
| 67 | 0.4 | LBG | 61.188 | 37.352 | 57.917 |
| 67 | 0.5 | L | 58.123 | 32.558 | 53.484 |
| 67 | 0.5 | BG | 43.218 | 25.135 | 42.023 |
| 67 | 0.5 | LBG | 59.234 | 34.432 | 55.398 |
| 67 | 0.6 | L | 56.360 | 30.053 | 50.838 |
| 67 | 0.6 | BG | 39.617 | 23.952 | 39.046 |
| 67 | 0.6 | LBG | 56.935 | 32.601 | 52.882 |
| 67 | 0.7 | L | 55.211 | 27.619 | 49.179 |
| 67 | 0.7 | BG | 35.824 | 21.242 | 35.595 |
| 67 | 0.7 | LBG | 55.326 | 29.559 | 50.501 |
| 68 | 0.0 | L | 66.475 | 40.593 | 63.724 |
| 68 | 0.0 | BG | 22.950 | 21.953 | 16.033 |
| 68 | 0.0 | LBG | 66.628 | 41.395 | 64.253 |
| 68 | 0.1 | L | 64.636 | 36.416 | 60.560 |
| 68 | 0.1 | BG | 25.479 | 22.162 | 21.129 |
| 68 | 0.1 | LBG | 65.249 | 37.714 | 61.668 |
| 68 | 0.2 | L | 63.372 | 39.038 | 60.638 |
| 68 | 0.2 | BG | 31.494 | 24.072 | 31.413 |
| 68 | 0.2 | LBG | 63.716 | 40.480 | 61.654 |
| 68 | 0.3 | L | 61.992 | 38.365 | 58.940 |
| 68 | 0.3 | BG | 36.858 | 26.108 | 37.683 |
| 68 | 0.3 | LBG | 62.452 | 40.410 | 60.243 |
| 68 | 0.4 | L | 60.536 | 35.296 | 57.049 |
| 68 | 0.4 | BG | 34.598 | 24.529 | 35.720 |
| 68 | 0.4 | LBG | 60.690 | 37.905 | 58.404 |
| 68 | 0.5 | L | 58.544 | 31.973 | 53.511 |
| 68 | 0.5 | BG | 32.069 | 23.822 | 32.307 |
| 68 | 0.5 | LBG | 59.042 | 34.320 | 55.287 |
| 68 | 0.6 | L | 56.207 | 26.731 | 49.413 |
| 68 | 0.6 | BG | 25.249 | 21.540 | 22.802 |
| 68 | 0.6 | LBG | 57.165 | 28.634 | 51.501 |
| 68 | 0.7 | L | 55.326 | 26.306 | 48.464 |
| 68 | 0.7 | BG | 28.352 | 20.936 | 27.616 |
| 68 | 0.7 | LBG | 56.169 | 29.129 | 50.868 |

Removing inputs can be out of distribution. These scores measure this frozen
readout response, not the intrinsic information ceiling or a learned Local/Memory probe.
