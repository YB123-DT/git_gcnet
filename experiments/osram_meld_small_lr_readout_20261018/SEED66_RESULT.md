# MELD small-LR: frozen Local / Memory / Full

INTERNAL DIAGNOSTIC ONLY. Readout interventions, not separately trained classifiers.
Same per-rate ACC-best checkpoint and exact selected masks. One CHM scan per batch.
L: Local in adapter + original skip; BG: Base/active Gap only, both Local paths off;
LBG: Full. Norm, adapter biases and task head remain unchanged.

| Seed | Readout | ACC | UA | W-F1 | CE |
|---|---|---:|---:|---:|---:|
| 66 | L | 60.594 | 35.746 | 56.604 | 1.249 |
| 66 | BG | 41.504 | 26.000 | 41.420 | 6.228 |
| 66 | LBG | 61.786 | 38.435 | 58.918 | 1.194 |

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

Removing inputs can be out of distribution. These scores measure this frozen
readout response, not the intrinsic information ceiling or a learned Local/Memory probe.
