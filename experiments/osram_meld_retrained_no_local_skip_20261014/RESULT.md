# MELD retrained no-Local-shortcut ablation

**INTERNAL TEST-ORACLE DIAGNOSTIC ONLY.** This is one seed and is not a
multi-seed paper result. Each missing rate uses the epoch with the highest test
accuracy, matching the retained MELD Flat protocol.

The model was retrained from scratch for 50 epochs with only the final
`local_skip(Local)` term disabled. Local remains inside
`emotion_adapter([Local, Base, Gap-A, Gap-T, Gap-V])`. OSRAM Memory, Base/Gap,
the task head, optimizer, cyclic 0.0--0.7 training, and
`gcnet-legacy-fixed-v1` masks are unchanged.

## Seed66 results

All values are percent.

| Missing rate | Flat epoch | Flat ACC | Flat W-F1 | No-skip epoch | No-skip ACC | No-skip W-F1 | ACC delta | W-F1 delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 5 | 65.977 | 63.204 | 22 | 66.667 | 63.735 | +0.690 | +0.532 |
| 0.1 | 5 | 63.870 | 61.038 | 22 | 65.134 | 62.359 | +1.264 | +1.320 |
| 0.2 | 2 | 62.605 | 58.628 | 10 | 63.831 | 60.217 | +1.226 | +1.589 |
| 0.3 | 2 | 59.847 | 55.291 | 22 | 61.188 | 57.830 | +1.341 | +2.539 |
| 0.4 | 2 | 58.774 | 53.872 | 14 | 59.732 | 55.817 | +0.958 | +1.945 |
| 0.5 | 2 | 57.011 | 50.893 | 17 | 57.318 | 52.258 | +0.307 | +1.366 |
| 0.6 | 2 | 55.326 | 48.119 | 20 | 56.169 | 50.362 | +0.843 | +2.242 |
| 0.7 | 2 | 53.678 | 45.190 | 20 | 54.598 | 47.688 | +0.920 | +2.498 |
| **8-rate mean** | — | **59.636** | **54.530** | — | **60.580** | **56.283** | **+0.943** | **+1.754** |
| **High mean (0.5/0.6/0.7)** | — | **55.338** | **48.068** | — | **56.028** | **50.103** | **+0.690** | **+2.035** |

## Interpretation and provenance

- Unlike the frozen shortcut-off intervention, retraining did not collapse.
  The remaining Adapter path learned to carry the task without the explicit
  Local shortcut.
- Seed66 improves at every evaluated missing rate, but one seed under
  per-rate Test-oracle selection is insufficient to establish a stable method
  improvement.
- Implementation/training commit: `60903a6`.
- Remote artifacts:
  `/data2/yb/remote_experiments/osram_meld_retrained_no_local_skip_20261014/seed66_60903a6`.
- Training completed all 50 epochs and retained all eight per-rate best
  checkpoints.
