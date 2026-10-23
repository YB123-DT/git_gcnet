# Frozen MOSI66 TRAIN-exemplar ON/OFF

INTERNAL TEST-ORACLE DIAGNOSTIC. No retraining. Exact original test masks.

OFF bypasses only retrieval from the SAME ON-selected checkpoint; Base/Gap/Local/Flat/head stay trained and unchanged.
Primary ON/OFF share one Memory scan; an additional disabled-wrapper forward independently verifies OFF.

| Variant | Scope | Original Flat WF1 | OFF WF1 | ON WF1 | OFF−Flat | ON−OFF | ON−Flat |
|---|---|---:|---:|---:|---:|---:|---:|
| feature-only | mean | 79.249 | 78.602 | 78.644 | -0.647 | +0.042 | -0.605 |
| feature-only | high | 73.410 | 73.426 | 73.165 | +0.016 | -0.262 | -0.245 |
| label-aware | mean | 79.249 | 78.923 | 79.255 | -0.326 | +0.332 | +0.006 |
| label-aware | high | 73.410 | 73.045 | 73.019 | -0.365 | -0.026 | -0.391 |

## Per-rate switch contribution

| Variant | Missing | Epoch | OFF WF1 | ON WF1 | ON−OFF | Corrections | Harms | Mean abs prediction change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| feature-only | 0.0 | 37 | 86.831 | 86.896 | +0.066 | 9 | 9 | 0.545376 |
| feature-only | 0.1 | 83 | 82.213 | 83.463 | +1.250 | 11 | 3 | 0.467266 |
| feature-only | 0.2 | 55 | 81.220 | 81.747 | +0.527 | 5 | 2 | 0.516607 |
| feature-only | 0.3 | 55 | 80.580 | 80.163 | -0.417 | 1 | 4 | 0.485884 |
| feature-only | 0.4 | 46 | 77.693 | 77.392 | -0.302 | 9 | 12 | 0.404473 |
| feature-only | 0.5 | 87 | 76.364 | 76.694 | +0.329 | 2 | 0 | 0.449712 |
| feature-only | 0.6 | 54 | 73.748 | 73.182 | -0.566 | 9 | 13 | 0.370752 |
| feature-only | 0.7 | 88 | 70.166 | 69.619 | -0.548 | 1 | 5 | 0.317364 |
| label-aware | 0.0 | 42 | 88.281 | 87.816 | -0.464 | 0 | 3 | 0.420095 |
| label-aware | 0.1 | 38 | 84.486 | 84.486 | +0.000 | 1 | 1 | 0.580237 |
| label-aware | 0.2 | 48 | 82.900 | 82.482 | -0.419 | 3 | 6 | 0.425359 |
| label-aware | 0.3 | 38 | 81.763 | 81.926 | +0.163 | 2 | 1 | 0.560986 |
| label-aware | 0.4 | 28 | 74.820 | 78.274 | +3.454 | 38 | 21 | 0.465575 |
| label-aware | 0.5 | 38 | 76.397 | 75.990 | -0.407 | 4 | 7 | 0.531613 |
| label-aware | 0.6 | 32 | 73.647 | 74.142 | +0.495 | 10 | 7 | 0.500355 |
| label-aware | 0.7 | 36 | 69.092 | 68.926 | -0.165 | 6 | 7 | 0.457786 |

## Limits and stopping

- Single seed; eight per-rate Test-best checkpoints, not a single deployable checkpoint.
- Same checkpoint and exact mask per switch; OFF receives no new epoch selection.
- Binary ACC/WF1 exclude y=0 and use pred>0; MSE includes all686 valid targets.
- Corrections/harms are OFF→ON; summed counts across rates repeat targets under different masks.
- OFF−Flat reflects differences between separately trained main paths; it is not a causal training decomposition.
- ON−OFF is frozen branch sensitivity, not a separately trained ablation or proof of a training bottleneck.
- All outputs/weights remain frozen. No HPO, further seed, training or module modification automatically follows.
