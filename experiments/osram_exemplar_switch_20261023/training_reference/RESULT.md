# MOSI66 TRAIN-exemplar transfer

INTERNAL TEST-ORACLE SCREEN

Only completed full training runs enter scores; smoke is not performance evidence.
All eight rates use exact reference test masks. Each rate selects its own TEST-best W-F1 checkpoint;
the eight-rate mean is not one deployable checkpoint. This is a seed66 exploratory comparison.

| Model | Scope | ACC | W-F1 | MSE |
|---|---|---:|---:|---:|
| Flat | mean | 79.268 | 79.249 | 1.494741 |
| Flat | high | 73.374 | 73.410 | 1.849947 |
| feature-only | mean | 78.754 | 78.644 | 1.507411 |
| feature-only | high | 73.272 | 73.165 | 1.823484 |
| label-aware | mean | 79.345 | 79.255 | 1.468161 |
| label-aware | high | 73.069 | 73.019 | 1.782892 |

## Per rate

| Model | Missing | Best epoch | ACC | W-F1 | MSE |
|---|---:|---:|---:|---:|---:|
| Flat | 0.0 | 48 | 87.348 | 87.251 | 1.018859 |
| Flat | 0.1 | 52 | 83.994 | 83.871 | 1.204979 |
| Flat | 0.2 | 49 | 82.622 | 82.595 | 1.342931 |
| Flat | 0.3 | 50 | 80.793 | 80.741 | 1.306329 |
| Flat | 0.4 | 62 | 79.268 | 79.305 | 1.534990 |
| Flat | 0.5 | 48 | 76.067 | 75.996 | 1.579528 |
| Flat | 0.6 | 48 | 75.305 | 75.328 | 1.828822 |
| Flat | 0.7 | 51 | 68.750 | 68.906 | 2.141490 |
| feature-only | 0.0 | 37 | 86.890 | 86.896 | 1.177396 |
| feature-only | 0.1 | 83 | 83.384 | 83.463 | 1.291697 |
| feature-only | 0.2 | 55 | 82.012 | 81.747 | 1.311742 |
| feature-only | 0.3 | 55 | 80.488 | 80.163 | 1.303792 |
| feature-only | 0.4 | 46 | 77.439 | 77.392 | 1.504207 |
| feature-only | 0.5 | 87 | 76.982 | 76.694 | 1.562596 |
| feature-only | 0.6 | 54 | 73.018 | 73.182 | 1.923381 |
| feature-only | 0.7 | 88 | 69.817 | 69.619 | 1.984476 |
| label-aware | 0.0 | 42 | 87.805 | 87.816 | 1.049194 |
| label-aware | 0.1 | 38 | 84.604 | 84.486 | 1.181164 |
| label-aware | 0.2 | 48 | 82.470 | 82.482 | 1.355293 |
| label-aware | 0.3 | 38 | 82.165 | 81.926 | 1.302513 |
| label-aware | 0.4 | 28 | 78.506 | 78.274 | 1.508452 |
| label-aware | 0.5 | 38 | 76.372 | 75.990 | 1.537859 |
| label-aware | 0.6 | 32 | 74.085 | 74.142 | 1.806757 |
| label-aware | 0.7 | 36 | 68.750 | 68.926 | 2.004059 |

## Scope and interpretation

- Feature-only and label-aware use identical retrieval structure and nominal parameter count.
- Feature-only supplies zero instead of neighbor labels; its label encoder weight has no task gradient.
- TRAIN bank excludes the entire query conversation during training. No validation/test labels enter retrieval.
- Inference requires the TRAIN feature/label bank. Its storage/runtime are not represented by parameter count.
- CHM scan, original Local Skip, Flat and task head are preserved. Only original sentiment MSE trains the model.
- Added behavior is a migration hypothesis, not evidence of a previously localized OSRAM bottleneck.
- F2 must exceed Flat and feature-only before discussing follow-up; no automatic seeds/HPO/dataset expansion.

## Run status

- feature-only: {'status': 'complete', 'epoch': 150, 'error': None}
- label-aware: {'status': 'complete', 'epoch': 150, 'error': None}
