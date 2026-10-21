# MOSI seed66: independent fixed-rate Flat vs cyclic Flat

INTERNAL TEST-ORACLE SCREEN

This is eight independently trained models, not one deployment checkpoint.
Each fixed-rate model trains and evaluates only its designated missing rate.
Per-rate TEST-best W-F1, earliest strict-best ties; original GCNet random-mask-v1.

| Rate | Epoch | Fixed ACC | Cyclic ACC | ΔACC | Fixed W-F1 | Cyclic W-F1 | ΔW-F1 | Fixed MSE | Cyclic MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 31 | 87.348 | 87.348 | +0.000 | 87.294 | 87.251 | +0.044 | 0.938107 | 1.018859 |
| 0.1 | 39 | 83.232 | 83.994 | -0.762 | 83.306 | 83.871 | -0.566 | 1.314530 | 1.204979 |
| 0.2 | 55 | 82.317 | 82.622 | -0.305 | 82.325 | 82.595 | -0.270 | 1.309044 | 1.342931 |
| 0.3 | 44 | 80.335 | 80.793 | -0.457 | 80.017 | 80.741 | -0.724 | 1.288260 | 1.306329 |
| 0.4 | 34 | 77.744 | 79.268 | -1.524 | 77.852 | 79.305 | -1.453 | 1.536127 | 1.534990 |
| 0.5 | 39 | 76.067 | 76.067 | +0.000 | 75.996 | 75.996 | +0.000 | 1.606810 | 1.579528 |
| 0.6 | 27 | 74.085 | 75.305 | -1.220 | 73.491 | 75.328 | -1.837 | 1.841454 | 1.828822 |
| 0.7 | 53 | 69.512 | 68.750 | +0.762 | 69.170 | 68.906 | +0.264 | 1.999723 | 2.141490 |

## Completed aggregates

| Scope | Fixed ACC | Cyclic ACC | Fixed W-F1 | Cyclic W-F1 | ΔW-F1 |
|---|---:|---:|---:|---:|---:|
| mean | 78.830 | 79.268 | 78.681 | 79.249 | -0.568 |
| high | 73.222 | 73.374 | 72.886 | 73.410 | -0.524 |

Only completed designated-rate results enter tables. A scope is aggregated only when all its rates finish.
The fixed-rate training distribution differs from cyclic; this is not a paired same-training-trajectory comparison.
Same seed/data/original task/architecture and original fixed evaluation mask are checked by the worker.
No multi-seed or independent generalization claim.

## State

- rate0.0: {'status': 'complete', 'epoch': 150, 'error': None}
- rate0.1: {'status': 'complete', 'epoch': 150, 'error': None}
- rate0.2: {'status': 'complete', 'epoch': 150, 'error': None}
- rate0.3: {'status': 'complete', 'epoch': 150, 'error': None}
- rate0.4: {'status': 'complete', 'epoch': 150, 'error': None}
- rate0.5: {'status': 'complete', 'epoch': 150, 'error': None}
- rate0.6: {'status': 'complete', 'epoch': 150, 'error': None}
- rate0.7: {'status': 'complete', 'epoch': 150, 'error': None}

## Completion and saved-output validation

All8 independent models completed150 epochs under source
`470e86066de9f5e0ef0d2926777cc2fe05a32ebe`; queue status complete.
No new training or inference was performed for this summary.

For all8 runs, CPU verification recomputed ACC/W-F1/MSE from saved selected
predictions; verified exact baseline labels, speakers, utterance IDs/order and
availability masks; verified selected checkpoint/prediction SHA256; checked
all150 epochs used only the designated rate with2 updates and no empty
availability; checked the original LR boundary at101; and verified earliest
strict TEST-best W-F1 selection. All8 full last checkpoints contain model,
Adam, RNG, identity,150-epoch history and matching selection state.
See `completed/VERIFICATION.json` and `completed/VERIFY_COMMAND.txt`.

These8 models use2400 optimizer updates in total versus300 for the single
cyclic trajectory; each model individually receives the same150epoch budget.
Fixed-rate mean W-F1 is78.681487 versus cyclic79.249087(-0.567599pp);
high-missing72.885633 versus73.409792(-0.524159pp). This seed does not support
an overall advantage for rate specialization, despite its greater aggregate
training budget. It does not establish statistical significance or a universal
failure of fixed-rate training. No additional seeds or configurations started.

Raw completed queue/report/CSV are preserved in `completed/`. All weights,
large prediction arrays and full logs remain in the original independent
remote execution directory; historical cyclic/V3/Nested records are untouched.

