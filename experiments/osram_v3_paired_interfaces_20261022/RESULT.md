# Three MOSI paired interfaces: launch status

INTERNAL TEST-ORACLE SCREEN

Observed 2026-10-22T23:46:05.666136+00:00. All six authorized seed66 formal workers are training;
all six four-update real-data CUDA smokes passed. No final performance is available.

| Variant | Epoch | Physical GPU | PID | State |
|---|---:|---:|---:|---|
| topology-full | 10/150 | 3 | 4081886 | running |
| topology-diagonal | 9/150 | 2 | 4083151 | running |
| local-adapter | 8/150 | 3 | 4084297 | running |
| local-shared | 6/150 | 2 | 4085451 | running |
| task-pre-norm | 5/150 | 2 | 4086697 | running |
| task-post-norm | 4/150 | 3 | 4087891 | running |

Pair1: topology-full vs topology-diagonal. Pair2: local-adapter vs local-shared.
Pair3: task-pre-norm vs task-post-norm. Each pair has identical initial tensors/RNG
and nominal parameter count, verified on real-data setup. Diagonal source Q/K has
structural zero task derivatives; this is documented rather than misreported as
an unused-branch failure. Original model parameters remain jointly trainable.

MOSI150: epochs1–100 LR3e-4, epochs101–150 LR3e-5; batch32,Adam,WD1e-5,MSE,
Flat1600 and original GCNet cyclic random missing0.0–0.7. Same accepted Flat66
exact evaluation masks; every formal row checks mask trace/update/LR parity.
Eight per-rate Test-best W-F1 checkpoints, matching predictions/masks and complete
last optimizer/RNG state are retained. No extra seeds, HPO, datasets or losses.

Runtime source: `2ff097765b108c7aca40293e845a87cdc9fa63c4`.
Remote output: `/data2/yb/remote_experiments/osram_v3_paired_interfaces_20261022/execution_v1`.
Tmux socket/session: `osram_v3_pairs` / `mosi_pairs`.

The controller automatically produces completed-run REPORT.json, PER_RATE.csv and
RESULT.md, with per-rate and mean/high B-minus-A differences. Old Flat results are
reused without retraining. All first formal updates passed the gradient gate;
LAUNCH.json contains exact source/data hashes, live command verification and
per-pair initialization evidence. Existing experiments/checkpoints were untouched.
