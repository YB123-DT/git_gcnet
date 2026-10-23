# Three MOSI paired interfaces: completed seed66 results

INTERNAL TEST-ORACLE SCREEN

All six jobs completed 150/150 epochs without reported errors, verified on
2026-10-23. The final completed-only machine report, per-rate scores/selected
epochs and artifact existence check are in `completed/REPORT.json`,
`completed/PER_RATE.csv` and `completed/ARTIFACT_CHECK.json`.

| Model | Eight-rate W-F1 | Delta vs Flat (pp) | High-missing W-F1 |
|---|---:|---:|---:|
| Accepted Flat66 | 79.249 | 0.000 | 73.410 |
| 1A full source interaction | 79.139 | -0.110 | 73.713 |
| 1B diagonal source interaction | 78.617 | -0.632 | 72.976 |
| 2A updated Local in Adapter only | 79.323 | +0.074 | 73.551 |
| 2B updated Local in Adapter and Skip | 78.871 | -0.378 | 72.779 |
| 3A task increment before Norm | 78.860 | -0.389 | 73.156 |
| 3B task increment after Norm | 78.806 | -0.443 | 72.581 |

Within-pair B minus A W-F1 is -0.523 / -0.452 / -0.054 pp for pairs1/2/3;
high-missing differences are -0.737 / -0.772 / -0.575 pp. Thus none of the
three B-side alternatives improves W-F1 in this seed66 screen. The small
2A-over-Flat gain is not evidence of a stable multi-seed improvement.

Selection is per-rate Test-best; the mean can combine checkpoints from
different epochs and is not single-checkpoint deployment performance. No
significance claim is made from one seed. No additional inference/training
was run for this report. All eight BEST/prediction pairs, fixed masks and
full last state exist for each completed run. Original source, mask and
initialization identity checks were enforced by the immutable workers.

## Historical launch record (not current run state)

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
