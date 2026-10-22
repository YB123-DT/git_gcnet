# MOSI head48: completed three-seed comparison

INTERNAL TEST-ORACLE SCREEN

All three seeds completed 150 epochs. Seed66 reuses the completed head48 grid run;
seeds67/68 are the requested new runs. Head count remains eight; K/V width changes
64 to 48. Write step remains 0.6. No new module or loss is enabled.

| Seed | Original head64 W-F1 | Head48 W-F1 | Difference (pp) |
|---:|---:|---:|---:|
| 66 | 79.249 | 79.497 | +0.248 |
| 67 | 79.402 | 79.085 | -0.317 |
| 68 | 79.194 | 79.297 | +0.103 |
| Mean | 79.282 | 79.293 | +0.011 |

High-missing (.5/.6/.7) mean W-F1: 73.471 to 73.510 (+0.039 pp).
The average is effectively unchanged, and improvement is not consistent across
seeds. Parameter count decreases from 13,509,793 to 11,245,857 (about 16.8%).
Context width and Flat input width also decrease; this is not a parameter-matched
memory-storage-only ablation.

Protocol remains GCNet random-mask-v1, cyclic rates 0.0–0.7, Adam/batch32,
weight decay 1e-5, MSE, LR3e-4 for epochs1–100 and 3e-5 for epochs101–150.
Scores average per-rate Test-best checkpoints, not one deployment checkpoint.
All paired comparisons use the original exact test masks and same-seed controls.
The inactive-module source bridge is documented in CORE_BRIDGE.json.

Full per-rate scores, selected epochs, sample SD, and completion details:
[completed/RESULT.md](completed/RESULT.md), [completed/PER_RATE.csv](completed/PER_RATE.csv),
[completed/QUEUE.json](completed/QUEUE.json).

Full provenance REPORT.json remains on biggpu at:
`/data2/yb/remote_experiments/osram_mosi_head48_3seed_20261022/execution_v1/REPORT.json`.
It is not copied to Git because its detailed mask provenance is large (7.4 MB).
Training snapshots and all checkpoint/prediction files remain unchanged remotely.

The separate head48/write_step0.4 seed66 experiment is still running and is not
included in the completed three-seed comparison above.
