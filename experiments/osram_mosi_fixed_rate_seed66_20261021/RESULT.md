# MOSI seed66 independent fixed-rate Flat

INTERNAL TEST-ORACLE SCREEN

Status: **finite eight-job queue started**,2026-10-21 14:44UTC snapshot.

| Fixed training/test rate | Status | Completed epochs at observation |
|---:|---|---:|
| 0.0 | running | 9/150 |
| 0.1 | running | 4/150 |
| 0.2 | running | 1/150 |
| 0.3 | running | 1/150 |
| 0.4 | queued | 0/150 |
| 0.5 | queued | 0/150 |
| 0.6 | queued | 0/150 |
| 0.7 | queued | 0/150 |

All four live workers report `train_rate_mode=fixed`, the designated single
`train_missing_rates`, original `gcnet-random-mask-v1`, V3 disabled and
13,509,793 original Flat parameters. Exact designated-rate test masks match
the accepted same-seed cyclic Flat. No completed performance is claimed yet.

Server: biggpu, s0 Python; physicalGPU2
`GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`, logicalcuda:0. Maximum4 parallel,
remaining4 automatically admitted as slots open without changing any protocol.
Persistent controller4096524; worker PIDs4096603/4098951/4100060/4101242.
Tmux socket `osram_fixed66`, session `mosi66_470e860`.

Immutable source `470e86066de9f5e0ef0d2926777cc2fe05a32ebe`,9641 tracked Git
blobs verified in `source_470e860`. Remote output:
`/data2/yb/remote_experiments/osram_mosi_fixed_rate_seed66_20261021/execution_v1`.
See `LAUNCH.json` for exact commands, worker/config/mask evidence and pending
rates. Existing V3/Nested experiments remain untouched.

23 worker/queue tests passed. Real fixed0.7 GPU smoke verified two original
optimizer updates, one-hot masks, checkpoint/RNG/prediction restoration and
same original initialization. All28 smoked source hashes matched final code
before commit/deployment. See `SMOKE.json` and `VALIDATION.md`.

The queue automatically writes remote `RESULT.md`, `REPORT.json`, and
`METRICS.csv` from completed runs, with per-rate ACC/W-F1/MSE and cyclic
differences. Best/last, matching predictions/Mask and histories are retained.
This is eight independently trained models, not one deployment checkpoint;
the total training budget spans eight trajectories, not a single cyclic run.
