# Observed startup — 2026-10-15 08:01 UTC

INTERNAL TEST-ORACLE SCREEN. Five formal workers launched; training not completed.

* Immutable source: `9fa9f571e7095d5b09f9c256d888c31a76c82744`.
* Server: biggpu; Python `/data2/yb/conda_envs/cmerc-repro/bin/python`.
* Source worktree: `/home/qinjinghui/remote_experiments/osram_meld_b6_threshold_code_9fa9f57`.
* Output: `/home/qinjinghui/remote_experiments/osram_meld_b6_threshold_hpo_20261015`.
* Persistent controller PID4050321; controller.log,queue/QUEUE.json,STRUCTURE.sqlite3.
* Latest resource scope: **physicalGPU2only**, UUID
  `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`, at most6total compute processes,
  counting external jobs. No experiments on GPU0/1/3/4/5/6/7were started by this task.
* Existing task processes and old checkpoints/results remain untouched.

Command from immutable source:

```bash
nohup /data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_meld_b6_threshold_hpo_20261015.run \
  --output /home/qinjinghui/remote_experiments/osram_meld_b6_threshold_hpo_20261015 \
  --gpus 2 --per-gpu 6 --trials 20 \
  > /home/qinjinghui/remote_experiments/osram_meld_b6_threshold_hpo_20261015/controller.log 2>&1 < /dev/null &
```

## Initial configurations and observed PIDs

All have rank2/steps14,seed66,50epochs,batch32,original mainLR/wd/dropout/masks.
At08:01:43UTCtheir worker summaries were running without error; no full epoch
had yet been written. This is not an experimental performance report.

| Trial | Threshold init | B6LR multiplier | Effective B6LR | Worker PID |
|---:|---:|---:|---:|---:|
| 0 | 0.1 | 1 | 0.000343323454 | 4052358 |
| 1 | 0.03 | 1 | 0.000343323454 | 4052381 |
| 2 | 0.3 | 1 | 0.000343323454 | 4052518 |
| 3 | 0.1 | 0.1 | 0.000034332345 | 4052528 |
| 4 | 0.1 | 10 | 0.003433234536 | 4052536 |

After these5complete, the controller automatically proposes the remaining15
configurations using completed feedback, keeping a bounded6pending window and
replenishing as jobs finish. Total budget20,not25. A whole-controller lock and
queue process-identity recovery prevent duplicate submission. No additional
seed,dataset,rank or depth search is authorized.

## Verified implementation and real batch

Local bounded regressions:57passed,4Optuna-only skipped. Complete same suite
on remote Python:61passed. Covers default checkpoint/forward/gradient behavior,
threshold routing,optimizer group membership/default topology/restoration,
GCNetmask-related contracts,real SQLite HPO/restart with5RUNNING initialization
barrier,explicit GPU2CLI and default4versus explicitly admitted6process limits.
Known optional torch-sparse/libcusparse11warning is unrelated to this path.

Real longest-train-conversation batch smoke (threshold1,multiplier30):3optimizer
steps complete; peak allocated1004.024MiB,elapsed9.741seconds. CE values
1.956717,3.511934,2.311216; gradient norms10.734012,21.445940,14.221547finite.
Nonzero B6gradient tensor counts0,1,3; all3B6parameter tensors actually changed.
The first0is expected from the original zero adapter output,not evidence of
successful gradient flow by itself; the subsequent steps establish it.

Measured optimizer provenance:

| Group | Learning rate | Parameter count |
|---|---:|---:|
| backbone excluding B6 | 0.0003433234536467373 | 12580960 |
| projector | 0.0003433234536467373 | 698628 |
| classifier | 0.0003433234536467373 | 11207 |
| B6core | 0.01029970360940212 | 29 |

Thus30xdoes not accidentally apply to CHM,Flat,encoder or classifier. Smoke is
implementation evidence only. `SMOKE.json`,`PREFLIGHT.json`,`LAUNCH_QUEUE.json`
preserve full observed settings and UUID-bound worker identities.

## Timing caveat

Previous3card20trial study took4h14m26s overall,with individual training46–140min
(median82),and replay/loading/statistics20–101s(median44). Its warmstart barrier
and shared-GPU contention contributed to wall time; it was not20inference calls.
Current GPU2-only run uses rough5–8hour planning guidance,not measured completion
time. At preflight GPU2already had a~1394MiBexisting task; other cards also had
large unrelated jobs. Six-way sharing can extend single-run latency even when
memory fits. Do not promise a6/4speedup; recalibrate from observed epochs.
