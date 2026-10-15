# Observed launch — 2026-10-15 02:59 UTC

INTERNAL TEST-ORACLE SCREEN. Training is running, not completed.

## Immutable execution

* Source: `9048ffdcbeb30006c49e45c0d1387cada4d85242`.
* Branch: `feature/ecerc-features`, GitHub `YB123-DT/git_gcnet`.
* Server: `ssh biggpu`; Python `/data2/yb/conda_envs/cmerc-repro/bin/python`.
* Source worktree: `/home/qinjinghui/remote_experiments/osram_meld_b6_structure_code_9048ffd`.
* Output: `/home/qinjinghui/remote_experiments/osram_meld_b6_structure_hpo_20261015`.
* Persistent controller PID: `3460407`; stdout: `controller.log`.
* Old screen/HPO/replication source worktrees and existing jobs were not modified,
  paused or killed. Protected baseline weights were not touched.

Command from the immutable source worktree:

```bash
nohup /data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_meld_b6_structure_hpo_20261015.run \
  --output /home/qinjinghui/remote_experiments/osram_meld_b6_structure_hpo_20261015 \
  --gpus 0,2,3 --per-gpu 4 --trials 20 \
  > /home/qinjinghui/remote_experiments/osram_meld_b6_structure_hpo_20261015/controller.log 2>&1 < /dev/null &
```

## Observed initial parallel jobs

| Trial | Rank | Steps | Host GPU | Worker PID | Completed epochs at observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 14 | 3 | 3461027 | 1/50 |
| 1 | 1 | 8 | 0 | 3461040 | 1/50 |
| 2 | 1 | 24 | 2 | 3461047 | 1/50 |
| 3 | 2 | 14 | 3 | 3461128 | 1/50 |
| 4 | 3 | 14 | 0 | 3461149 | 0/50 |

All5worker summaries were running without an error. After these5fixed initial
configurations complete, the finite TPE study admits additional distinct
configurations from completed feedback, with at most12pending/running trials.
Every launch checks the four-total-process/GPU limit, counting external jobs
and its own not-yet-visible reservations. Actual concurrency can be lower when
existing jobs, memory reserve, or disk capacity prevent admission.

Healthy GPU UUID mapping:

* GPU0: `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`.
* GPU2: `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`.
* GPU3: `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`.

## Verification

Local model/config/mask/queue/replication regression:47passed,3skipped because
Optuna is not installed locally. Corrected remote tests, including real SQLite
proposal/resume integration:29passed. The integration uses synthetic unit-test
scores; it is not20completed MELD experiments. A known optional torch-sparse
import warning is present; neither CHM nor B6 uses that extension.

The first remote test caught duplicate proposals because installed Optuna5's
default multivariate sampler bypassed the independent uniqueness guard. The
flag is now explicitly `multivariate=False`; the regression verified20unique
proposals plus idempotent restart. No formal training ran under that old source.

Real MELD worst-length train-batch smoke used rank3/steps28, padded length24,
three optimizer steps, peak allocated1470.322MiB (~1.436GiB), elapsed8.302seconds.
CE:1.95435,3.50519,2.32138; finite gradient norms10.7796,21.4542,14.0856.
Nonzero B6 gradient tensor counts:0,1,3. The initial zero count is expected while
the original Flat final projection is zero; steps2/3 establish actual learning.
All3B6parameter tensors changed. B6 has57scalar parameters at28steps, compared
with29at14steps; rank changes factor computation, not parameter-vector length.
This shared-GPU smoke establishes implementation, not performance or throughput.

`PREFLIGHT.json`, `SMOKE.json` and `LAUNCH_QUEUE.json` preserve hashes/config,
measured GPU check and initial queue observations. Future scores belong in the
remote generated `RESULT.md`, `RESULT.csv`, `SUMMARY.json`, `TRIALS.json`, and
`STRUCTURE.sqlite3`; do not invent performance before those trials complete.
