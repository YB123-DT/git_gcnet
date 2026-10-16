# GPU3 launch evidence

INTERNAL TEST-ORACLE SCREEN. Four additional runs are running, not completed.

- Server: `ssh biggpu`; physical GPU3, UUID
  `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a` (V10032GB).
- Controller source: `005616d848f98a0011a3e2670847aa104397b5c9` at
  `/home/qinjinghui/remote_experiments/osram_meld_b3_gpu3_controller_005616d`.
- Unchanged numerical worker source: `737fdb16e5b4f4ee7a426a0979efbc7f59c5ed8c`
  at `/home/qinjinghui/remote_experiments/osram_meld_b3_code_737fdb1`.
- Output: `/home/qinjinghui/remote_experiments/osram_meld_b3_gpu3_extension_20261016`.
- tmux session: `meld_b3_gpu3_add4_1016`; controller PID: `1409739`.
- Environment: `/data2/yb/conda_envs/cmerc-repro/bin/python`;
  Torch1.13.1+cu117, Optuna5.0.0.

The new SQLite study imports fifteen completed original results as reference-only
TPE observations. All twenty original configurations, including five still
running, are excluded from new proposals. Four jobs were proposed immediately
using TPE, without another warm-start batch:

| New job | Bottleneck tokens | Rounds | Worker PID | Physical GPU |
|---|---:|---:|---:|---:|
| b3_gpu3_add_01 | 7 | 4 | 1411899 | 3 |
| b3_gpu3_add_02 | 6 | 2 | 1411919 | 3 |
| b3_gpu3_add_03 | 1 | 3 | 1411929 | 3 |
| b3_gpu3_add_04 | 3 | 4 | 1412068 | 3 |

All four queue records were verified concurrently running. The original GPU2
controller/SQLite/queue remain under their original owner and twenty-run budget.
The separate extension adds exactly four, for twenty-four combined training runs.

Every new job keeps MELD seed66,50epochs,batch32,lr0.001,WD1e-5,dropout0.5,
Flat1600 with original Local Skip, unchanged CHM and `gcnet-random-mask-v1`.
Objective remains0.6*paired_delta_ACC+0.4*paired_delta_UA; Test-best results are
saved separately with matching selected predictions/masks and corresponding UA.

## Checks

- Local tests:3 passed,1 skipped only for unavailable local Optuna.
- Remote actual Optuna/SQLite tests:4 passed, no skips, including four-proposal
  uniqueness, partial ask recovery, repeat invocation, frozen evidence hashing,
  and reservation of a source queue ahead of its exported trial list.
- Syntax compilation and whitespace checks passed.
- Numerical source and dataset/common-mask hashes verified before launch.
- GPU3 was healthy and idle with32495MiB free; disk had148GiB free at preflight.
- Reused same-source largest-structure (8tokens/4rounds) real-batch smoke:
  peak1886.161MiB, finite task gradients and actual Block2 updates. No additional
  numerical GPU smoke or architectural changes were needed for this controller.

Checkpoint retention, on/off replay, source/config hashes and result files use
the same original worker. No performance improvement is inferred from launch.
