# Observed startup

INTERNAL TEST-ORACLE THREE-SEED REPLICATION. Training is running, not completed.

All seven new jobs actually launched on biggpu and their real CUDA PID→UUID
bindings were verified. Persistent controller PID2157874. Standalone launcher
source commit0193f34; original immutable numerical training source9fa9f57.
Exact full source SHAs and source66summaries/configs appear in MANIFEST.json.

| New job | Physical GPU | PID |
|---|---:|---:|
| Flat_matched_seed66 | 2 | 2158352 |
| Flat_matched_seed67 | 3 | 2158513 |
| Flat_matched_seed68 | 5 | 2158642 |
| B6_default_seed67 | 2 | 2158847 |
| B6_default_seed68 | 2 | 2158871 |
| B6_tuned_seed67 | 2 | 2158890 |
| B6_tuned_seed68 | 2 | 2159167 |

GPU2UUID:GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4.
GPU3UUID:GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a.
GPU5UUID:GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62.
No new job was placed onGPU0/1/4/6/7; other users and existing studies untouched.

Observed first epoch: Flat67CE1.4625/Flat68CE1.4675,finite; the fiveGPU2jobs
were initialized with real CUDA allocations and first epoch in progress. No
Traceback or CUDA OOM was present in any of the seven startup logs. These are
execution checks,not experimental performance claims or final selected scores.

Bounded local regression:14passed,1Optuna-only skipped,known unused
torch-sparse/libcusparse warning. Independent old/new default compatibility
audit:14related tests passed and actual old/new rank2/14decomposition forward
and gradients exactly equal. Real remote source,dataset,config,disk and GPU
preflight passed. Source and masks were not changed for this task.

Remote output:
`/home/qinjinghui/remote_experiments/osram_meld_b6_top2_matched_flat_3seed_20261015`.
Logs:`controller.log`, `queue/logs/`; runtime:`RUNNER.json`, `queue/QUEUE.json`.
Checkpoint/prediction artifacts stay remote under `runs/`.
On terminal completion the controller writes RESULT.md and SUMMARY.json/CSV.
The committed LAUNCH_QUEUE is a dated startup snapshot,not ongoing live state.
