# Locked top-three seed confirmation

INTERNAL DIAGNOSTIC ONLY

Selected before confirmation from completed C01–C20 methods, ranking seed66
8-rate BEST Test-oracle W-F1; controls excluded. Unfinished methods excluded.

| Method | seed66 8-rate (%) | high (.5/.6/.7) (%) |
|---|---:|---:|
| C20 CAGrad three-exit supervision |80.645543|75.883865|
| C07 Recurrent Independent Mechanisms |80.513719|76.256848|
| C08 TTT-MLP |80.089774|75.200594|
| Original Flat |81.068095|76.352251|

Reuse completed seed66; six authorized new runs = each method × seeds67/68.
From scratch, same-seed original Flat reference config. Those reference configs
were checked to differ only in seed. 100 epochs, all method-specific coefficients,
task objectives, network dimensions, Adam/lr/batch and missing/evaluation protocol
remain fixed. No new hyperparameter sweep or GPU smoke. No new control training.

Deployment must copy original immutable source_00522bd; only run.py and
multiseed.py differ. Record original_model_commit and wrapper commit in SNAPSHOT.
GPU6 only. C20/C07 admitted gradually using real peaks +20%+512MiB and 2GiB
headroom (3GiB initial reservation). C08 runs singly after the original batch
queue and the confirmation light runs drain, with 18GiB admission reservation.
No original process is stopped; failed jobs are recorded, not blindly relaunched.

Status: deployed; persistent GPU6 dispatcher running, PID3701922.
First admitted job C20 seed67 PID3708408 at 2026-10-05T04:50:40.039271+00:00.
Other jobs automatically fill measured capacity; C08 remains heavy/exclusive.
Source wrapper commit e69f2c1; original model commit00522bd. Snapshot seal verified
all source hashes, with only the two orchestration scripts differing.
Remote root /data2/yb/remote_experiments/osram_core20_top3_3seed_20261005;
tmux osram_core20_top3_3seed_20261005; live runs/DISPATCH.json and dispatcher.log.
Each run is under runs/{method}/seed_{seed}/ with train.log and PROVENANCE.json.

Launch within the immutable code snapshot:

```bash
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
-m experiments.osram_core20_20261005.multiseed \
--root /data2/yb/remote_experiments/osram_core20_top3_3seed_20261005/runs \
--reference-root /data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full \
--data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json \
--original-runs /data2/yb/remote_experiments/osram_core20_20261005/runs
```

Final analysis should show
all three seed-matched deltas vs Flat, mean/sample SD, eight per-rate means and
high-missing means. Do not replace missing runs with best single-seed scores.
No significance or formal validation-selected performance claim.

## Partial update 2026-10-05 06:04 UTC

| Method | seed | Status | 8-rate W-F1 (%) | High (%) |
|---|---:|---|---:|---:|
| C07 |66|complete, reused|80.514|76.257|
| C07 |67|complete, 100 epochs|80.059|75.720|
| C07 |68|complete, 100 epochs|79.063|73.977|
| C07 |3-seed mean|complete|79.879|75.318|
| Flat |3-seed matched mean|reference|80.559|75.594|
| C20 |68|complete, 100 epochs|79.821|74.597|
| C20 |67|failed after 29 recorded epochs|—|—|
| C08 |67|pending|—|—|
| C08 |68|pending|—|—|

C20 seed67 failed with RuntimeError: CAGrad simplex solve failed:
Positive directional derivative for linesearch. No completed three-seed mean
may be reported for C20. Preserve its full last_training checkpoint; do not
blindly retry unchanged solver or replace the failed run with a best partial score.
Solver failure has not yet been diagnosed/repaired in this status update.
C08 waits for the original queue: C19 still running, 22 recorded epochs.
Live dispatcher snapshot archived in TOP3_PROGRESS_20261005.json.
