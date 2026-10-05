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
