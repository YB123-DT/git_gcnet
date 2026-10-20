# MOSI direct-source three-seed follow-up complete

INTERNAL TEST-ORACLE SCREEN

Completion verified at 2026-10-20T14:16:10.233816+00:00.
All three seeds completed150epochs; automatic queue/report succeeded.
See completed/RESULT.md,METRICS.csv,SUMMARY.json,FINAL_VERIFICATION.json.
PROGRESS.md/PROGRESS.json remain the historical37/34epoch snapshot only.
Source a55b2cf2a3db04962d87baa789ce272b3e3eb26a; original numerical model/sampler code unchanged.
Server biggpu, healthy physicalGPU0 / GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45.
Controller PID2726231; tmux osram_source_role_direct_3seed:mosi_direct3_a55b2cf.

| Seed | State | Completed epoch | Worker PID |
|---|---|---:|---:|
|66|Existing complete run reused|150|—|
|67|complete|150/150|2729543|
|68|complete|150/150|2729639|

Final eight-rate W-F1:seed67=78.578491,seed68=78.471468.
Three-seed W-F1=78.833183±.536500 vs Flat78.890919±.206465(-.057736pp).
High W-F1=72.620078±.754093 vs Flat72.907882±.374122(-.287804pp).
Std is sample std across seeds; rates averaged within each seed first.
Seed66 improvement did not replicate in67/68. No stable overall/high gain.
Both latest gradient hooks show14connected/14nonzero new parameter tensors.

Both new formal runs passed same-seed Flat initialization/config/split/mask startup
gates and first-epoch training-mask/LR/update parity. First-epoch new-module gradients
are14connected/2nonzero due zero final initialization, not proof all upstream parameters
already have nonzero updates. Subsequent coverage remains logged. No extra formal66 run.

Short real MOSI startup smoke completed; full optimizer/model/RNG and selected prediction
reload passed. No smoke score is used. Each new run uses original s0,MSE,Adam,batch32,
wd1e-5,LR1e-3 for100then1e-4 for50,GCNet-random-mask-v1,per-rate earliestTest-best W-F1.
No architecture,feature,loss,missing or extra hyperparameter search was introduced.

Existing66 final:ACC79.458841,W-F1 79.449590; high ACC73.424797,W-F1 73.477948.
Compared with same-seed Flat,W-F1+.321564pp overall/+.139839pp high.
All24selected NPZ metrics independently recomputed with max error0; exact
labels/IDs/speakers/masks match their same-seed Flat cohort. All450training
mask/update/LR pairings and earliest per-rate Test-best selection passed.
All24BEST files and three full last.pt states remain present remotely.

Outputs: /data2/yb/remote_experiments/osram_mosi_source_role_direct_3seed_20261020/execution_v1.
Queue published24rate/seed rows,individual paired Flat comparisons and across-seed
mean/sample std. The finite task is complete; no extra runs were started.
LAUNCH.json records effective complete67config;68differs only in seed as verified.
Model weights,feature data and large predictions are not committed.
