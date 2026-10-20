# MOSI direct-source three-seed follow-up running

INTERNAL TEST-ORACLE SCREEN

Verified at 2026-10-20T13:12:26.067774+00:00.
Source a55b2cf2a3db04962d87baa789ce272b3e3eb26a; original numerical model/sampler code unchanged.
Server biggpu, healthy physicalGPU0 / GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45.
Controller PID2726231; tmux osram_source_role_direct_3seed:mosi_direct3_a55b2cf.

| Seed | State | Completed epoch | Worker PID |
|---|---|---:|---:|
|66|Existing complete run reused|150|—|
|67|running|1/150|2729543|
|68|running|1/150|2729639|

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
Three-seed mean/std are PENDING67/68 completion, not filled using partial scores.

Outputs: /data2/yb/remote_experiments/osram_mosi_source_role_direct_3seed_20261020/execution_v1.
Queue will automatically publish24rate/seed rows, individual paired Flat comparisons,
across-seed mean/sample std and overall/high after both jobs finish. No more runs
will be added automatically. All8BEST and full last.pt per newseed retained remotely.
LAUNCH.json records effective complete67config;68differs only in seed as verified.
Model weights,feature data and large predictions are not committed.
