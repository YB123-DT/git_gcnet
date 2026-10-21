# MOSEI current-missing Flat, requested seeds67/68 at1e-4

INTERNAL TEST-ORACLE SCREEN

User context: asked whether1e-4 had been run,then requested a new-Missing MOSEI
rerun,seeds67/68. Finite interpretation announced before execution: exactlytwo
fresh original Flat runs,constant1e-4,60epochs,batch32,Adam,wd1e-5. No new
seed66,source-role block,Nested,Router,additional dataset or HPO is included.
Original no-JEPA cfg84 Flat1600+LocalSkip,CHM jointwrite.6,8heads×64 unchanged.

Reuse current MOSEI setup/data/runtime,not the historical low-LR snapshot.
Original gcnet-random-mask-v1 cyclic0–.7;officialsplit and isolated commonGCNet
testbank match the completed current-source direct-model references perseed.
Fresh original initialization,trainseed and mask/update traces are checked
against those references; only architecture removal and LR differ. Never
initialize from trained reference weights. This1e-4Flat is NOT a same-LR
comparator for the previously trained1e-3 direct model; label differences.

Select strict per-rate TEST W-F1-best epochs,earliestties. ACC/MSE,predictions,
labels and exactMask from the samecall. Retain eightBEST and full model/Adam/
RNG/history last.pt. No validation selection or extra training holdout.

biggpu physicalGPU1 UUID GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,two parallel
jobs,healthyUUID checked,GPU4forbidden. Runtime remains
`/data2/yb/reproduction_workspace/envs/s0/bin/python` (Torch2.2.2+cu121,
NumPy1.24.3,sklearn1.3.2). Data manifest and allfeatures are SHA-verified by
the original setup. Reuse already-verified identical Flat numerical path;
no additional standalone GPUsmoke or extra training job. Validate actual
constantLR and finite normal training on the first formalepoch.

Run under persistenttmux from an immutable committed snapshot:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_mosei_gcnet_flat_lr1e4_20261021.run \
 --source-commit SOURCE_SHA \
 --output /data2/yb/remote_experiments/osram_mosei_gcnet_flat_lr1e4_20261021/execution_v1 \
 --gpu-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049
```

Fresh isolated outputs,no peer kills,protected results untouched. Explicit
same-source --resume uses fullstate,refuses live duplicatePIDs/completedjobs.
Final RESULT.json/RESULT.md generated only after both60epoch trajectories
complete. Per-rate Test-oracle means are not a single deploymentcheckpoint.
