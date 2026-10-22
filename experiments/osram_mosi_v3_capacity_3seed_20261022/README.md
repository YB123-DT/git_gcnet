# MOSI Flat512 + V3-128: supplement seeds67/68

INTERNAL TEST-ORACLE SCREEN. Add exactly two independent fresh runs; reuse
seed66 from the six-arm screen without repeating or modifying that run.

Selected configuration: Flat intermediate512/output1600,V3 width128,PairMLP256,
four V3 heads,total7,864,639 parameters. CHM8heads,key/value64,joint write,
write_step0.6,read-before-write,LocalSkip,original classifier and MSE unchanged.

MOSI archiveds0 environment,Torch2.2.2+cu121/NumPy1.24.3/sklearn1.3.2;official
features/splits,batch32,Adam,WD1e-5,original dropout.150epochs:3e-4 for1..100,
3e-5 for101..150. No additional stage,loss,method,HPO or dataset.

Only runtime seed changes;its training order/masks/RNG follow that seed's
completed original Flat reference. Test masks remain the accepted commonseed66
GCNet masks at each rate. No fixed training mask or missing-protocol change.
Per-rate Test-best WF1;ACC/MSE come from the same predictions and exact masks.
Keep8BEST+matched predictions+fixed_masks.pt+full last/model/Adam/RNG/history.

Reference candidate66:

```text
/data2/yb/remote_experiments/osram_mosi_v3_capacity_20261022/execution_v1/runs/f512_v128
```

Original Flat66/67/68 are reused read-only from the accepted3e-4/3e-5 runs.
Their eight-rate WF1 is79.2491/79.4024/79.1937(mean79.2817),high-missing
73.4098/73.7766/73.2281(mean73.4715). New67/68 each require own reference
initialization/config/train-order/mask/step/LR parity,not seed66 training traces.

The original capacity worker gained default-preserving injection for seeds,
config/reference/source catalog only;no model or task math changed. Existing
workers remain bound to source86e3900. Candidate reports compare actual core
model/sampler/task hashes across all three seeds even though entrypoint hashes
differ. Tiny schema evolution is normalized through current TrainConfig defaults.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
 -m experiments.osram_mosi_v3_capacity_3seed_20261022.run \
 --source-commit SOURCE_SHA \
 --gpu-uuid GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
 --output /data2/yb/remote_experiments/osram_mosi_v3_capacity_3seed_20261022/execution_v1
```

biggpu physicalGPU2,health/UUID checked. Two independent two-epoch/four-update
real-MOSI smokes precede measured concurrent admission of the two formal jobs.
Keep64GiB RAM/20GiB disk and1.5xGPU peak+2GB margin. No batch shrink or signals
to other tasks. tmux persistence and QUEUE.json preserve commands/PIDs/status.

RESULT.md/REPORT.json/PER_RATE.csv show completed-only per-seed paired results.
Three-seed mean/sampleSD appears only when all three candidates complete.
Partial two-seed means are not presented as three-seed results.

Candidate was selected usingseed66 test results;selection and Test-oracle bias
remain. The supplement does not add small-Flat-only67/68 controls,so cannot prove
V3-specific structural gains against a matched small-Flat three-seed control.
All existing weights/masks/results and ongoing MOSEI experiments stay untouched.
