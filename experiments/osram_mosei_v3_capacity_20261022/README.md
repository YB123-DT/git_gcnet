# MOSEI: smaller Flat, wider V3

INTERNAL TEST-ORACLE SCREEN. Exactly two fresh seed66 runs are authorized here.
No multi-seed expansion, HPO, checkpoint deletion or change to running queues.

| Arm | Flat intermediate | Flat output | V3 width | PairMLP hidden | Expected total parameters |
|---|---:|---:|---:|---:|---:|
|small_flat|1024|1600|off|off|10,080,865|
|wide_v3|1024|1600|256|512|12,397,503|

Recorded full model counts from GPU smoke are authoritative. Original Flat1600
has13,509,793 parameters; original Flat1600+V3 has13,844,799. Flat Adapter alone
shrinks9,535,104->6,106,176 parameters. V3 grows335,006->2,316,638 parameters.
The new combined model is smaller overall than the old large-Flat+V3 model.

Only intermediate widths change. Four attention heads, four read-role decoders,
joint source tracking, pairwise-before-aggregation hierarchy, zero-init outputs,
Local Skip, output1600 and task head remain. CHM8heads/key64/value64, joint
Block Delta Write,read-before-write,step0.6,Base/Gap Query and Missing stay fixed.
All parameters train together from scratch; no pretrained checkpoint is loaded.

## Protocol

MOSEI archived s0 environment:
`/data2/yb/reproduction_workspace/envs/s0/bin/python`
Torch2.2.2+cu121,NumPy1.24.3,sklearn1.3.2. Same accepted official feature/label
files and splits are SHA256-verified by the original setup. Adam,constant1e-3,
WD1e-5,batch32,60epochs,original dropout,emotion-only MSE,cyclic0..0.7,
gcnet-random-mask-v1. Same accepted seed66 test-mask bank; every training epoch
must match the original mask/order hash, optimizer-step count and LR trace.

Per-rate TEST-best W-F1 selection, strict improvement/earliest tie; ACC/MSE
come from the same saved predictions/mask. No extra inference for final metrics.
Eight selected weights may come from different epochs, not one deployed model.
Preserve8 BEST, same-call predictions, fixed_masks.pt, history and complete
last.pt (model,Adam,RNG,epoch,config/identity). Smoke weights are also retained.

## Run

Use one committed immutable source snapshot on biggpu. GPU1 was checked healthy
and free; UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`. Never use GPU4.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_mosei_v3_capacity_20261022.run \
  --source-commit SOURCE_SHA \
  --gpu-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
  --output /data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/execution_v1
```

The controller runs two independent three-real-batch GPU smokes first. It checks
zero-init equality to same-width Flat, finite opened V3 gradients, original
parameter updates,one scan per train batch,model/Adam/RNG/prediction reload.
Only successful smokes and measured GPU peaks permit the two concurrent runs.
Reserve64GiB RAM and20GiB disk. No OOM-driven batch/module shrink and no signals
to other tasks. Each phase has a fresh directory; workers support explicit
same-source full-state --resume, not silent overwrite.

QUEUE.json gives persistent process status,REPORT.json/RESULT.md/PER_RATE.csv
give completed-only scores and provenance. Historical large Flat/default V3
are reused references, not new same-source runs. The main paired comparison is
the new small_flat versus wide_v3. It tests the complete capacity change, not
a parameter-matched proof that the relation structure itself is necessary.

CPU checks:

```bash
python -m pytest -q tests/test_source_role_v3.py tests/test_source_role_v3_integration.py \
  tests/test_source_role_v3_capacity.py tests/test_source_role_v3_runner.py \
  tests/test_mosei_v3_capacity_runner.py
```

New width defaults64/128 preserve original state keys/shapes,initialization,RNG,
outputs and gradients against source98bdb5f; old checkpoint configs gain only
these explicit defaults. Existing immutable experiment sources are not modified.
