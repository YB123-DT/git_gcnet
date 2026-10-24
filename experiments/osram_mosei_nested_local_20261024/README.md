# Four local tests around MOSEI Nested32

Current execution (2026-10-24): the user authorized relocation to idle physical
GPU1. Follow `MIGRATION_GPU1.md` and remote `execution_gpu1_v1/QUEUE.json` for
live status. The original GPU2 `execution_v1` is preserved and marked migrated;
it is NOT the active queue. Three full checkpoints resume at epochs14/12/13,
and the pre-epoch OOM arm receives one fresh retry. No new configurations.
The GPU2 launch command and resource notes below document the original setup.

INTERNAL TEST-ORACLE SCREEN. Exactly four fresh seed66/fold1 runs on biggpu
physical GPU2, not an automatic HPO or multi-seed expansion.

|Variant|Graph dim|Depth|Graph groups|MLP hidden|Backbone LR|Nested LR|
|---|---:|---:|---:|---:|---:|---:|
|nested32_lr5e5|32|3|8|32|1e-4|5e-5|
|nested32_lr2e4|32|3|8|32|1e-4|2e-4|
|nested32_depth2|32|2|8|32|1e-4|1e-4|
|nested32_groups4|32|3|4|32|1e-4|1e-4|

60epochs,constant learning rates,Adam,WD1e-5,batch32,global gradient clipping1.0,
original MSE/dropout/splits/features/cmerc-repro. Original gcnet-random-mask-v1
cyclic0–0.7 training and accepted exact test-mask bank are unchanged. TEST-best
W-F1 independently selects each rate; ACC/MSE use the same saved predictions.
All original and Nested parameters train end-to-end from canonical random
initialization. No pretrained weights,extra loss,Gate,V3,JEPA or paired views.
CHM8x64/joint BlockWrite/read-before-write/write_step0.6,Flat1600 and LocalSkip
are unchanged. Graph groups are NOT CHM heads. Residual and zero decoders remain.

The LR opt-in partitions the existing Adam parameter group AFTER construction:
original backbone stays1e-4; every parameter in `osram.meaningful_block`,
including tokenizer/core/decoders, enters the Nested group at the approved LR.
Parameters must be covered exactly once; both groups inherit the original
weight decay and are globally clipped together. Without the option, the exact
old optimizer factory is returned unchanged. Default configs/models/RNG are not
modified. Group name/LR/parameter counts are saved in identity and SUMMARY;
actual group LR vectors are logged each epoch and restored with full Adam state.
The CLI option only authorizes Nested32/MOSEI66/fold1 at5e-5 or2e-4.

Reuse completed read-only controls in
`/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/{none,nested_dim32}`.
No new Flat is trained. A new controller entry delegates admission/recovery to
the existing finite GPU2 controller via explicit planner/reporter/validator
callbacks; default old-controller behavior is unchanged. Old immutable sources,
results and all checkpoints are protected. The only numerical-package edit is
one registry entry for32/3/4/32; the2-layer32/MLP32 mechanism already exists.
Source proof removes the five accumulated registry additions relative to the
old controls and requires exact remaining registry bytes plus identical130
other numerical model/sampler files.

GPU2 UUID `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4` only (logical cuda:0),
GPU4 forbidden. At inspection another `python cv_fmnv.py` job used about23GB
on GPU2. It is not signalled or modified. The queue reserves6144MiB per worker
plus2048MiB global margin and charges external usage: currently this admits
one of our jobs, with three waiting until capacity permits. Hard ceiling4,
64GiB RAM/25GiB disk minimum. No batch/precision changes or silent OOM retry.
An SSH-independent tmux controller automatically fills authorized free slots.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_mosei_nested_local_20261024.run \
  --source-commit SOURCE_SHA --output /absolute/new/execution_v1
```

Run from committed immutable source with SOURCE_SNAPSHOT.json. Complete-only
RESULT.md/JSON and PER_RATE.csv include both controls/all4candidates, eight-rate
and high0.5/.6/.7 ACC/WF1/MSE, deltas versus BOTH controls, epochs,parameters,
time and memory. Pair validation permits only the explicitly listed local LR
vectors, not a changed global LR. Init/mask/split/source and training-mask/update
traces must match. No partial winner, automatic promotion or significance claim.
Retain all8BEST,exact predictions/masks,history and complete model/Adam/CPU+CUDA
RNG last.pt. Smoke tests are engineering checks, NOT performance evidence.
