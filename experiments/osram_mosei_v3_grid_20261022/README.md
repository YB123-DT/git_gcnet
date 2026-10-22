# MOSEI small-Flat / larger-V3 capacity grid

INTERNAL TEST-ORACLE SCREEN. Exactly four new seed66 runs;two old runs reused.

| Configuration | Flat hidden | V3 width / PairMLP | Total parameters | Action |
|---|---:|---|---:|---|
| f512_plain | 512 | off | 7,032,929 | train |
| f512_v128 | 512 | 128 / 256 | 7,864,639 | train |
| f512_v256 | 512 | 256 / 512 | 9,349,567 | train |
| f1024_v128 | 1024 | 128 / 256 | 10,912,575 | train |
| f1024_plain | 1024 | off | 10,080,865 | reuse |
| f1024_v256 | 1024 | 256 / 512 | 12,397,503 | reuse |

Old largeFlat has13,509,793 parameters;the original V3 module has335,006.
Width128/256 V3 modules have831,710/2,316,638 parameters. Remote effective counts
are checked. Flat output1600 and original LocalSkip/task head stay unchanged.

No new architecture:only existing Flat intermediate and V3 width switches.
CHM8heads,key/value64,Base/Gap Query,joint Block Delta Write,step0.6 and causal
read-before-write remain. No Gate,JEPA,Nested,completion,extra loss or dual view.

## Fixed protocol

CMUMOSEI seed66,official splits/features,archiveds0 environment:
`/data2/yb/reproduction_workspace/envs/s0/bin/python`,Torch2.2.2+cu121,
NumPy1.24.3,sklearn1.3.2.60epochs,constantLR1e-3,Adam,WD1e-5,batch32,
original dropout,MSE,emotion-only,cyclic0..0.7,new gcnet-random-mask-v1.
This is NOT the MOSI3e-4/3e-5 schedule or its150epoch budget.

Same original seed66 test availability hashes for every model at each rate.
Training masks are still freshly generated,not fixed;each epoch verifies
original training-mask/order hash,optimizer-step count and LR trace.
Per-rate TEST-best WF1,ACC/MSE from same-call selected predictions. Eight
selected checkpoints may use different epochs,not one deployment checkpoint.

Reuse completed weights/predictions without retraining or new inference:

```text
/data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/execution_v1/runs/small_flat
/data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/execution_v1/runs/wide_v3
```

Their eight-rate/highWF1 are83.077816/79.849565 and83.104360/79.819696.
Original largeFlat66 reference83.279076/80.181722 remains read-only. Reporting
verifies new/reused model/sampler/task core hashes,data/split/config and exact
mask parity;entrypoint/instrumentation changes are explicitly recorded.

## Execution

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
 -m experiments.osram_mosei_v3_grid_20261022.run \
 --source-commit SOURCE_SHA \
 --gpu-uuids GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
 --per-gpu 2 \
 --output /data2/yb/remote_experiments/osram_mosei_v3_grid_20261022/execution_v1
```

biggpu GPU1/2,two jobs per GPU maximum. Each GPU runs one3-real-batch smoke at
a time,then measured peak+50%margin+2GB reserve controls formal admission.
Reserve64GiB RAM/50GiB disk;never shrink batch/model for concurrency. GPU4 is
forbidden. Preserve all old/new checkpoints,including smokes,and all other tasks.

Reuse the old verified worker/queue via default-preserving narrow injection,
not four duplicated model wrappers. Same-widthFlat zero-init predictions and
unrelated parameter initialization must match exactly. All new V3 groups must
receive finite/nonzero gradients;model/Adam/RNG/prediction reload checked.
Only new-run gradient hooks stop after first-epoch validation to avoid ongoing
instrumentation overhead;task math,optimizer and RNG are not changed.

Persistent tmux queue automatically finishes four jobs and updates completed-only
RESULT.md/REPORT.json/PER_RATE.csv including explicitly reused rows. Keep8BEST,
same-call predictions,exact masks and complete last/model/Adam/RNG/history.
Single-seed screening only,no automatic HPO or multi-seed expansion.
