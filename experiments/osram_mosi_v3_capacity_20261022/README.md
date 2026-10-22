# MOSI: smaller Flat, larger V3

INTERNAL TEST-ORACLE SCREEN. Finite six-arm seed66 capacity allocation screen.

| Mode | Flat hidden | V3 width / PairMLP | Total params | Flat params | V3 params |
|---|---:|---|---:|---:|---:|
| f512_plain | 512 | off | 7,032,929 | 3,058,240 | 0 |
| f512_v128 | 512 | 128 / 256 | 7,864,639 | 3,058,240 | 831,710 |
| f512_v256 | 512 | 256 / 512 | 9,349,567 | 3,058,240 | 2,316,638 |
| f1024_plain | 1024 | off | 10,080,865 | 6,106,176 | 0 |
| f1024_v128 | 1024 | 128 / 256 | 10,912,575 | 6,106,176 | 831,710 |
| f1024_v256 | 1024 | 256 / 512 | 12,397,503 | 6,106,176 | 2,316,638 |

Counts above were instantiated on CPU with actual feature dimensions and original
emotion-only unused-aux exclusion; remote effective-config counts are authoritative.
Original Flat total13,509,793,default V3 module335,006. Every Flat output remains1600.
V3 heads remain4; original CHM heads8,key/value64 remain unchanged.

All six train jointly from scratch. Same official MOSI features/splits,archived s0
Python,Torch2.2.2+cu121,NumPy1.24.3,sklearn1.3.2.150epochs,LR3e-4 first100 and3e-5
final50,batch32,Adam,WD1e-5,dropout0.5,projector_dropout0.1,emotion-only MSE,
cyclic0..0.7,gcnet-random-mask-v1. No auxiliary loss,Gate,JEPA,Nested or completion.
LocalSkip,CHM Query/Key/Value,joint Block Delta Write,step0.6,read-before-write stay.

## Matched protocol

Reuse read-only original Flat66 completed reference:

```text
/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003
```

Its accepted WF1:79.2491 eight-rate,73.4098 high-missing. Original defaultV366
completed reference:79.1394 /73.7128. Neither historical run is retrained here.
The old conversation-mask-v2 80.7 score is not the new GCNet-missing baseline.

Every arm verifies the same exact seed66 test availability hashes per rate. This
does NOT fix training masks or use the same mask across rates. Training keeps
the original sampler and independently generated per-epoch masks,verified against
the original mask/order/step/LR trace. Per-rate TEST-best WF1 selects eight BEST
checkpoints; ACC/MSE must come from those same predictions. Never describe this
eight-rate score as single-checkpoint independent generalization.

Small-Flat initialization is identical within each width. V3 zero-init predictions
must exactly equal that same-width Flat. All unrelated original parameter bytes
must remain equal. Preserve8 BEST+predictions+exact masks+complete last/Adam/RNG;
smoke weights also remain. No deletion or overwrite of existing artifacts.

## Implementation and execution

No core model edit is required: reuse existing osram_adapter_hidden_dim and V3
width switches. Worker supports only six modes,seed66,smoke/train and explicit
same-source full-state resume. V3 gradient hooks collect first-two-epoch cumulative
evidence then are removed; scalar task gradient diagnostics and all computation
math/RNG remain unchanged. A scan counter remains enabled.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_mosi_v3_capacity_20261022.run \
  --source-commit SOURCE_SHA \
  --gpu-uuids GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
  --per-gpu 3 \
  --output /data2/yb/remote_experiments/osram_mosi_v3_capacity_20261022/execution_v1
```

On biggpu GPU3 and7,one smoke per GPU at a time. Each smoke has two epochs/four
real optimizer updates to open both initially-zero original adapter and V3 decoder.
Checks finite/nonzero original and V3 task gradients,single scan,zero-init
equivalence and full model/optimizer/RNG/prediction restore. After all pass,at
most3 formal jobs per GPU,admitted by actual peaks with1.5x reserve+2GB margin,
including pending allocations of live peers. Insufficient space queues jobs,
never shrinks batch/width. GPU4 is forbidden. Existing MOSEI queues are untouched.

QUEUE.json persists commands/PIDs/states; RESULT.md/REPORT.json/PER_RATE.csv
summarize only completed runs. The controller needs tmux or equivalent persistence.
No automatic seeds67/68,HPO or additional methods.
