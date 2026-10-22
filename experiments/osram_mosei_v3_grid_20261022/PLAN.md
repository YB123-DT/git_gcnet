# Complete the finite MOSEI capacity grid without repeated runs

Exactly four new seed66 runs:
f512_plain(512,off),f512_v128(512,128/256),f512_v256(512,256/512),
f1024_v128(1024,128/256). Reuse completed f1024_plain and f1024_v256 from
osram_mosei_v3_capacity_20261022/execution_v1/runs/small_flat and wide_v3.
No repetition of those configurations;no seeds67/68,HPO or other datasets.

All original MOSEI settings:60epochs,constant1e-3,batch32,Adam,WD1e-5,original
dropout,emotion-only MSE,new gcnet-random-mask-v1,cyclic0..0.7,official split,s0.
Flat output1600/LocalSkip/CHM8x64/Query/KV/joint write/step0.6 unchanged.
Per-rate TEST-best WF1 with matching ACC/MSE/predictions/masks,internal only.

- [x] Tests-first worker config/count/source injection;default oldtwo-arm behavior
  unchanged. Newwrapper firstepoch-only gradient hooks,not task math changes.
- [x] Reuse existing finite multi-GPU queue with default-preserving injection;
  four new independent workers on healthy GPU1/2,two per GPU by measured peaks.
- [x] Same-widthFlat zero-init equality,exact original masks/init/data/split,
  per-epoch train-mask/order/step/LR parity;full last/Adam/RNG and eightBEST kept.
- [x] Report complete-only grid,explicit reused rows,original LargeFlat and
  original V3 references. Validate source model/task core unchanged across runs.
- [x] Regression checks,immutable git archive,real MOSEI three-batch smokes,
  actual formal launch verification,small evidence commit/push. No existing
  checkpoints/results or other tasks changed or deleted.

One-seed capacity screening,not proof of stable performance or causal mechanism.
No new inference for already completed rows;reuse same-call selected predictions.
