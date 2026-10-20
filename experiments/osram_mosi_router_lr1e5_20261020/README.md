# MOSI backbone LR1e-5 paired screen

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY.

User2026-10-20 requests reducing MOSI backbone learning rate to1e-5.
Finite scope: seed66 Flat and the accepted one-stage three-candidate Text-source
Router, two fresh150epoch runs on biggpu. No extra seeds or search.

Only backbone LR changes: constant1e-5 for all150epochs, replacing1e-3 for100
then1e-4 for50. Router stays3e-4 for100 then3e-5 for50. This explicitly tests
a constant low backbone LR, not an extra late-stage decay or checkpoint fine-tune.
Adam,wd1e-5,batch32,dropout,MSE,original Local Skip/Flat1600/CHM unchanged.
One-stage jointly train all original backbone and Router parameters.

Original gcnet-random-mask-v1 host/guest algorithm and cyclic0.0–0.7 remain.
Use the same fixed legal test masks, initialization, official data split and
training-mask sequence as completed seed66 under osram_mosi_gcnet_testbest_20261020.
Check every epoch's actual masks/optimizer step count against that reference;
do NOT require equal MSE or LR after deliberately changing LR.
Original protected weights only support the existing strict code/data prediction
bridge and are discarded before fresh initialization. No old results overwritten.

Reuse s0 Python/Torch2.2.2+cu121/NumPy1.24.3/sklearn1.3.2. A three-batch real GPU
smoke checks gradients, both updates, full checkpoint/RNG restoration and exact
selected-prediction reload before launching both runs. Smoke is not a score.
One healthy UUID-bound GPU, up to two training children subject to real RAM/VRAM
admission; GPU4 forbidden. Queue persists under tmux; no peer tasks killed.

Each rate selects TEST W-F1 BEST with earliest ties; save same-call predictions,
mask,ACC/MSE,8BEST and complete last.pt (Adam/RNG/history/best ledger).
No validation selection or added holdout. This is not independent test evidence.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_mosi_router_lr1e5_20261020.run \
  --output /data2/yb/remote_experiments/osram_mosi_router_lr1e5_20261020/execution_v1 \
  --source-commit SOURCE_SHA --gpus HEALTHY_GPU_UUID
```

The queue automatically publishes RESULT.md/METRICS.csv/SUMMARY.json only after
both150epoch trajectories and paired integrity checks complete. Until then,
status and scores are provisional. No promise that a smaller LR improves W-F1.
