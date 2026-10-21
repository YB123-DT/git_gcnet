# Replicate MOSI Flat3e-4→3e-5 across seeds66/67/68

INTERNAL TEST-ORACLE SCREEN

Reuse completed seed66 from
`/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003`.
Only two new runs:seed67 and68,fresh original Flat150epochs,batch32,
Adam,WD1e-5,dropout0.5,original MSE,first100 LR3e-4,last50 LR3e-5.
Original no-JEPA Flat1600/LocalSkip/CHM8×64 joint write0.6 remain unchanged.
Original gcnet-random-mask-v1,cyclic0–.7,per-rate earliest Test-best W-F1.
No Router,source-role,Nested,new loss,new dataset or further search.

Reuse the existing low-LR training/queue implementation with opt-in allowed
seeds and explicit same-seed reference parent. Legacy default seed66 paths,
constant LR wrappers,and the original two-LR plan remain unchanged.
Each seed verifies its own original initialization,data split,seed protocol,
same common evaluation masks,and all150 training mask hashes/update counts.
The reviewed disabled source-role integration is explicitly recorded by
exact source hash pairs;unknown edits are rejected,not silently ignored.

No redundant GPU smoke or original Flat rerun. Reuse the just-completed
identical model validation and1.25GiB peak. At most two concurrent jobs on
healthy biggpuGPU7;4GiB/job plus2GiB margin,64GiB RAM and12GiB disk admission.
Do not kill/move other tasks or change batch size to fit. GPU4 forbidden.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_mosi_mid_lr_3seed_20261021.run \
 --output /data2/yb/remote_experiments/osram_mosi_mid_lr_3seed_20261021/execution_v1 \
 --source-commit SOURCE_SHA \
 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
```

Run under tmux from a committed immutable snapshot;archived s0 environment
Torch2.2.2+cu121,NumPy1.24.3,sklearn1.3.2. Independent outputs;retain eightBEST
and matching same-call masks/predictions plus full model/Adam/RNG last state.
Explicit same-source `--resume` only;no duplicate active/completed jobs or
silent failure retries. Protected previous results/checkpoints are read-only.

After67/68 finish,automatically produce RESULT.md,SUMMARY.json,METRICS.csv for
all three seeds and their same-seed original Flat controls. Verify saved
selected predictions,per-rate epochs,actual LR schedules and aggregate
arithmetic before reporting. Publish per-seed/mean/sampleSD and high-missing
scores. Different worker-wrapper commits across reused66 and new67/68 are
recorded separately;the numerical model/sampler hashes must be identical.
These results remain internal per-rate Test-oracle exploration,not independent
generalization evidence or a single-checkpoint deployment score.
