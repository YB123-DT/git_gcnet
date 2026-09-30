# Formal launch

Code3d372aa. Requested server biggpu, host GPU6 Tesla V100-SXM2-32GB,
UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153, process logical cuda:0.
Three seeds66/67/68,100 epochs each, up to3 concurrent children, independent
outputs; stagger20 seconds and require at least5GB free before each admission.
No formal feature-only task was launched on GPU0. Existing unrelated tasks remain.

Detached coordinator PID3875106; seed66 PID3875109 verified on GPU6 by nvidia-smi,
1952MiB, epoch1 complete. Current admission/progress in remote children.json/logs.
Exact launch timestamp/configs/reference hashes archived in launch_records.

```bash
cd /data1/yb/remote_experiments/osram_feature_only_gate_20260930/code
/data2/yb/reproduction_workspace/envs/s0/bin/python experiments/osram_feature_only_gate_20260930/run.py --launch --gpus 6 --max-tasks-per-gpu 3
```

Runs at `/data1/yb/remote_experiments/osram_feature_only_gate_20260930/runs`.
Immutable committed source snapshot overlaid on prior verified environment;
per-seed provenance records source hashes. Python3.10, torch2.2.2 CUDA12.1.
Original dataset `/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset`.
Fresh one-stage joint training; not resume or checkpoint initialization.
Original cfg84 random-only/no-JEPA objectives and optimizer unchanged.
Retain eight best checkpoints per seed using original internal Test-oracle rule.
Launch verification does not establish completion or performance improvement.
