# Local Skip Gate launch

Codeafb1774. Server biggpu hostGPU6 Tesla V100-SXM2-32GB,
UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153; process logical cuda:0.
CoordinatorPID3239269, first child seed66 PID3239276 verified GPU6 allocation
1728MiB and original feature dimensions512/1024/1024. Up to3concurrent seeds
66/67/68, stagger20seconds, at least5GB free per new admission.

Snapshot `/data1/yb/remote_experiments/osram_local_skip_gate_20261001/code`;
outputs `/data1/yb/remote_experiments/osram_local_skip_gate_20261001/runs`.

```bash
cd /data1/yb/remote_experiments/osram_local_skip_gate_20261001/code
/data2/yb/reproduction_workspace/envs/s0/bin/python experiments/osram_local_skip_gate_20261001/run.py --launch --gpus 6 --max-tasks-per-gpu 3
```

Fresh100-epoch one-stage joint runs, not resume or loaded Flat checkpoints.
Only semantic delta osram_local_skip_gate=true. Original batch32/Adam.001/
weightdecay1e-5/cyclic random masks/emotion-only MSE retained. W-F1 is primary
assessment; do not reject a run merely because MSE increases. No extra loss.
Eight best checkpoints perseed retained under historical per-rate Test-oracle
protocol, internal diagnostics only. Flat references not rerun.

Committed source overlaid on verified isolated environment; unrelated local
dirty gcnet/model.py excluded. Python3.10, torch2.2.2 CUDA12.1, unchanged dataset
`/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset`.
launch_records stores exact timestamp/config/reference hashes; remote perseed
provenance records code hashes. Currentstatus in remote status/children/logs.
Startup verification does not establish completion or performance improvement.
