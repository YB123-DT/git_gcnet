# Full training launched

Status: RUNNING, not a completed performance result.
Implementation commit: 237c03c, feature/osram-uniform-forced-text.
Server: biggpu; host GPU0 only, 3 concurrent seeds.
GPU UUID: GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45.

Coordinator PID277872; seed66 PID277918; seed67 PID277919; seed68 PID277920.
Each child loaded MOSI dimensions512/1024/1024 and appeared on host GPU0.
Launcher command:

```bash
cd /data1/yb/remote_experiments/osram_cfg84_history_input_gate_20260929/code
/data2/yb/reproduction_workspace/envs/s0/bin/python -u experiments/osram_cfg84_history_input_gate_20260929/run.py --launch --max-tasks-per-gpu 3
```

Live logs/configs/checkpoints/provenance:
/data1/yb/remote_experiments/osram_cfg84_history_input_gate_20260929/runs.
100 epochs per seed, original reference protocol, one-stage joint training.
The local launch_records directory is a launch-time snapshot, not live status.
Do not edit or resync the isolated running code. Match source hashes in each
PROVENANCE.json to the implementation commit when collecting final results.

After the detached diagnostic float64-moment change, gate+runner tests were
rerun remotely: 11 passed, one existing PyG deprecation warning. Earlier
52-test regression, actualCUDA identity/finiteupdate and real-data smoke
are described in VERIFICATION.md. No model behavior changed in that final
diagnostics-only adjustment. Gate performance remains unknown until completed.
