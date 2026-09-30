# One-stage L1 launch

Code commit4c98d3e. Started2026-09-30T09:27:30.998910+00:00 on biggpu hostGPU0
UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45. Detached coordinatorPID3284282,
max3 concurrent seeds66/67/68,100 epochs each, original batch/optimizer/masks.

Source `/data1/yb/remote_experiments/osram_evidence_gate_l1_20260930/code`,
outputs `/data1/yb/remote_experiments/osram_evidence_gate_l1_20260930/runs`.
Python `/data2/yb/reproduction_workspace/envs/s0/bin/python`.
Only committed core changes and this runner synced into the isolated verified
snapshot; unrelated dirty local gcnet/model.py was not copied or committed.

This is fresh one-stage joint training, not Stage2, not a resume, and not a
checkpoint-initialized run. Gamma0.2, lambda0.001, penalty typeL1. Original L2
results remain untouched. Existing reference checkpoints/configuration hashes,
effective configurations and source hashes are archived in launch_records and
remote per-seed provenance. Source dataset/features/split unchanged.

First admitted child seed66 PID3284286. Current admissions/progress are in
remote children.json/history.json; launch does not mean training completed.
Eight best checkpoints per seed retained with original Test-oracle protocol.
