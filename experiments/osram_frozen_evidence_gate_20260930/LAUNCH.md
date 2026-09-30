# Stage2 launch record

- Code commit: `88937c7`; core unchanged from the verified Evidence Gate.
- Started: 2026-09-30T03:43:25.819708+00:00.
- Coordinator PID1831907, detached with lock.
- Server biggpu, host GPU0 UUID `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`.
- 24 tasks (seeds66/67/68 × eight target rates),100 epochs each, at most3
  simultaneous tasks, staggered resource-checked admission on GPU0.
- Source: original Flat per-rate checkpoints; fresh Gate optimizer, no Stage1
  retraining and no original weights updated.
- Snapshot `/data1/yb/remote_experiments/osram_frozen_evidence_gate_20260930/code`.
- Logs/checkpoints `/data1/yb/remote_experiments/osram_frozen_evidence_gate_20260930/runs`.
- Python `/data2/yb/reproduction_workspace/envs/s0/bin/python`.

Initial child seed66/rate0.0 PID1831910 completed epoch1 with frozen-state check
passing; seed66/rate0.1 PID1833563 was also running. Subsequent
admissions and current progress are recorded in remote children.json/history.json.
This document records startup, not completion. Source and training provenance
are stored independently for every seed/rate. Compact launch records are archived
without weights; unrelated dirty local gcnet/model.py was not synced or committed.

Full result remains pending. Best includes unchanged epoch0; report final-epoch
scores and best selected epochs alongside best deltas to avoid claiming a benefit
from simply retaining the original checkpoint. All results are Test-oracle
internal diagnostics, not formal validation-selected results.
