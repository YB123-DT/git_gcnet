# Authorized hierarchical gate training — 2026-09-30

User authorized full launch after implementation verification. Code commit
`e5e17f3` (module `af4f6ab`), on biggpu host GPU0,
UUID `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`; GPU4 excluded.

- Seeds 66/67/68, 100 epochs each, at most three concurrent GPU0 jobs.
- Fresh one-stage joint training; not checkpoint initialization or resume.
- Original cfg84 batch32, Adam lr=.001, weight decay=1e-5, cyclic random
  missing rates0–.7, emotion-only. Only semantic delta:
  `osram_hierarchical_evidence_gate=true`.
- No L1/L2 gate penalty, JEPA, prediction completion, persistent mix or freezing.
- Existing Flat references reused, not retrained. Keep eight per-rate best
  checkpoints per seed. Selection remains Test-oracle **internal diagnostic**,
  not validation-selected paper performance.

Isolated source snapshot:
`/data1/yb/remote_experiments/osram_hierarchical_evidence_gate_20260930/code`.
Output root:
`/data1/yb/remote_experiments/osram_hierarchical_evidence_gate_20260930/runs`.
Snapshot based on verified dependency environment, overlaid with committed
repository archive. Unrelated local dirty files were not included.

Launch from snapshot:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python experiments/osram_hierarchical_evidence_gate_20260930/run.py --launch
```

Detached coordinator PID3694753; first child seed66 PID3694756. Launch records
contain effective configurations, source reference hashes and timestamp.
Per-seed PROVENANCE contains implementation hashes; environment and input data
unchanged from verified cfg84 setup (Python3.10, torch2.2.2 CUDA12.1).
Dataset: `/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset`.

Preflight passed for all three seeds. First child loaded A/T/V dimensions
512/1024/1024 and allocated GPU0. Coordinator staggers starts by20 seconds and
checks UUID/free memory before each admission. Current status/children/history
remain in remote output root. Launch is not evidence of training completion.
