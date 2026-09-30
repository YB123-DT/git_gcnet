# Authorized launch — 2026-09-30

Status: **running**, not completed. Internal Test-oracle comparison only.

- Code commit: `af37f56` (module implementation `7d2de7e`).
- Server: `ssh biggpu`, host GPU0 UUID `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`.
- Started UTC: `2026-09-30T01:59:41.601258+00:00`.
- Coordinator PID: `1461848`, detached persistent subprocess with launch lock.
- Seeds: 66, 67, 68; max concurrency3 with free-memory admission and stagger.
- Code snapshot: `/data1/yb/remote_experiments/osram_local_evidence_gate_20260930/code`.
- Outputs: `/data1/yb/remote_experiments/osram_local_evidence_gate_20260930/runs`.
- Python: `/data2/yb/reproduction_workspace/envs/s0/bin/python`, 3.10.20,
  torch2.2.2+cu121, CUDA12.1.

Command: see README. Original reference configs verified for all three seeds;
100 epochs, batch32, Adam lr0.001, weight_decay1e-5, original features/split and
cyclic eight-rate masks retained. New gate enabled, penalty coefficient0.001.
No previous checkpoint is loaded; this is from-scratch training, not resume.
Eight `best_miss_0p*.pt` files are retained per seed by the original trainer.
The existing trainer's saved best weights are not claimed to be complete
optimizer/RNG resume checkpoints.

Snapshot copied from the verified isolated cfg84 code then updated with the
committed launcher. Core SHA256 matches the prior VERIFICATION.md. Launcher:
`5919faaac139c2f020df6a6053417c2aba95b0437da2de426086df2e7febcce8`.
Unrelated local dirty `gcnet/model.py` was not synced or committed. Final
effective configs and reference config/metric hashes are in launch_records;
each remote seed also records source hashes in PROVENANCE.json.

Observed after launch: seed66 PID1461851 completed epoch1 with JEPA loss0 and
finite task loss; seed67 PID1463685 and seed68 PID1464857 are also active.
All three were admitted on GPU0 after resource checks. No accuracy claim is made
from the first epoch. Check current remote children.json/logs for later status.

Pre-launch launcher regression: 7 tests passed; original module verification:
63 tests passed and actual cfg84 initial train/eval output equality on GPU0.
Remaining: finish all three100-epoch runs, validate canonical masks/checkpoints,
then report all seeds and paired reference deltas, including negative outcomes.
