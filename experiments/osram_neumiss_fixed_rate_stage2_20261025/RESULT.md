# Fixed-rate stage2: launch record

Status: **running**, no completed performance claim. INTERNAL TEST-ORACLE SCREEN.

- Source: `0870e99179e34c448c93c9aae60688de0ecb9231`.
- Server: `ssh biggpu`; cmerc-repro Python/Torch1.13.1+cu117.
- Snapshot: `/data2/yb/remote_experiments/osram_neumiss_fixed_rate_stage2_20261025/source_0870e99`.
- Outputs: `/data2/yb/remote_experiments/osram_neumiss_fixed_rate_stage2_20261025/execution_v1`.
- tmux: `fixed_stage2_mosi_20261025`; controller PID3896047.
- Methods: NeuMiss and Deep Signature only; no new CSDI experiment.
- Each rate0–0.7 uses its own accepted frozen Flat BEST; TRAIN and TEST both
  restricted to this same rate. No cyclic switching or cross-rate evaluation.
-15 trained branches:150epochs each, batch32, LR3e-4 then3e-4/10 after100,
  Adam WD1e-5. NeuMiss0 is explicitly a baseline replay, not fake training.
- GPU0 UUID `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`; GPU3 UUID
  `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`; max4 new jobs each. GPU4 excluded.
-22 remote CPU tests passed. Both real rate0.7 full-batch two-update smokes
  passed, including frozen-parent integrity and exact selected checkpoint replay.
- First eight formal processes were observed alive under PID3896685–3896692;
  remaining conditions are persistent queued jobs, admitted as slots release.

Original parents/checkpoints and ongoing one-stage jobs are unchanged. After
all conditions finish, the controller generates per-rate and aggregate result
tables with paired parent differences. Extra stage2 budget and compounded
Test-oracle selection prevent interpreting these as independent test evidence.
