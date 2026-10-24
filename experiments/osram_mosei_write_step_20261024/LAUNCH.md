# Verified finite launch

INTERNAL TEST-ORACLE SCREEN. No completed formal scores yet.

Runner/source commit `2938efd55bbb14c4d6d539f57945ec97607b8e8e`;
immutable composite has1489 Python/AGENTS hashes, manifest
`c37d46c2917c51cfea300e9a683e35c6a0ef8bd8d25a904f13f7c9c33a35e167`.
Numerical base is the accepted cmerc snapshot; all new files are standalone
experiment code/tests. No model/sampler or running capacity source was edited.

Source: `/data2/yb/remote_experiments/osram_mosei_write_step_20261024/source_v1`.
Outputs: same root `execution_v1`; tmux `mosei_write_step_20261024`;
controller PID1581587, log `controller_v1.log`.

| Write step | Worker PID | Initial verified state |
|---:|---:|---|
| .2 | 1581694 | live, loading/preflight |
| .4 | 1582226 | live, loading/preflight |
| .8 | 1582408 | live, loading/preflight |
| 1.0 | 1583014 | live, loading/preflight |

All four commands use `/data2/yb/conda_envs/cmerc-repro/bin/python`, GPU7 UUID
`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`, seed66/60epochs/constant1e-4.
Initial receipt is intentionally preflight: it does NOT claim that each formal
job has already logged effective config or completed an epoch.
Exact receipt: `LAUNCH_RECEIPT.json`. All existing GPU0/1 workers left untouched.

20 relevant configuration/plan/protocol tests passed locally and in remote
cmerc. A separate real GPU step.4 smoke completed:3 actual Memory scans and
3 optimizer updates,72 connected/nonzero original task-gradient tensors,
finite task optimization, exactly restored selected predictions after checkpoint
reload, and one saved CUDA RNG state. It took61.436seconds with peak allocated
686.126MiB. No smoke weights are used to initialize formal training.

Smoke: same root `smoke_v1`, log `smoke_v1.log`; original optional torch-sparse
import warning retained. No dependencies, precision, batch, or model shape changed.
The controller will admit at most4 workers subject to full-phase reservations,
RAM/disk/margin checks, preserve failures and automatically summarize only
when all four formal runs complete. The existing completed Flat0.6 is reused.

For the already running capacity batch, at12:46:47UTC, elapsed about33minutes:
pure Flat25614/60, Flat12811/60, Flat384/76810/60; Nested arms7–8/60.
Average observed epoch time was2.25–3.12minutes for pure Flat and4.06–4.66minutes
for Nested combinations. Linear remaining estimates were1.7–2.6hours versus
3.5–4.1hours, not a controlled benchmark: GPUs have different configurations
and shared foreign workloads. Current rooted implementation has explicit
per-active-pattern/per-root/per-layer loops, with9/17/25 active head-level
root nodes depending on current missingness. This identifies implementation
overhead, not a measured per-operator runtime attribution. No speed rewrite
was applied to active runs.
