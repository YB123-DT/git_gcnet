# MELD write/retention diagnostic status

**INTERNAL DIAGNOSTIC ONLY — BLOCKED, no real-data diagnostic result.**

The original seed66 `gcnet-random-mask-v1` Flat source directory on biggpu
contains config, history, metrics and all eight saved prediction/mask files,
but no `best_miss_0p0.pt` through `best_miss_0p7.pt`.

Its `PROVENANCE.json` explicitly records `discarded_checkpoints.count=8`,
`discarded_checkpoints.bytes=425530104`. Read-only backup searches under
`/data2/yb` found old-mask Flat and canonical no-Local-skip weights but no
matching original canonical Flat weights. No alternative model was used.

Source:
`/data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1/meld/meld_flat_s66`

Implemented both authorized diagnostics, runtime checks, streaming rows,
CSV aggregation and plotting entrypoint. Synthetic tests check exact original
forward parity, normal state replay parity, analytic decay, constant hold,
strict-past candidate pools and capture restoration. These do not establish
MELD performance or diagnose its learned weights.

Verification on biggpu with the established PyTorch 1.13.1+cu117 environment:
`python -m pytest tests/test_meld_write_retention.py -q` — **4 passed (1.98s)**.
Local synthetic checks also passed under PyTorch 1.12.0+cu113. Syntax checks
passed for audit, runner and analysis. The actual eight-rate runner preflight
was attempted on biggpu and failed with the eight explicit missing checkpoint
paths, before GPU allocation, dataset loading or result-directory creation.

Diagnostic code was staged separately at
`/data2/yb/remote_experiments/osram_meld_write_retention_20261014/code`.
No model training, real-data inference, source checkpoint mutation or deletion
was performed by this task. No performance or learned-state findings exist.

To complete: recover the original eight checkpoints. Saved predictions and
Local/Base/Gap caches cannot reconstruct the Key/Value projectors or full
Memory state. No retraining is authorized in this task.

## Subsequent authorization and recovery plan

The user subsequently authorized searching for backups and reproducing the
original baseline if none exists. Local MELD checkpoint search and a remote
configuration scan under `/data2/yb` found only the canonical original seed66
Flat r0/r1 directories with **zero .pt files**. Available retained MELD
weights belong to different missing-mask protocols or no-Local-skip models.

Implemented `reproduce.py` to train exactly the historical `b4a3138` code and
source config: MELD seed66, 50 epochs, batch32, Adam lr1e-3, cyclic GCNet
random mask, original Flat/Local skip, per-rate Test-oracle. Keep eight best
checkpoints, compare historical/new results, then automatically run both
audits and analysis. This remains a reproduction for internal diagnostics.

Reconstruction statistics were vectorized across source/head pairs to avoid
per-row GPU synchronization. A fifth regression test verifies equivalence to
the scalar metric calculation. Actual job launch/status is recorded below;
no learned-state conclusion is available yet.

## Running reproduction and automatic audit

- Server: `ssh biggpu`, host `user22`, physical GPU0.
- GPU UUID: `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`.
- Started UTC: `2026-10-14T10:30:38`.
- Pipeline PID: `4018218`; baseline training PID: `4018230`.
- Original training code: detached Git worktree at `b4a3138`.
- Diagnostic/pipeline code: detached Git worktree at `27ae943`.
- Output: `/data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66_reproduced_b4a3138`.
- Status file: `PIPELINE.json`; training log: `training.log`.
- Source config SHA256: `ce88eb4166a18f4bd1539326035c8a4f706fe5a4b6c42115f5b8b6599cccd58d`.
- Startup confirmed epoch001 completed, config/provenance created and CUDA
  execution on GPU0. Tests after vectorization: **5 passed on biggpu (1.98s)**.

This is a new reproduction. Once training succeeds, the same persistent job
runs the audit and analysis without another user command. No observed
write/retention/overwrite conclusion is reported while these stages are pending.
