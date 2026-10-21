# Source-role tokens redo

INTERNAL TEST-ORACLE SCREEN

Status: CANCELLED BY USER at2026-10-21T11:52:31.797049+00:00.
The user suspects the architecture does not match the intended design.
Controller3091132 and unfinished MOSEI worker3098451 were verified by exact
command/output path and stopped with SIGINT to their isolated process group.
Both processes exited; no unrelated experiment was signalled. No automatic
resume, new variants or further training are authorized by the old launch.

This controlled redo keeps Base/Gap-A/Gap-T/Gap-V × historical Value-source
A/T/V as 12 independent 128-dimensional tokens before Attention. It removes
the previous early active-Gap mean. The previous mean followed the original
user-provided formula; this is a design revision, not a correction of an
unfaithful implementation of that formula.

The same original Flat pre-norm zero-initialized residual connection is retained
explicitly to keep the final readout interface fixed. Token encoding and parameter
count also change; this is not a one-factor causal test of mean pooling. CHM updates, original Query/K/V,
read-before-write, Local Skip, task head/loss and GCNet missing stay unchanged.

| Dataset | Seed | Budget | New W-F1 mean | New high W-F1 | Status |
|---|---:|---:|---:|---:|---|
| MOSI | 66 | 150 epochs | 78.870661 | 73.162256 | completed150 before stop |
| MOSEI | 66 | 60 epochs | 82.694594 | 79.542845 | user stopped after20; incomplete |

MOSI matched original Flat150 is79.249087/73.409792 W-F1 mean/high:
the completed candidate is lower by0.378426/0.247536 percentage points.
MOSEI20 is not a completed60-epoch result and is not ranked as one.
All scores remain per-rate TEST-best internal exploration, not a deployable
single-checkpoint or independent-test result.

Eight BEST weights and their matching prediction/mask files for each dataset
were SHA256-validated after stopping. Both full `last.pt` files loaded with
model/optimizer/RNG/identity/history and matched the last recorded completed
epoch (MOSI150, MOSEI20). No weights, predictions or logs were deleted.
Unfinished-epoch updates are not claimed saved. [CANCELLED.json](CANCELLED.json)
is the authoritative stop record; LAUNCH.json/SMOKE.json are earlier snapshots.

Configuration and reproduction command: [README.md](README.md).
Immutable source: `da4a9f4898267a5de3f79c9dae5a9b3328399739`, pushed to
`github:feature/ecerc-features`;9600 deployed Git blobs matched exactly.
Added parameters530,470; full model14,040,263. CPU verification64 passed,
one Git-history-only test explicitly deselected; local runner12 passed.

| GPU smoke only | Steps/scans | Peak MiB | Seconds | Gradient paths nonzero |
|---|---:|---:|---:|---:|
| MOSI | 2/2 | 3085.500 | 28.202 | 38/38 |
| MOSEI | 3/3 | 1526.664 | 60.009 | 38/38 |

Both actual batches verified all12 read-input slots, role identity and upstream
Attention gradients, exact zero-init Flat predictions, and full checkpoint
model/Adam/RNG restoration. Original and added parameters both updated.

Server biggpu, original s0 environment, healthy physicalGPU7 UUID
`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`. Persistent tmux socket
`osram_role_tokens`, session `pair66_da4a9f4`, controller PID3091132.
Remote output:
`/data2/yb/remote_experiments/osram_source_role_tokens_20261021/execution_v1`.
Remote `QUEUE.json` and the interrupted MOSEI `SUMMARY.json` are now explicitly
marked `cancelled_by_user`; finished MOSI retains `complete`. The controller
has stopped, so its automatic final report will not run. The old source-mean
experiment had already finished both datasets and was not modified.

See [SMOKE.json](SMOKE.json), [LAUNCH.json](LAUNCH.json) and
[VALIDATION.md](VALIDATION.md) for provenance and verification boundaries.
Model correctness and successful smoke do not establish performance improvement.
