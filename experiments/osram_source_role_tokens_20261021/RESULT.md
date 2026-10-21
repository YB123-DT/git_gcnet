# Source-role tokens redo

INTERNAL TEST-ORACLE SCREEN

Status: RUNNING — both real-data GPU smokes passed and two fresh seed66
training processes launched. No completed new training scores claimed.

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
| MOSI | 66 | 150 epochs | N/A | N/A | launched PID3098447 |
| MOSEI | 66 | 60 epochs | N/A | N/A | launched PID3098451 |

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
Read live `QUEUE.json` and per-run `SUMMARY.json` there; this report is a startup
snapshot. The controller automatically generates completed per-rate metrics,
RESULT.md, REPORT.json and METRICS.csv after the finite jobs finish.

See [SMOKE.json](SMOKE.json), [LAUNCH.json](LAUNCH.json) and
[VALIDATION.md](VALIDATION.md) for provenance and verification boundaries.
Model correctness and successful smoke do not establish performance improvement.
