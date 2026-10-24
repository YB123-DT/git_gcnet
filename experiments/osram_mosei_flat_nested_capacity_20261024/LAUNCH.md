# Actual finite launch,2026-10-24

INTERNAL TEST-ORACLE SCREEN. Formal results are not yet claimed.

- Server: biggpu.
- Runner commit: `22d4514`; numerical base remains
  `7ec78786b12b6006bd8f6ebcfab32ee9f6026a23`.
- Immutable composite source:
  `/data2/yb/remote_experiments/osram_mosei_flat_nested_capacity_20261024/source_7ec7878_runner_22d4514`.
- Output: same root, `execution_v1`; controller log `controller_v1.log`.
- tmux: `mosei_flat_nested_capacity_20261024`; controller PID at launch1361210.
- `SOURCE_MANIFEST.json` records1359 Python hashes. Numerical packages exactly
  match the historical width screen except one registry entry for Nested256;
  removing that line reproduces the original registry byte-for-byte.
- s0 Python/Torch2.2.2+cu121/NumPy1.24.3, LR1e-3,60epochs,batch32,Adam/WD1e-5,
  MSE, gcnet-random-mask-v1 cyclic0.0-0.7, accepted exact test masks and
  per-rate TEST-best W-F1 unchanged. No cmerc or CUDA RNG repair is introduced.

At11:56:07 UTC all seven new process IDs were alive, in data/preflight; the
existing Flat256 had completed3epochs. No completed-score claim from startup.

| GPU | Arm | PID | Actual total params | Actual Nested params |
|---:|---|---:|---:|---:|
| 0 | f128 | 1361431 | 4746977 | 0 |
| 0 | f384 | 1362011 | 6270945 | 0 |
| 0 | f768 | 1362216 | 8556897 | 0 |
| 0 | f256_n64 | 1362374 | 5668196 | 159235 |
| 1 | f256, already running | 1337899 | 5508961 | 0 |
| 1 | f256_n128 | 1362930 | 5941348 | 432387 |
| 1 | f512_n128 | 1363447 | 7465316 | 432387 |
| 1 | f512_n256 | 1363609 | 8355684 | 1322755 |

Counts were measured from actual CPU model construction for all seven new
configs, not inferred from names. Every new config preserved the exact
same-width non-Nested parameter initialization digest. Six new/local scope
tests passed; nine relevant tests passed in real remote s0, with one existing
torch-geometric deprecation warning. Syntax/diff/source-extension checks passed.
First formal training epochs verify numerical task gradients, scan count,
training-mask trace and original LR/update protocol; no extra GPU smoke sweep.

Flat256 remains in its original independent run:
`/data2/yb/remote_experiments/osram_mosei_flat256_20261024/execution_v1/runs/flat256`.
It is observed by the new queue and counted toward GPU1's maximum4, not
restarted or duplicated. Physical GPU0 UUID43d98f5a... and GPU1 UUID56b14af1...
are recorded in each command/summary; process logical device is cuda:0.
All protected checkpoints, masks and predictions remain intact. Foreign
tasks are not signalled. GPU4 is not used.

The queue automatically reports all eight approved arms and reused plain
512/1024/1600 controls. Only completed runs enter score rows; per-rate CSV and
same-width Nested deltas are generated without new inference. At final
completion RESULT.json requires all11entries. This is a one-seed internal
screen, not an independent-test generalization or significance claim.
