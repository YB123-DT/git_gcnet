# V3 Hierarchical Source–Role Interaction

INTERNAL TEST-ORACLE SCREEN

Status: **both approved fresh training jobs running**, observed2026-10-21
13:01–13:02UTC. No completed V3 performance result yet.

| Dataset | Seed | Budget | LR | Status | PID |
|---|---:|---:|---|---|---:|
| MOSI | 66 | 150epochs | 3e-4 epochs1–100, then3e-5 | running | 3638584 |
| MOSEI | 66 | 60epochs | constant1e-3 | running | 3638587 |

Both use original batch32/Adam/WD1e-5/MSE/cyclic0–.7
`gcnet-random-mask-v1`; no extra loss or second view. Per-rate Test-best W-F1
checkpoints, matching ACC/MSE/predictions/masks and full `last.pt` are retained.
Eight-rate means are not single-checkpoint deployment performance.

## Implemented structure

Shared512→64 read projection; separate343-d Base–each-active-Gap pairs per
source; Base aggregates messages only AFTER pair encoding. Same-role3-source
attention uses zero/single/multiple-source handling. Four role-specific Local
queries and four independent zero-init64→512 decoders correct original forward
Base/Gap inputs. **No global128→1600 residual** and no change to Local Skip,
original Flat, CHM writes/queries or task head. Added335,006 parameters;
total13,844,799.

## Verification actually executed

- 16 s0 remote CPU module/integration tests; 6 local runner tests passed.
- Original write/scan/address/projection function ASTs unchanged.
- Both real-data GPU smokes passed; all41 monitored gradient paths finite and
  nonzero after actual updates, including all four role queries/decoders and
  actual source-read tensors. Original parameters also updated.
- Zero-init baseline predictions exact; original initialization and exact
  per-rate Flat test masks matched. One Memory scan per training batch.
- Model/Adam/RNG restored and predictions exactly replayed from checkpoint.

| Smoke | Actual optimizer updates | Peak allocated MiB | End-to-end seconds |
|---|---:|---:|---:|
| MOSI | 4 | 3198.563 | 70.036 |
| MOSEI | 3 | 2244.944 | 75.142 |

Smoke includes setup/evaluation/checkpoint I/O, is not a controlled speed
benchmark, and proves no performance benefit. See `VALIDATION.md` and
`LAUNCH.json` for evidence/hashes.

## Actual execution

- Server: biggpu; original s0 Python/Torch2.2.2+cu121/NumPy1.24.3/sklearn1.3.2.
- Host GPU1: `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`; logicalcuda:0.
- Immutable source: `f62310bb3d4737c827133d4a4c5ef35b6bbc269f`,9621 Git
  blobs verified; snapshot `source_f62310b`.
- Persistent tmux socket `osram_v3`, session `seed66_f62310b`, controller3626149.
- Remote root:
  `/data2/yb/remote_experiments/osram_source_role_v3_20261021/execution_v1`.
- Logs: `train_CMUMOSI.log`, `train_CMUMOSEI.log`; worker `runs/<dataset>/`.
- Existing Nested and old checkpoints remain untouched; old SourceRelation /
  12-token runs are not resumed. No automatic seed expansion or HPO.

On completion, the controller automatically writes per-rate ACC/W-F1/MSE,
eight-rate/high-missing aggregates and matched Flat comparisons to remote
`RESULT.md`, `REPORT.json`, `METRICS.csv`. Those completed results are not yet
claimed here. This local record is the verified launch, not a completion claim.
