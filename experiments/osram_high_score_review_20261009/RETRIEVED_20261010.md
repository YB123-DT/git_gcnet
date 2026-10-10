# Retrieved high-scoring one-stage block shortlist

INTERNAL DIAGNOSTIC ONLY. Read-only retrieval, no training or inference.
Scope: cfg84 no-JEPA MOSI, original100 epochs, per-rate Test-oracle BEST.
Not an exhaustive ranking of all historical architecture/hyperparameter runs.
W-F1 %, equal-rate mean; high=.5/.6/.7. Confirmed three seeds are66/67/68.

| Block | Seed66 mean8 | Seed66 high | Three-seed mean8 | Three-seed high |
|---|---:|---:|---:|---:|
| Flat reference |81.068|76.352|80.559|75.594|
| XCiT-XCA |81.042|76.257|80.213|75.365|
| Original Nested GNN |80.992|76.077|80.489|75.252|
| Original Neural Production (Base/Gap only) |80.981|76.475|80.459|75.558|
| D3-W256 scalar shift filter |80.932|76.200|79.953|74.992|
| CWN cellular evidence |80.821|76.004|not verified here|not verified here|
| Perceiver IO |80.818|76.126|not verified here|not verified here|
| Tucker fusion |80.813|75.984|not verified here|not verified here|

All block rows are one-stage joint training, not frozen two-stage correction,
label-oracle splice, or separately changed task objectives. Original Flat remains
in their pathways. Placement and sources checked in RESULT.md and
../osram_readout_path_audit_20261009/RESULT.md; do not describe Perceiver as
consuming the already fused1600-d hidden: it computes a parallel pre-norm residual
from Local/Memory.

Existing lower-LR continuation is separate: Original Nested100 then50 more epochs
at1e-4 gives three-seed80.562/75.296 and seed6681.163/76.077. It has150-epoch
budget and retains old BESTs; do not insert it into the100-epoch ranking.
Source: ../osram_nested_low_lr_extend150_20261010/RESULT.md.

Takeaway: Original Nested is closest to Flat on confirmed three-seed mean8;
Original Neural Production is closest on high missing. None of these confirmed
block means exceeds Flat. This retrieval does not authorize new experiments.

Evidence read: RESULT.md; ../osram_readout_top3_3seed_20261005/RESULT.md;
../osram_nested_low_lr_extend150_20261010/RESULT.md. Original review includes
source metrics paths and module integration descriptions.
