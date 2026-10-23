# What was actually tried: MOSEI Flat/Nested capacity

2026-10-23, read-only repository/remote lookup. INTERNAL DIAGNOSTIC ONLY.

No completed MOSEI **new GCNet Missing smallFlat+largeNested** experiment was
found in the checked records. Do not replace that conclusion with the claim
that every possible historical run was exhaustively disproved.

| Existing series | Dataset/protocol | Actual changed configuration |
|---|---|---|
| `osram_mosei_nested_tuning_20261011` | MOSEI, old Missing | Only Nested dim/depth/MLP/groups; Flat fixed |
| `osram_mosei_dim32_depth2_followup_20261012` | MOSEI, old Missing | Nested selected structure, three seeds; no small Flat |
| `osram_mosei_nested_gcnet_3seed_20261021` | MOSEI, new GCNet Missing | Original Nested64/depth3/MLP64/groups8, Flat1600 fixed |
| `osram_mosei_v3_capacity_20261022` | MOSEI, new GCNet Missing | Flat1024+V3-256; module is V3, not Nested |
| `osram_nested_equal_budget_20261009` | MOSI, old Missing | Actual smallFlat256+Nested64/704 capacity comparison |
| `osram_medium_flat_nested_20261009` | MOSI, old Missing | Flat384–1280 with fixed original Nested |

Direct implementation evidence: MOSEI old tuning
`experiments/osram_mosei_nested_tuning_20261011/run.py::candidate_config`
requires changed fields to be exactly `{'osram_meaningful_block'}`; it does
not change Flat width. Its six candidates are dim32/depth2/dim32+depth2/
MLP128/groups4/dim32+depth2+MLP64.

## Completed current MOSEI, original Flat1600/Nested

| Seed | Flat W-F1 | Nested W-F1 | Delta pp |
|---:|---:|---:|---:|
| 66 | 83.279 | 83.227 | −0.052 |
| 67 | 83.147 | 83.113 | −0.034 |
| 68 | 83.298 | 83.255 | −0.043 |
| Mean | 83.241 | 83.198 | −0.043 |

High-missing mean W-F1: Flat80.086/Nested80.100. Constant1e-3,60epochs,
batch32,WD1e-5,s0,GCNet-random-mask-v1,per-rate TEST-best W-F1.
Actual completed report read from biggpu:
`/data2/yb/remote_experiments/osram_mosei_nested_gcnet_3seed_20261021/execution_v1/RESULT.md`.
The local source document for that experiment still describes the prepared
state; it must not override verified completed runtime artifacts.

## Nearby capacity evidence, not a Nested MOSEI result

MOSEI new-Missing seed66: Flat1600 W-F1=83.279, Flat1024=83.078,
Flat1024+V3-256=83.104. This is V3 capacity transfer only.

The actual old MOSI equal-budget record gives Flat1600=81.068,
Flat256=79.529, Flat256+Nested64=80.362, Flat256+Nested704=78.152.
These are not current-Missing MOSEI scores and cannot be used in its ranking.

The current user request authorizes only the missing MOSEI Flat66 LR1e-4 run.
No new Nested capacity experiment is launched from this historical lookup.
