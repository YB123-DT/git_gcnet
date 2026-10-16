# What the four local-parameter trials are built on

Verified against actual `PLANNED_JOBS.json`, original baseline documentation,
and completed worker summaries. No training or evaluation was added for this note.

The architecture is original cfg84 no-JEPA Flat1600, with original raw Local Skip,
plus the existing B3 `mbt_role_bottleneck` input adapter. It is not Nested or B6.
Global lr0.001,WD1e-5,dropout0.5,batch32 and GCNet random missing stay unchanged.
Both `initial_backbone_checkpoint` and `b2_base_checkpoint` are null: all four
models train from scratch, not by loading the61.020 model and continuing training.

## Relevant completed reference scores

All rows below are MELD seed66,eight-rate Test-best ACC with matching UA:

| Model/configuration | ACC | UA |
|---|---:|---:|
| Original Flat, current screen reference | 60.694444 | 35.293130 |
| B3 default:4tokens/2rounds,dim128,heads4,FFN256,LR0.001 | 60.469349 | 36.654640 |
| Previous highest-ACC B3:1token/3rounds,other knobs default | 61.020115 | 35.472568 |

The user-accepted three-seed Flat baseline remains60.693ACC/36.255UA,including
accepted seed66 at60.675ACC/35.436UA. The current screen reference is a separate
same-hyperparameter seed66 rerun,not a replacement for that accepted baseline.
These stochastic Test-best results must not be confused with common-mask replay.

## Important distinction

The four new models jointly vary capacity/LR and tokens/rounds. Their tokens/rounds
are4/3,2/3,3/3,5/4 respectively. **They are not four single-variable modifications
of the61.020 configuration,whose token/round choice is1/3.** There is no common
trained parent checkpoint whose starting score can be assigned to all four.

Calling these B3-local parameters correctly identifies which module changes,
but does not imply the winning B3 structure or its weights were held fixed.
This distinction should have accompanied the earlier local-parameter description.
