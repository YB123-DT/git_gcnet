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

## Follow-up specifically for4tokens/3rounds, checked11:18 UTC

The supplied resume ID corresponds to this same experimental conversation.
The current4/3 follow-up is `b3_capacity_01`,not a separate fixed4/3 multi-trial
study. It changes dimension128→64,internal heads4→2,FFN256→512 and adapter
LR0.001→0.0001 together. At33/50completed epochs it is still running.

Its current eight-rate Test-best ACC60.498084/UA35.511931 were recomputed only
from saved selected artifacts; each ACC matched selection metadata. The completed
default-capacity4/3 reference is ACC60.756705/UA37.357830. The interim differences
are therefore approximately−0.259ACC/−1.846UA percentage points,not a final
performance conclusion. No new inference,training,search or cancellation was
performed for this check.
