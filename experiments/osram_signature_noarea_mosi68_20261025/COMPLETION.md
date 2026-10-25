# Completed matched MOSI seed68 control

All8formal jobs completed150epochs; controller finished at
2026-10-25T15:53:24Z,approximately4minutes15seconds after launch including smoke
and staggered startup. Only frozen-parent Signature branches trained,not OSRAM
from scratch. No failed jobs,additional seeds or parameter search.

Independent validation of saved artifacts is in VERIFICATION.json. All selected
weight/prediction hashes matched; all150per-epoch sampling/update/N/LR traces
matched full Signature; labels,availability,metadata and frozen Flat predictions
were array-identical across the paired methods. Selected ACC/WF1 and arithmetic
eight-rate/high-rate means were independently recomputed. The automatic paired
report completed successfully. Local and remote17focused tests passed before
launch; every CUDA update was checked for finite loss/gradient/clip norm.

| Scope | Model | ACC | W-F1 |
| --- | --- | ---: | ---: |
| Eight-rate mean | Flat | 79.173018 | 79.074201 |
| Eight-rate mean | Full Signature | 79.115854 | 79.127840 |
| Eight-rate mean | Retrained NoArea | 79.173018 | 79.191066 |
| High0.5–0.7 | Flat | 72.764228 | 72.770415 |
| High0.5–0.7 | Full Signature | 73.221545 | 73.307194 |
| High0.5–0.7 | Retrained NoArea | 73.221545 | 73.368640 |

NoArea minus full: +0.057165ACC/+0.063226WF1 points overall,approximately
0ACC/+0.061446WF1 high-missing. Individual rates show mixed signs; see RESULT.md.
The tiny aggregate advantage is not statistical superiority. On seed68 the full
Signature gain over Flat was already small overall,so neither mechanism should
be described as a large MOSI improvement.

Together with the separate IEMOCAP68 control,this does not support attributing
the branch benefit mainly to antisymmetric directional area. It does not isolate
which retained feature is useful: endpoints,time,counts,symmetric quadratic
features,current Local and learned capacity remain bundled. This remains
seed68-only,frozen-parent stage2,internal per-rate Test-oracle evidence,not a
one-stage model,three-seed result or independently selected generalization claim.

Source5fb3659 and the original numerical source4d5aa09 remain untouched. Remote
output,full last checkpoints,selected predictions and exact masks are preserved
under the LAUNCH.json output path. Only small text/JSON/CSV records are committed;
no weights,data or large prediction arrays are uploaded.
