# External review handoff: source-slot MoE

INTERNAL TEST-ORACLE SCREEN. Observation snapshot: residual epoch142/150;
replacement complete150/150. Source implementation b92490d.

## Evidence

| Model | Eight-rate ACC | W-F1 | MSE | Trainable parameters |
|---|---:|---:|---:|---:|
| Accepted cmerc Flat66 |79.535061|79.460447|1.496185|13,509,793|
| Flat + MoE |79.001524|78.899805|1.461273|14,897,331|
| MoE replacing Adapter |78.182165|78.191477|1.571141|5,362,227|

Same new GCNet missing, accepted exact test masks and data order; seed66,
MSE, batch32, Adam WD1e-5, LR3e-4 first100 epochs then3e-5 for50.
Per-rate selection by test W-F1; MSE comes from those same selections, not
separately MSE-selected checkpoints. Not independent generalization evidence.

Twelve Value-source/role reads projected512→64 with source/role IDs, retained
as separate fixed concatenated slots. Four independent MLP859→256→128 experts
see all slots, Local64, current availability, historical trajectory and mask.
Router859→128→4 softmax; weighted128-dimensional output→zero-initialized1600.
Original LocalSkip, final LayerNorm and scalar head retained in both arms.
No auxiliary loss, privileged teacher, new Memory query or changed write.

Actual last-training-batch diagnostics (NOT whole-test statistics):

| Arm / epoch | Mean expert weights | Mean entropy (max ln4=1.386294) | Mean pairwise output cosine |
|---|---|---:|---:|
| Residual100 |.249,.282,.228,.242|1.380260|.184354|
| Residual142 |.247,.283,.237,.233|1.380892|.224775|
| Replace150 |.229,.264,.232,.275|1.363614|.057855|

Residual142 per-expert across-sample weight std: .0123,.0222,.0187,.0145.
All38 new parameter tensors received finite nonzero gradients during training;
this is coverage evidence, not a gradient-strength or sufficient-learning claim.
Initial predictions exact versus Flat; single causal scan, masks, causality,
source gradient path and checkpoint reload checked.

## What is not established

- Near-uniform routing is not single-expert collapse, nor evidence that experts
  produce identical outputs. No complete-test routing or routing intervention yet.
- Lower MSE but lower polarity WF1 raises an objective/decision-boundary question;
  it does not localize mistakes or prove an objective mismatch caused the drop.
- Replacement removes9,535,104 active Adapter parameters, adds1,387,538 MoE
  parameters, and funnels the new contribution through a shared128-dimensional
  representation. It is not a capacity-matched test of expert versus nonexpert fusion.
- Replacement bypasses Adapter Dropout, changing the CUDA stochastic operation
  sequence. Initial parameters and availability masks match, not all random draws.
- No same-checkpoint residual-off experiment yet; direct branch contribution and
  joint-training changes to the original backbone remain unseparated.
- One seed only; no significance/variance claims.

## Questions for reviewer

Review code and evidence before suggesting another architecture. Does dense soft
mixing without an explicit specialization objective plausibly become an almost
fixed ensemble here? How can that be tested without assuming sharp routing is good?
Does flatten859→256→128 preserve the distinctions needed for decision-making?
How should the replacement comparison separate capacity, representation bottleneck
and removal of the original nonlinear path? What minimal frozen test would
separate router selection, learned expert transformations, and backbone drift?
Could the observed MSE/WF1 divergence be localized to near-zero polarity examples
without treating metric selection as proof of a mechanism?

Rank at most two inexpensive tests, state alternative explanations and rejection
criteria. Do not use 'Flat is already strong' or 'the residual is weak' as a
substitute for evidence; do not prescribe additional HPO/seeds before identifying
what a test would distinguish.

Sources: live `execution_v1/{residual,replace}/{SUMMARY,history}.json` under
`/data2/yb/remote_experiments/osram_source_moe_mosi_20261025`; accepted Flat
`/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2/SUMMARY.json`.
Code: `gcnet_missing_m3/source_moe.py`, `osram.py`, experiment `run.py`.
