# Completed finite cyclic Signature screen

All48 formal jobs completed successfully. The persistent controller finished at
2026-10-25 11:13:25 UTC and exited normally. Final QUEUE and RESULT both report
complete; RESULT contains48 per-rate rows and no invalid/incomplete groups.

Immutable numerical source is `e0ca94c4c33de0efda2c2f4786fd289067e1ab2c`.
Server,source/output directories,environment and UUID mapping are in LAUNCH.md.
All protected parents and old experiment outputs remain unchanged. Only the
unchanged Signature branch trained. No extra seeds,HPO or models were started.

## MOSEI three-seed comparison

Each entry is an equal eight-rate mean. Sentiment selection remains same-rate
Test-best W-F1, not independently maximized ACC or MSE.

| Seed | Frozen Flat ACC | Signature ACC | Frozen Flat WF1 | Signature WF1 | ΔWF1 pp |
| --- | ---: | ---: | ---: | ---: | ---: |
|66|83.626857|84.018987|83.477635|83.798754|+0.321119|
|67|83.396395|83.544304|83.248318|83.356464|+0.108146|
|68|83.468630|83.654375|83.305343|83.483993|+0.178650|
|Mean|83.497294|83.739222|83.343765|83.546404|+0.202638|

Signature seed-level sample SD is0.248456 ACC and0.227654 WF1 (n=3).
Mean MSE changes0.608090→0.597487. All three WF1 differences are positive,
but improvements are small and this exploratory comparison does not establish
statistical significance or independent generalization.

The completed fixed-rate stage2 MOSEI66 reference is83.797158 WF1 versus
cyclic83.798754, a difference of+0.001596pp: effectively equal here. Fixed-rate
MOSEI67/68 are not part of this experiment and must not be invented to form a
three-seed cyclic-versus-fixed comparison. Reference is
`experiments/osram_signature_stage2_extension_20261025/RESULT.json`.

## MELD seed66

| Version | ACC | UA | WF1 | CE |
| --- | ---: | ---: | ---: | ---: |
|Protected frozen Flat|61.786398|38.420205|58.889822|1.195518|
|Cyclic Signature|61.714559|38.338685|58.846609|1.166448|
|Difference|−0.071839|−0.081520|−0.043212|−0.029070|

UA/WF1/CE all use the ACC-selected epoch's exact predictions and mask. CE
improves,but ACC/UA/WF1 slightly decline: this does not support a classification
benefit on MELD. There is only one MELD seed in this finite screen.

## MOSI completion and prior seed68

The new66/67 cyclic WF1 means are79.847119 and79.678873. Together with the
previously completed68 cyclic79.127840, three-seed mean is79.551277 versus
frozen Flat79.338163 (+0.213114pp). The fixed-rate Signature three-seed mean
is79.588415,so cyclic is−0.037138pp lower. This does not support cyclic being
better than independent-rate stage2. Prior68 evidence remains in
`experiments/osram_signature_cyclic68_20261025/RESULT.json`; it is not counted
as a new formal job in the48-run queue.

## Verification and limitations

The controller verified source/parent hashes, selected checkpoint/prediction
integrity,epochs/budgets,earliest strict selection,original GCNet missing,
frozen-parent/no-gradient checks and matched initialization/training-mask traces
across the eight parents within each dataset/seed. After completion, all48
selected prediction NPZ files were independently loaded and every stored metric
(including corrections/harms and MELD class recalls/support) was recomputed and
matched the selected summary exactly. No missing or failed run was omitted.

Original Flat and Signature use each rate's own accepted frozen parent and exact
test mask. Eight branches are eight separately trained models; these means are
not a single checkpoint's deployment performance. Selection uses the test set
at both the old-parent and new-branch stages,with additional stage2 training
budget. Results remain INTERNAL TEST-ORACLE exploration,not paper-grade
independent test evidence. No inference from higher attention or a new module's
successful implementation is substituted for actual task gains.

Full per-rate/per-seed/high-missing records are in RESULT.md/JSON/CSV.
Large weights,predictions,data and caches stay on biggpu and are not uploaded.
