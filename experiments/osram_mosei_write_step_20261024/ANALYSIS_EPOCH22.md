# Equal-budget interim check: first22epochs

INTERNAL TEST-ORACLE SCREEN. Interim analysis, not completed60epoch results.
Read-only query of saved histories; no training, new inference or mask sampling.

All five arms use the same first22 training mask traces, optimizer update counts
and constant learning-rate vectors. Initial parameter digest is identical:
`9a6a03edc69f7bd45c04464be8105eac96ec01d9a0922bbd2eb496a69ce82f77`.
Per-rate W-F1 is selected by earliest test maximum within epochs1–22 only;
ACC comes from the same call. This avoids comparing the full60epoch control
against incomplete20–23epoch candidates.

| write_step | ACC | W-F1 | High W-F1 |
|---:|---:|---:|---:|
| .2 | 83.269 | 83.161 | 79.485 |
| .4 | 83.390 | 83.090 | 79.712 |
| .6 control | 83.606 | 83.448 | 79.820 |
| .8 | 83.293 | 83.121 | 79.667 |
| 1.0 | 83.365 | 83.112 | 79.808 |

## Example: write_step.4 vs.6 per-rate W-F1

| Missing rate | .6 control | .4 | Delta pp |
|---|---:|---:|---:|
| .0 | 87.758 | 87.419 | -.339 |
| .4 | 83.411 | 82.433 | -.979 |
| .7 | 77.800 | 78.493 | +.693 |

Different conditions move in different directions; the eight-rate mean hides
that heterogeneity. This comparison changes training and selected checkpoints,
not a frozen single-checkpoint sensitivity measurement.

The scalar multiplies the original block update correction
`(Vbar - D @ Kbar) @ P`, scaling both new-value injection and erasure of old
associations along the observed keys. It is not a selective relevant-history
writer, does not add observations, and cannot manufacture never-observed Text.
This explains why increasing write strength is not inherently an improvement;
it does not prove which mechanism caused these interim score differences.
Learned-alpha/beta compensation or readout adaptation remain untested hypotheses.

Source0.6: `osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none`;
new arms: `osram_mosei_write_step_20261024/execution_v1/runs/step_{0p2,0p4,0p8,1p0}`,
all under `/data2/yb/remote_experiments/`. Read `history.json` and for each
rate take `max(range(22), key=lambda i: history[i]['test'][rate]['router']['WF1'])`;
then aggregate the selected eight metrics. Final60epoch ranking remains pending.
