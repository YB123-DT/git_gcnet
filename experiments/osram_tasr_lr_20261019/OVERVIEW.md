# TASR completed experiment overview

INTERNAL DIAGNOSTIC ONLY. This round has12 completed training jobs:
four original frozen F1/F2 jobs, six F2 LR jobs, one frozen random-only MELD
job and one from-scratch joint random-only MELD job. No new job was launched
for this read-only overview. LR queue was rechecked statuscomplete,6/6 workers
complete at30epochs, no errors/failures, each original backbone unchanged.

## Frozen experiments: same fixed miss=.3 backbone per dataset

F0=no Adapter; F1=source-neutral G/3 control; F2=actual Value-source reads.
Only the Adapter trains; same seed66,30epochs and validation random8 ACC/CE
selection. Default training is .8random+.2Persistent, except the last MELD row.
IEMOCAP6 is ONLY fold5, not a fivefold mean. No cross-dataset score comparison.

### MELD

| Method | LR | Random8 ACC | UA | W-F1 | Persistent6 ACC | UA | W-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 fixed Flat |—|61.336207|38.860509|58.642901|57.605364|32.040601|50.850579|
| F1 source-neutral |3e-4|61.350575|38.912840|58.646722|57.586207|32.012838|50.845623|
| F2 |3e-4|61.307471|38.964871|58.637211|57.637292|32.094797|50.919355|
| F2 |3e-6|61.345785|38.867107|58.652345|57.605364|32.040601|50.850579|
| F2 |1e-5|61.355364|38.869950|58.659877|57.605364|32.040601|50.852059|
| F2 |3e-5|61.393678|38.912260|58.694478|57.598978|32.044304|50.860298|
| F2 random-only |1e-5|61.360153|38.876305|58.664634|57.598978|32.038706|50.846801|

The largest frozen MELD random8 ACC among these candidates is61.393678,
only+0.057471pp vs matchingF0. Its Persistent6 ACC is slightly lower thanF0.
The original3e-4 F2 Persistent gains are small. No convincing broad benefit
is established by these single-seed numbers.

### IEMOCAP6 seed66/fold5

| Method | LR | Random8 ACC | UA | W-F1 | Persistent6 ACC | UA | W-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 fixed Flat |—|62.615527|62.092186|62.383000|49.794619|49.899143|48.903857|
| F1 source-neutral |3e-4|62.630930|62.111220|62.405095|49.845964|49.922054|48.946995|
| F2 |3e-4|62.707948|62.085688|62.499920|50.349148|50.237083|49.475234|
| F2 |3e-6|62.615527|62.092186|62.383000|49.794619|49.899143|48.903857|
| F2 |1e-5|62.623229|62.106654|62.393568|49.804888|49.906434|48.914437|
| F2 |3e-5|62.607825|62.080608|62.379588|49.794619|49.890094|48.904613|

Original3e-4 F2 has the clearest limited positive signal: Persistent6 vsF1
+0.503183pp ACC/+0.315029pp UA/+0.528239pp W-F1. AgainstF0 the differences
are+0.554529/+0.337940/+0.571377pp. V-only deteriorates; see original per-pattern
report. Smaller LRs do not improve on that configuration. This is one fold,
not a statistically validated or multi-seed result. F1 was NOT LR-tuned too;
new F2 LR rows are optimization comparisons, not independently matched F1 ablations.

## Joint experiment: separate protocol, do not mix with frozen rows

MELD seed66 from scratch, all13,350,218 parameters trainable,50epochs,
random-only CE, LR1e-5/wd1e-5, per-rate test ACC-best (user override).

| Model | Test-best Random8 ACC | UA | W-F1 |
|---|---:|---:|---:|
| Original Flat seed66 |61.786398|38.420205|58.889822|
| One-stage TASR seed66 |61.786398|38.420205|58.889822|

No classification gain in that run; metrics/selected epochs match at every
rate. This table uses a different evaluation/selection protocol than frozen
fixed-anchor rows. It is NOT legitimate to call61.786−61.336 a TASR gain, nor
to compare seed66 directly against the accepted three-seed mean61.557.
Final joint artifact checks are now verified in
../osram_tasr_joint_20261019/FINAL_BRANCH_CHECK.md: all eight selected class
prediction arrays match original Flat after exact identity/label/mask alignment.
The Adapter learned nonzero weights and has nonzero finite gradients; its
same-checkpoint on/off replay produces zero corrections/harms, with only one
wrong-to-wrong class change at random0.7. It is not an entirely inactive module,
but its small numerical contribution yields no measured classification benefit.

## Sources and verification

- Original F1/F2/F0:../osram_tasr_20261018/COMPARISON.json and RESULT.md;
  all raw paired masks/labels/IDs had already been verified there.
- All six LR jobs:remote
  `/home/qinjinghui/remote_experiments/osram_tasr_lr_20261019/runs_v1/queue/QUEUE.json`
  and per-job SUMMARY.json. The automatically generated RESULT.md and
  LR_RESULTS.csv were downloaded unchanged. CSV90rows verified:6 jobs ×
  (8 random rates +7 Persistent patterns), with paired counts/deltas.
- Random-only frozen:../osram_tasr_random_only_20261019/RESULT.md and both
  preserved raw SUMMARY JSON files.
- Joint:../osram_tasr_joint_20261019/RESULT.md and TESTBEST.csv; original
  remote completion evidence is described there without inventing a failed
  full-SUMMARY download.
- No training, new inference, test selection, hyperparameter changes, GPU
  allocation or deletion performed in this overview task. These remain
  exploratory scores with historical Test-oracle anchors, not paper claims.
