# Launch verified

INTERNAL TEST-ORACLE SCREEN

Actual execution, not a proposed command. Source `eb0e81e` committed and pushed
before immutable deployment. Source archive SHA256 is recorded in LAUNCH.json.

`biggpu` healthy physicalGPU1 only. At this observation six formal children
were live (MOSEI66/67,MELD66/67,IEMOCAP6-seed66-fold1/2),GPU1 used17,788MiB
and99%utilization. Fifteen remaining approved jobs are persistently queued,
automatically filling released slots under the measured memory/RAM policy.
These PID/resource figures are a timestamped launch snapshot, not live status.

The21run plan is MOSEI60epochs×3seeds,MELD50epochs×3seeds,IEMOCAP6-100epochs
×fiveLOSO-folds×3seeds. No additional Flat rerun or search. Direct source sums
replace task-side Base/Gap; original LocalSkip remains. One-stage end-to-end
task training, original GCNet missing. Per-rate Test-bestWF1 for MOSEI and
Test-bestACC for MELD/IEMOCAP; matched UA/WF1 from same selected predictions.

49focused local tests passed. All three real GPU smokes passed: original and
new parameters updated,14new-module gradient tensors were connected/nonzero,
oneMemoryscan/trainingbatch,initialpredictionerror0,strict full checkpoint
reload. Measured peaks1,792.9MiB MOSEI,1,512.4MiB MELD,6,092.5MiB IEMOCAP6.
The optional unused torch-sparse/libcusparse warning is preserved in logs;
the actual Torch OSRAM training paths pass without that optional operator.
Smokes do not establish performance improvement.

No baseline weights,predictions,masks,existing experiments or peers were
deleted,overwritten,killed or migrated. Source/code/tests and launch evidence
are uploaded; weights and large prediction files stay remote. RESULT.md is
generated only after all21formal runs complete successfully.

Read-only review found no launch blocker. MELD retains the original resampled
test evaluation function,but its explicit initialization/trainingRNG boundary
is documented by this new driver. Do not claim exact historical61.557 training
trajectory parity without checking actual masktraces. MOSEI previously had
only a completed seed66 matched Flat; no seed67/68 comparator is fabricated.

Remote root:
`/data2/yb/remote_experiments/osram_source_role_direct_cross_20261020`

Persistent queue:
`execution_v1/QUEUE.json`; controller log:`controller.log`.
tmux:`tmux -L osram_direct_cross attach -t direct_cross_eb0e81e`.
Every worker keeps per-epoch metrics,eight selectedBEST+exactpredictions/masks,
and a complete model/Adam/RNG/history`last.pt` for same-source recovery.
