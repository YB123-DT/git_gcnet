# Completed: all21 approved formal runs

All21runs completed successfully at full budgets; no failed/partial jobs in
the report. Full results are in `completed/RESULT.md`,machine-readable scores
in `completed/METRICS.csv` and `completed/REPORT.json`; reference comparisons
and their limits are in `completed/COMPARISON.md`.

Independent completion verification: all168selected prediction conditions
recomputed,maximum metric discrepancy6.06e-9; all168selected model SHA256 and
prediction SHA256 checked;21complete-length histories and21last checkpoints
present. No new inference or training was used for completion verification.
Full verification metadata is retained in `completed/FINAL_VERIFICATION.json`.
The persistent queue finished in6.488hours including preflight/smokes/admission.
GPU1 has released this task's training allocations; no peer task was stopped.

Three-seed eight-rate means: MOSEI ACC83.453/W-F1 83.261; MELD ACC61.569/
UA37.058/W-F1 58.319; IEMOCAP-6 fivefold ACC60.729/UA58.882/W-F1 60.360.
All remain internal per-rate Test-oracle exploration. No automatic extension.

## Historical launch snapshot (retained,not live status)

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
