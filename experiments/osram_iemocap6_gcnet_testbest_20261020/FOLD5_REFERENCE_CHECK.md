# Fold5 reference check,2026-10-20

INTERNAL DIAGNOSTIC ONLY. Currentcorrected Test-best suite fold5sixjobs are
allpending at thisinspection. No currentfold5gain can yet be reported.

The following is ONLY the old validation-selected fold5round,with extra10%
trainingconversationholdout. It does NOT satisfy the restored requested
per-rateTest-best protocol. Its weights were deleted by userauthorization;
these preserved summarymetrics remain valid records of that distinctexperiment.

| Seed | Flat ACC | Router ACC | Delta ACC pp | Flat UA | Router UA | Delta UA pp |
|---:|---:|---:|---:|---:|---:|---:|
|66|58.841651|59.765866|0.924214|59.674207|59.634800|-0.039407|
|67|55.691620|60.027726|4.336106|59.173334|58.259126|-0.914207|
|68|56.061306|58.510474|2.449168|55.261797|58.329751|3.067953|
|Mean|56.864859|59.434689|2.569830|58.036446|58.741226|0.704780|

ACC increased in allthreeoldseeds; UA increased only inseed68,not66or67.
Do not use this to claim success of the currentTest-best/full-trainingfold
round. No new training,inference or schedule changes were made by thisaudit.

Source:the sixfold5reference SUMMARY.json files indexed by QUEUE.json under
`/data2/yb/remote_experiments/osram_iemocap6_router_fivefold_20261019/execution_v1`.
Their prioridentities/hashes and deletionstatus are pinned in CLEANUP_PLAN.json
and CLEANUP_RECEIPT.json in thisdirectory.
