# Launch receipt — not final results

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY.

Verified on biggpu at 2026-10-23 23:51:56 UTC. Controller and persistent tmux
are live; all five admitted workers have completed their first real training
epoch without a reported error. Snapshot details and process IDs: LAUNCH.json.
The source snapshot remains immutable at commit
`45d06f2ec780c450e9f1a6dceaaca072cbd2d76b`; this receipt does not alter it.

Only Nested128, IEMOCAP6, seeds66/67/68, folds1–5 are scheduled. Reuse one
completed seed66/fold5 run and execute14 new runs. Each runs100epochs,
constantLR1e-3, original batch32/missing/loss/model. No additional Flat training.

Six-process ceiling was requested. The reference worker's measured peak is
5536MiB; six peaks exceed32GB before context overhead. Resource admission
therefore currently runs **five**, not six, workers and automatically fills
released capacity. At the verified sample GPU0 was99% utilized,31714MiB used,
781MiB free. Do not force a sixth process or silently shrink the model/batch.
Memory measurements are point-in-time observations, not an OOM guarantee.

The earlier IEMOCAP6 groups4 worker was interrupted solely because the user
stopped other IEMOCAP6 variants. Its receipt is retained at the old queue's
`execution_v1/IEMOCAP6_FIXED128_SCOPE_CHANGE.json`. Existing MOSEI workers were
not interrupted. No protected checkpoint was removed or overwritten.

46 regression tests passed both locally and in remote cmerc-repro; all1464
snapshot files and131 unchanged numerical model/sampler files were verified.
The reused completed run passed strict checkpoint/prediction/mask/history
checks. These checks establish implementation/protocol integrity, not improved
performance. Complete fifteen-run results and fivefold/seed aggregation are
still pending. Historical Flat fivefold scores are not strict paired controls;
see README.md for the evaluation-mask and initialization caveats.
