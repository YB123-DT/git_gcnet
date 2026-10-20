# Status: all24 new runs complete; five-fold results verified

2026-10-20: ordered queue completed24new100epoch jobs afterMOSI succeeded.
The existing sixfold5 jobs were reused read-only, giving30complete runs total.
RESULT.md containsallfold/seed summaries;METRICS.csv contains240rate rows.
SUMMARY.json recordscompactrun details, source/mask checks and seed-first
five-fold aggregation. All240prediction files passed storedSHA256 checks and
independent ACC/UA/W-F1 recomputation (max error2.85e-14pp).

Five-fold mean then three-seed mean:Flat ACC55.738/UA55.583 versusJoint
ACC56.019/UA56.089. ACC deltas byseed66/67/68 are approximately
+2.033/-0.046/-1.142pp. Overall gain is small and not consistent acrossseeds;
the earlierfold5 +2.570pp ACC result does not generalize as a five-fold gain.
The scoring/holdout/mask-provenance limitations inREADME/RESULT remain.
No additional jobs were launched and no checkpoint was deleted. Earlier waiting
and PID records below are historical; the queue is now complete.

tmux:iemo6_folds_3864630; controllerPID2978054.
Immutable source:38646302489b1ce74cf2d4ddeb9a9ec0ae5eb49d (shared with MOSI).
Output:/data2/yb/remote_experiments/osram_iemocap6_router_fivefold_20261019/execution_v1.
QUEUE.json verified `waiting_for_upstream`,24pending jobs and0training jobs.
After the six-job new GCNet-MOSI queue succeeds, this controller performs its
real Joint smoke and starts the approved IEMOCAP jobs automatically.
GPU7/1/3,at most2training children/card. Protected fold5 summaries and unchanged
core source hashes passed preflight and are retained read-only.

Startup/source/resource evidence for both suites is in
`../osram_mosi_gcnet_router_3seed_20261019/LAUNCH.json`.
No IEMOCAP training has started in this new remaining-fold queue yet; do not
describe a waiting controller as a running model.77local related tests passed.

Approved scope:24 new jobs for folds1–4, Flat/Joint × seeds66/67/68,100epochs.
Completed fold5 results will be reused without retraining. Queue must follow
the six new GCNet-missing MOSI jobs, not run the older failed legacy queue.
