# COMPLETE — Original Flat three seeds, full fivefold

Verified remotely on2026-10-17:RUNNER/SUMMARY complete;30/30validfoldresults,
including10reusedseed66and20newseed67/68. No new training was launched during
this status audit. RESULT.md,SUMMARY.json and SUMMARY.csv were retrieved from
the existing authorized output. The following launch/pause notes are historical.

Verified each of30foldsummaries:100epochs,`gcnet-random-mask-v1`,Flat readout,
no meaningful Block,original Local Skip,source785a012,Torch1.13.1+cu117.
Recomputed per-rate UA from stored class recall,rate means,fivefold seed means
and three-seed mean/sample SD. All agree. Retention check:240BEST checkpoints
and30last_training states exist. No fresh model inference or test-mask sampling.

Current three-seed eight-rate means:4class ACC79.351±0.242%,UA79.632±0.368%;
6class ACC61.714±0.264%,UA60.101±0.186%. All INTERNAL TEST-ORACLE diagnostics,
not a single-checkpoint deployment score or independent-test claim.

## Historical running/launch note

Latest user instruction supersedes the pause: finish original IEMOCAP4/6 Flat
three-seed results on GPU5, two concurrent fold jobs. Reuse Flat66; train only
seeds67/68,100epochs/full fivefold each. No B6 or MELD rerun. Runtime stays
cmerc-repro and numerical source785a012. Actual two-worker launch and runtime
have been verified; see LAUNCH_GPU5.md. This is not training completion.

## Historical pause record

User instruction on2026-10-16: pause new experiments and first inventory which
existing results are usable, their scores, environments and evaluation scopes.

The20new Flat67/68 fold jobs have **not started**. The implementation was
committed as1128548 and transferred to an immutable remote checkout; this is
deployment, not training. The intended output root
`/home/qinjinghui/remote_experiments/osram_iemocap_flat_3seed_cmerc_20261016`
does not exist, and no RUNNER/queue state has been created there. No controller
or worker for this new experiment was launched. The read-only process search
matched only its own inspection command, not an experiment process.

Existing Flat66 runner and B6 runner both report complete. No unrelated GPU
process was killed, paused or modified. OMX reports no active execution mode.

Keep code, completed66/B6 evidence and all checkpoints intact. Do not launch
this prepared queue or treat seed67/68 as running/completed without a new user
instruction to resume. The latest stop instruction overrides the preceding
authorization to start the three-seed completion.
