# Scope reduction: IEMOCAP6 only

User explicitly requested no IEMOCAP4. The new effective plan is14 jobs:
MOSEI seed66 (7) and IEMOCAP6 seed66/fold5 (7). No extra seeds/folds or methods.

The operator verified the exact original controllerPID3725311 and paused only
that parent to prevent further admissions. Exact-command-verified IEMOCAP4 Flat
workerPID3726264 received SIGINT; its checkpoints, masks, scores and logs remain.
Original MOSEI FlatPID3725708 was NOT signalled, reseeded, migrated or restarted.

The scope-only rescope.py supervisor takes over the persistent queue and adopts
living retained workers by verified PID/command. It cancels all7 IEMOCAP4 ledger
entries without deleting evidence. All14 retained workers keep the original
immutable numerical source3777ab1 and commands, including pending jobs. Only the
controller/reporting scope changes. The original queue is backed up before the
handoff; controller and worker commits are recorded separately.

Remote receipt: execution_v1/IEMOCAP4_SCOPE_CHANGE.json.
Old queue evidence: execution_v1/QUEUE_before_iemocap4_cancel.json.
The original LAUNCH.json and STATUS.md are historical21-job snapshots; this
file supersedes their scope, not their recorded launch facts.

Verification:23 targeted scope/protocol/source/parity tests pass, including a
regression that reduction leaves the living MOSEI command and PID unchanged,
does not add --resume and preserves the original worker source SHA. No new GPU
smoke is needed: no numerical model, optimizer, loss or sampler was changed.

Remote handoff verified: new controllerPID3776376 is running with active_jobs14
and all7 IEMOCAP4 entries cancelled_by_user. Original parentPID3725311 and
IEMOCAP4 workerPID3726264 have exited. Original MOSEI workerPID3725708 remains
the SAME live process and has advanced to epoch3/60, with no error. Worker source
remains3777ab1; scope supervisor source is3114dda. New persistent session:
`nested_cmerc_scope6_20261023`; log: `controller_scope6.log` in the remote task root.
All23 tests also passed in the actual remote cmerc-repro environment. No files
or checkpoints were deleted. IEMOCAP6 and remaining Nested jobs continue through
the reduced queue's existing GPU/RAM/disk admission, not a new training protocol.
