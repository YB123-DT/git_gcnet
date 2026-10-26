# Signature / NeuMiss checkpoint cleanup

2026-10-26, biggpu. **Both removal phases complete.** Initial1819 removals are
recorded below; the remaining112 were removed at07:17:44UTC after successful
Nested finalization. Total1931 weights /27569302071logical bytes, about25.676GiB.
User explicitly requested deletion to release storage; only these methods are
exempted from the prior keep-new-Missing-checkpoints instruction.

| Item | Actual verified result |
|---|---:|
| Model best/intermediate/recovery files removed |1819|
| Logical bytes removed |27441052711 (25.5565GiB)|
| Allocated bytes removed |27446284288|
| Retained evidence files hashed before/after |3638, all unchanged|
| Signature BEST retained for current Nested |112,128249360bytes (122.31MiB)|
| Unexpected skips or failures |0|
| Exact immediate target paths confirmed absent |1819/1819|

## Deferred phase completed,07:17:44UTC

Nested128/128 training branches completed, then all192 total selected artifacts
were verified. A collector-only legacy-field bug was repaired without any
retraining or model/mask changes (see Nested POSTPROCESS_RECOVERY.json). Once
QUEUE and RESULT both reported complete with the original training source, the
existing watcher removed the exact112 planned Signature references. All3638
retained-evidence hashes were checked again and remain unchanged. The watcher
exited successfully; `DEFERRED_RECEIPT.json` is the actual completed receipt.
These112 weights are now also permanently unavailable. Flat/Nested remain.

Removed Signature/NeuMiss frozen and joint training weights, including Signature
NoArea and serial/parallel variants, with their best/intermediate/last states.
Kept all code/source snapshots, scores, exact masks (including `.pt` masks),
predictions, labels, histories, logs and feature/support caches. No Flat, Nested,
CSDI or Ridge model was a target. For an independent example after cleanup,
the joint MOSI Flat folder still has96 `.pt` files and CSDI93.

Deletion was permanent: no trash/recovery copy was retained. These removed
weights can no longer be loaded or resumed. Historical scores/evidence remain
auditable; old reports mentioning retained weights are superseded by this dated
cleanup record. This releases **disk storage**, not GPU VRAM or process RAM.

## Why112 files remain temporarily

The unchanged, actively running frozen Nested3seed runner verifies each completed
same-seed Signature reference BEST SHA when starting/resuming a branch. Deleting
these files early would break pending Nested jobs even though their parameters
are not used for Nested initialization. Protect exactly112 reference paths, not
all checkpoints or whole directories. The final Nested collector itself does
not require Signature weights.

At the initial capture a persistent CPU-only watcher ran in tmux
`sig_neu_cleanup_20261026`, PID1314272. It waited for the original128-job Nested queue
to report complete and the matching-source final collector to validate192 total
selections. Only then may it remove the original112 inventoried files, after
checking stat/SHA, process exclusions and all retained evidence again. It cannot
discover or delete future new models. The48h deadline fails closed: if completion
is not verified or evidence changes, dependencies stay in place. No GPU is used.

Post-removal Nested state was train,79complete/12running/37pending, with no error.
No training process was killed, moved or restarted by this cleanup.

## Reproduction and audit locations

Implementation source97eca5bb4f76dc9afff2d94791dbb5e6e1c8edcf; executor SHA256
`8084da39a9faa6bd6e34ec62f246959df0a806fc64d43013753f972e880c42c1`.
Local and remote CPU checks:6/6passed, including an actual temporary-file unlink
that preserves masks/deferred weights; no model or environment changes.

Remote directory:
`/data2/yb/remote_experiments/osram_signature_neumiss_cleanup_20261026/`.

- `PLAN.json`: exact1931 model paths/stat/size/SHA and3638 retained-evidence hashes.
  SHA256`9d879961708c9da0b97d786b84698168c78ccb9676213bf0a4b3feda42337f2a`.
- `immediate_receipt/DELETED.jsonl`: fsynced per-file removal journal.
  SHA256`ec6767b723c873b1df9a09c4f6ed3f01a7ad5fb1d5b3c378d9d16564b3a1f4f9`.
- `immediate_receipt/RECEIPT.json`: original complete receipt; reproduced in Git.
- `deferred_watcher.log`: waiting/completion output.
- `deferred_receipt/RECEIPT.json`: successful deferred deletion at07:17:44UTC,
  copied to Git as DEFERRED_RECEIPT.json. The earlier AUDIT.json is a historical
  waiting-state capture, not current status.

`AUDIT.json` preserves verified watcher command/PID, target-absence checks and
Nested progress. Full2.13MB manifest/journal stay remote with SHA anchors; Git
contains only code, tests, plan and compact receipts, not model/data arrays.
Global free space changes also include concurrent training writes; report exact
unlinked bytes rather than claiming the whole disk-free delta is this task.
