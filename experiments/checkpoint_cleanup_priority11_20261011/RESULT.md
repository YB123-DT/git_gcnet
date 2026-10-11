# Cleanup completed

biggpu2026-10-11: deleted379explicit weight files from the11completed runs in
PLAN.md,20.478 GiB logical bytes. All1266retained run evidence files passed
post-deletion SHA-256 checks. No experiment directory or non-weight file was
deleted. Every targeted checkpoint path is now absent; script exited0.

Free /data2 space increased from27.355 to47.835 GiB. /data1 remained about
46.3 GiB (minor unrelated changes). Exact byte counts in SUMMARY.json.

The archival MANIFEST records paths, hashes, sizes and pre-deletion inode/device
identities for both deleted weights and preserved evidence. Existing run
provenance retains its historical completion state; this cleanup record
supersedes any assertion that deleted weight artifacts remain available.

No backups of these weights were made. They require external copies or retraining
for future inference/resume. Baseline, Nested, high-score protected candidates,
multi-seed and150epoch weights were outside the explicit deletion allowlist.
No new training/inference was launched by this operation.
