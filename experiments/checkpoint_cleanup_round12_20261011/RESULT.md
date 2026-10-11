# Approved23-run cleanup complete

Completed biggpu2026-10-11, script exit0. Only the12round1 and11round2 candidate
allowlist was targeted, not the full batches. Deleted839checkpoint files,
49,937,752,235logical bytes (46.508 GiB). All2629retained run evidence files
were present and SHA-256-identical after deletion. Every target path absent.

Free /data2 space:51,361,906,688→101,301,620,736bytes (about94.35 GiB now).
Exact storage and file evidence in SUMMARY.json and MANIFEST.json. Existing
source snapshots, configs, histories, logs, metrics, predictions and reports
were retained. All other models were outside the allowlist.

Deleted weights have no backup created by this operation; manifest hashes
cannot restore them. Future inference/resume requires an external copy or
retraining. Historical run provenance remains unmodified as evidence of original
completion; this cleanup manifest supersedes old checkpoint-presence claims.

Verification: Python compile, complete/outputs_verified on23runs, existing
config/metrics and8predictions per run, live Python command exclusion, no
symlinks, per-file inode/content match before unlink, retained evidence hash
check after deletion. No new training or inference launched.
