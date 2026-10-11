# Approved checkpoint cleanup: batches2/3/4

Completed on biggpu2026-10-11. Explicit user approval covers only medium
Flat/Nested parameter sweep and the two Nested direct-capacity batches.
Batch1, original baselines, cross-dataset seeds,150epoch runs and paired-view
weights were not targeted. No training was launched by this cleanup.

All18target runs reported complete/outputs_verified and retained eight prediction
files. No live Python command referenced the target batches at preflight.
Exact paths, sizes, inode/device and SHA-256 recorded before deletion in
MANIFEST.json. Only647explicit checkpoint files were unlinked (28.486 GiB
logical bytes). No recursive directory deletion; no symlink traversal.
All2050remaining run evidence files were verified unchanged by SHA-256.
Code snapshots outside run directories remain untouched.

Free space before/after: data1 30.738→46.669 GiB; data2 14.798→27.347 GiB.
Filesystem free-space deltas can include unrelated concurrent activity.

Deleted weights were not backed up by this operation and cannot be recovered
from the hash manifest. Future inference/resume of these deleted tuning runs
requires another existing copy or retraining. Original provenance is preserved
as a historical completion record; this manifest supersedes its weight-presence
claims. Configs/logs/metrics/predictions/reports remain available.

Verification: cleanup script exited0; SUMMARY status=complete; all targeted
paths absent; every retained evidence hash unchanged; Python compile check.
