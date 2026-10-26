# Signature / NeuMiss model-checkpoint removal

Explicit user authorization2026-10-26 supersedes the earlier keep-all-new-Missing
rule only for Signature/NeuMiss model weights. Preserve every Flat/Nested/CSDI/
Ridge checkpoint and all code, scores, masks, predictions, histories and logs.

Inventory exact model paths under the enumerated Signature/NeuMiss experiment
roots, plus their named subruns in four-block/three-block MOSI and Signature
NoArea serial/parallel variants. Never remove an experiment directory, source
snapshot, feature cache, dataset or generic `.pt` file by extension. Only the
explicit best/last model filename patterns in method-confirmed run directories
qualify. Preserve symlinks/hardlinks, unknown formats and active-run outputs.

Current frozen Nested3seed queue depends on112 exact Signature selected BEST
paths (122.31MiB). They cannot be deleted while pending jobs/recovery still call
the reference checkpoint verifier. Save these as a separate deferred list;
delete only after the exact128-job Nested queue and its final192-selection
collector are complete. A persistent bounded watcher may execute only these
same inventoried files after checking their stat/SHA again, not discover new
targets. If the queue fails or evidence changes, retain them.

Before irreversible removal, save per-file absolute path, run, size, stat,
SHA256 and classification. Hash all retained evidence in affected run dirs.
Validate the entire selected list, process/output/open-file exclusions and
evidence before the first unlink. Journal each exact unlink with fsync. Verify
targets absent, all retained evidence hashes unchanged, and Nested live workers
unaffected. Record logical and allocated removed bytes separately from global
disk-free changes (other jobs continue writing).

No trash copy is retained because it would not release space. Removed model
states cannot be loaded/resumed afterward; preserved scores remain auditable.
This releases disk storage, not GPU VRAM or process RAM. Commit only code,
manifest, receipts and small statistics to the user's branch, not model files.
