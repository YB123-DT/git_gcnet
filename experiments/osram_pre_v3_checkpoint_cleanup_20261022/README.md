# Superseded pre-V3 checkpoint cleanup

User authorization2026-10-22: delete model checkpoints from the two superseded
pre-V3 versions, not current V3 or accepted Flat baselines.

Scope is restricted to exact paths in PLAN.json under these two biggpu roots:

- /data2/yb/remote_experiments/osram_source_relation_attention_20261021
- /data2/yb/remote_experiments/osram_source_role_tokens_20261021

Delete only81 inventoried best_miss_0p0..0p7.pt and last.pt model/recovery files
in their formal runs, smokes and the SourceRelation epoch100 recovery copy.
Preserve all masks (including .pt masks), predictions, labels, logs, configs,
source snapshots, JSON/CSV/Markdown reports and all other experiment roots.
Neither experiment has a live worker/controller using its paths. The first
has completed recovered runs despite its old failed-controller ledger; the
second was explicitly cancelled by the user after MOSI150/MOSEI20.

Before unlinking any file: validate the entire exact path list, allowed parent,
regular-file/no-symlink/no-hardlink status, inode/size/mtime and absence of active
experiment processes. Save SHA256 and byte size for every removed checkpoint.
Never recursively delete a directory or use a deletion wildcard. Afterward:
verify every target absent and every retained-evidence hash unchanged.

This is permanent removal to release disk space; no trash/recovery copy is
retained. Old results remain auditable, but these old checkpoints cannot be
loaded or resumed afterward. Current V3 and Flat remain outside the delete set.
The algorithm is not labelled mathematically invalid by this cleanup; these are
the versions the user superseded because they did not match the intended design.
