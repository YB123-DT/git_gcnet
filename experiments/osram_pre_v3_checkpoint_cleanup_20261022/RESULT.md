# Superseded pre-V3 weight cleanup completed

User-authorized permanent removal completed on biggpu2026-10-22.

| Version | checkpoint files deleted |
|---|---:|
| SourceRelation early-Gap-mean, including smoke and epoch100 recovery |45|
| 12-token SourceRoleTokens, including smoke |36|
| Total |81|

Deleted14221067270logical bytes (13.244GiB). No trash/recovery weight copy
retained. All189 inventoried evidence files passed unchanged SHA256 checks,
including fixed_masks.pt, predictions, configs, histories and analysis reports.
Code/source snapshots remain. No model checkpoint remains in the two scoped
version roots. Current V3/Flat/Nested and other experiment roots were not targets;
no training process was stopped or signalled.

An initial safety check rejected JSON-rounded nanosecond timestamps before any
unlink. PLAN.json now uses decimal strings, preserving exact integer times.
All81 targets were validated again before the first deletion.

PLAN.json records per-file SHA256/bytes/inode/time and retained-evidence hashes.
RECEIPT.json records every actual unlink and post-cleanup verification. Historical
scores remain internal Test-oracle diagnostics; cleanup does not establish any
scientific mechanism or model validity claim. These two versions can no longer
be loaded/resumed for fresh inference; saved predictions can still be analyzed.
