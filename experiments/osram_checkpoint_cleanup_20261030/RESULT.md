# Obsolete checkpoint cleanup — completed

2026-10-30, biggpu. User explicitly allowed obsolete **new-Missing** weights to
be removed too. Only the seven previously listed experiment families were in
scope. This is disk cleanup, not training, RAM cleanup or a model modification.

Removed **1284 model checkpoint/recovery files**, 145,733,509,665 logical bytes
(**135.725 GiB**). Allocated bytes removed: 145,738,895,360. No trash/backup copy
was retained: deleted weights cannot be loaded or resumed. Historical scores do
not imply their weights are still available.

| Family | Files removed | Logical GiB |
|---|---:|---:|
| V3 three-seed, 20261021 | 216 | 33.873 |
| V3 connection screen, 20261022 | 108 | 17.662 |
| V3 paired interfaces, 20261022 | 108 | 16.955 |
| Source-Role Direct cross-dataset, 20261020 | 162 | 24.784 |
| MOSI Memory grid, 20261022 | 135 | 22.167 |
| Early Nested cmerc tuning, 20261023 | 27 | 4.411 |
| Early reg-only hyperparameter sweep, 20260919 | 528 | 15.872 |

## Preservation and verification

- All **3514 inventoried retained files** passed full SHA256 comparison before
  and after removal: scores, configurations, logs, labels, predictions, exact
  masks and retained models. Six external dependency records also remained
  unchanged. Code/snapshot/cache directories were excluded from deletion and
  from this evidence-hash walk; they were not removed.
- Kept Memory-grid formal `baseline`, `head_48`, `step_0p4`; original Flat controls
  and representative Nested dim32/dim128/MLP128/depth2/default configurations in
  the early tuning family; canonical old cfg01/cfg11/cfg84/cfg88; all three MOSEI
  Direct models (seed66 diagnostic representative, seed67/68 external references).
- Kept the current Frozen Nested three-seed, MOSEI frozen LR/capacity experiments,
  their parent Flat checkpoints and all other families outside this finite scope.
  One incomplete/unscored early IEMOCAP groups4 run was excluded, not silently
  removed. No training process was killed, paused or migrated. Read-only
  inventory attempts were interrupted while correcting archived-source traversal;
  no model file was removed by those attempts.
- Local and remote scoped safety tests: **16 passed** each. Includes real unlink
  in temporary test directories, filename/scope exclusions, changed-evidence
  fail-closed behavior, control/winner protection, and archived-source exclusion.
- Independent post-removal audit: all1284 exact paths absent; journal/manifest/
  receipt counts, path sets and logical bytes agree; all3514 retained files still
  present at their inventoried sizes; six dependency hashes independently checked.
- Disk free at execution start/end: 358,916,857,856 / 504,655,044,608 bytes.
  Subsequent `df -h /data2`: about **470G available**, 87% used. Concurrent work can
  change global free space; removed-file totals above are the task attribution.

## Audit locations

Base source commit: `c5c28695215f767a218e1cdcc6869f2010a5cba6`; the new utility is
pinned by its complete file hash, not claimed to be part of that prior commit.

Remote root: `/data2/yb/remote_experiments/osram_checkpoint_cleanup_20261030/`.

- `PLAN.json`: exact owned regular-file paths, stat, SHA256, retained evidence,
  dependency records, protected/skipped runs. SHA256:
  `59f48349464d806e12c7453c676443705408926b09093f96c308862cb4bed2f6`.
- `receipt/DELETED.jsonl`: fsynced per-file deletion journal. SHA256:
  `8b7510d7cc3f2894466c408461d2f8f0d63cd0a41a028bae6bfa53df6cfbfc10`.
- `receipt/RECEIPT.json`: completed receipt, mirrored here as `RECEIPT.json`.
- `execute.log`: executor completion output. Persistent executor used tmux
  `checkpoint_cleanup_20261030`; no GPU was requested.
- `code_v1/`: exact utility, reused previous cleanup safety helper and tests.
  Utility SHA256:
  `b9230c41c2638c9f47b7be0496fca02067d5dae4eed0362350f3a9ecc3b59b59`.

Only task code, tests, compact receipt and text records are committed. Full
manifest/journal stay remote; no checkpoint, dataset or feature array is uploaded.

## Limits

Dependency discovery combines an independent current-code audit, explicit
baseline/winner exclusions and bounded non-snapshot current output references.
It is not a proof about arbitrary future scripts or historical dynamic Python
path construction. Deleted variants require retraining for new inference; their
existing predictions and scores remain auditable. No project-wide model suite or
training was run because model behavior/code was not changed.
