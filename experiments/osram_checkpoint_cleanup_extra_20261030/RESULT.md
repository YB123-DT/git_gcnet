# Additional checkpoint cleanup and storage explanation

Completed on biggpu, 2026-10-30. Removed **332 further obsolete model/recovery
files**, **30,439,314,344 logical bytes (28.349GiB)**. Allocated bytes removed:
30,440,919,040. Deleted paths have no retained recovery copy. Existing scores and
predictions remain usable, but deleted variants cannot be freshly evaluated or
resumed without retraining.

Together with the preceding seven-family cleanup, these two passes removed
**1616 weights / 164.074GiB**. This total excludes earlier Signature/NeuMiss and
pre-V3 cleanups. `remote_experiments` now displays **684G**; `/data2` has about
**498.35GiB** free (`df -h` rounds to499G), 86% used. Global free space is affected
by concurrent work; attributed removal is measured from the exact files above.

## Why the experiment tree is large

The inventory immediately before this extra pass reported712G for the tree.
Allocated file classification is saved in `SPACE_BEFORE.json`:

| Category, before this pass | GiB |
|---|---:|
| Recognized model checkpoint filenames | 451.230 |
| Data/features/masks/other tensor files | 146.670 |
| Source snapshots including copied historical artifacts | 70.803 |
| Compressed prediction or feature arrays | 4.007 |
| Code, logs, text and other files | 38.018 |

File totals exclude directory metadata and symlinks. The146.670GiB tensor bucket
can contain unrecognized model names; it is not all regenerable cache and must
not be erased wholesale. Source snapshots contain about1.19million files and
can include copied results rather than just Python source. Nothing in these
non-checkpoint categories was authorized for deletion in this pass.

Storage multiplies across eight per-rate BEST selections, three seeds and five
folds:120BEST files plus15LAST files for one five-fold configuration. Newer full
checkpoints also contain Adam/RNG/history, not only model weights. For example,
`experiments/osram_text_path_router_joint_20261019/run.py::checkpoint_payload`
copies both `model.state_dict()` and `optimizer.state_dict()` into each saved
payload. FP32 Adam moments can add approximately twice the parameter storage.
Repeated searches leave many such complete configurations behind.

Suggested policy for future runs (not implemented by this cleanup): use
weights-only BEST for inference and a separate complete LAST for resumability;
retain selected configurations and their parents, and retire rejected HPO
weights after preserving scores/masks/predictions. Do not rewrite existing
protected baseline files or hashes under this suggestion.

## Actual deletion scope

| Family | Deleted files | GiB |
|---|---:|---:|
| MOSEI V3 capacity | 18 | 2.723 |
| MOSEI V3 grid | 36 | 4.533 |
| MOSI hparam verify | 40 | 1.072 |
| MOSI V3 capacity | 54 | 6.649 |
| MOSI write-gate/gap-strength | 120 | 6.042 |
| No-JEPA Flat2400, 200epoch retry | 24 | 2.749 |
| No-JEPA Flat2400 | 40 | 4.581 |
| V3 classification (candidate scope only) | 0 | 0 |

The11 V3 classification runs remained protected by external diagnostic
references. Two unscored retry runs (seed69/70) were not deleted. Flat plain/
small_flat controls, canonical baseline ancestors and chosen MOSI f512_v128 were
retained. Current Frozen Nested, original Flat parents, Conditional Reread,
Exemplar and B6 were out of this deletion scope. All feature/data/code/cache
directories and all saved score/prediction/mask evidence remain.

## Correctness and audit

- Reused the previous exact-file safety executor rather than creating another
  deletion engine. The small new driver selects exactly eight extra families
  and restores the previous module globals even on errors.
- **20 scoped tests passed locally and remotely**: original16 plus four new
  scope/ancestor/context-restoration/real-temporary-unlink tests. Compilation and
  whitespace checks passed. No GPU, new dependency or training was used.
- Executor compared full SHA256 for **1887 retained files before and after
  deletion**, and81 external dependency records, all unchanged.
- Independent audit verified all332 targets absent; journal and manifest paths,
  counts and byte totals agree; all1887 retained files present at recorded sizes;
  all81 dependency fingerprints independently checked; final manifest and driver
  hashes match. No training process was killed, paused or migrated.

Remote root:
`/data2/yb/remote_experiments/osram_checkpoint_cleanup_extra_20261030/`.

- `PLAN.json`: final eight-family exact-path/stat/SHA manifest.
  SHA256`36eeaf9a63432bd0d99ce7a94cf5fc96657ec11c44f531dcdaa610062cc28020`.
- `RAW_PLAN.json`: intermediate inventory from the reused engine, not the
  authoritative follow-up authorization. The final manifest adds the driver
  hash and explicit eight-family scope.
- `receipt/DELETED.jsonl`: fsynced deletion journal.
  SHA256`bb7b7c7518e3599bf1c90061d0aaa54bda18b34f5fe595975430c2427952add2`.
- `receipt/RECEIPT.json`: completion receipt, mirrored here.
- `code_v1/`: immutable driver + previous executor/helpers + tests.
  Driver SHA256`0710f33bd17db5e55c286abd53efe17c3916b9c324d999a7fe95ba6fe220d02c`.
  Reused executor SHA256`b9230c41c2638c9f47b7be0496fca02067d5dae4eed0362350f3a9ecc3b59b59`.
- `execute.log`: completion output; persistent tmux runner was
  `checkpoint_cleanup_extra_20261030`.

Base repository source:8646e0ea422da26553f884caf437771db8239671; the new driver is
pinned separately by hash. Full manifests/journals remain remote, not weights
or arrays in Git. Limits remain those of the reused engine: static dependency
inspection is not a proof about arbitrary historical dynamic paths. No model
performance or project-wide training regression was claimed or tested.
