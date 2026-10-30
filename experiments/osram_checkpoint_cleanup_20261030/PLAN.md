# Bounded obsolete checkpoint cleanup, 2026-10-30

User authorization: “行先清掉，新miss也可以删”. This overrides the prior
keep-all-new-Missing rule only for the seven families listed below, not for the
whole project. This is storage cleanup, not model/code refactoring or training.

1. Inventory exact owned regular model files in seven explicitly named families.
2. Preserve original Flat controls, current baselines, Nested128 winner, original
   Nested representative, MOSI head48 and standard Memory baseline. Protect
   additional externally referenced parent/checkpoint dependencies and live runs.
   Retain step0.4 and the MOSEI seed66 Direct model as comparison/diagnostic
   representatives. Independent read-only audit confirmed these exclusions.
3. Require saved score metadata for every eligible run. Do not select incomplete
   unscored runs, source/controller snapshots, masks, predictions or features.
4. Run the scoped safety tests. Save exact file/stat/SHA256 inventory and hash all
   non-deleted files in these families, including retained model weights.
5. Recheck inventory, live-process exclusions, dependency fingerprints and retained
   evidence before deletion. Unlink exact approved files only; no recursive or
   wildcard deletion. Journal every deletion and verify evidence afterward.
6. Record actual count/bytes, excluded runs and irrecoverability; retain code,
   scores, labels, predictions, masks, logs and configuration. Commit only this
   task's utility/tests/compact receipt/report, preserving unrelated changes.

Families: osram_v3_three_seed_20261021,
osram_v3_connection_screen_20261022, osram_v3_paired_interfaces_20261022,
osram_source_role_direct_cross_20261020, osram_mosi_memory_grid_20261022,
osram_nested_cmerc_tuning_20261023,
osram_reg_only_hparam_sweep_20260919_100.

All other families, including the current Frozen Nested three-seed results and
MOSEI frozen Nested LR/capacity experiments, are out of deletion scope. Deleted
weights cannot be loaded or resumed afterward; retained evidence is not a
substitute for weights. No trash/backup copy is retained, to actually free space.

Dependency discovery supplements the current-repository read-only audit with
non-snapshot output records dated 2026-10-20 onward (plus non-dated non-code
registries). This does not claim static resolution of arbitrary Python paths or
archived source copies. Snapshot/code/cache trees are not deletion candidates
and are excluded from the evidence-hash walk; scientific run artifacts are hashed.
