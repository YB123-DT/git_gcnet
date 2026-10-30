# Additional bounded checkpoint cleanup

2026-10-30 user asks to remove more obsolete checkpoints after the first135.725GiB
cleanup. This authorizes the following eight extra families, not the entire tree:

- osram_v3_classification_20261021
- osram_mosi_v3_capacity_20261022
- osram_mosei_v3_grid_20261022
- osram_mosei_v3_capacity_20261022
- osram_mosi_hparam_verify_20260918
- osram_mosi_write_gate_gap_strength_20260920
- osram_nojepa_capacity2400_20260919
- osram_nojepa_capacity2400_200_20260919_retry

Reuse the previous tested exact-manifest executor, source/dependency exclusions,
owned regular-file/stat/SHA guards, live-run checks and retained-evidence hashing.
No directory, snapshot, feature, mask, prediction, score, config or log deletion.
Retain Flat-only plain/small_flat controls, cfg01/cfg11/cfg84/cfg88 baseline
ancestors, and selected MOSI f512_v128 anchor. Current Flat/Frozen Nested parents
and results remain outside this scope. Unknown/unscored runs remain protected.

Order: add scope regression tests; verify; inventory exact targets and retained
files; inspect the manifest; execute in persistent tmux; independently audit
absence/preservation and report actual bytes. No extra training or GPU use.
Deletion is permanent. Some archived V3 attention diagnostics cannot be repeated
after removal; their saved outputs and the selected representative remain.

Conditional Reread, Exemplar, B6 and current accepted Nested families are NOT in
this pass. Do not reinterpret this as permission to erase all old variants.
