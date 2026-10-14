# Width512 three-seed extension

INTERNAL DIAGNOSTIC ONLY. Authorized on 2026-10-14.

Reuse completed seed66 width512; add only seeds67/68 (50 epochs each).
Training uses frozen historical code `b4a3138`, original per-seed canonical
GCNet source config, batch32, Adam1e-3, cyclic0.0–0.7, original Local skip,
no-JEPA causal Flat. Only `osram_adapter_hidden_dim=512` changes; output1600.
Retain all eight per-rate ACC-selected checkpoints. Exact winning-epoch mask
recovery runs after training to report corresponding UA, never final-mask UA
alongside a different mask's selected ACC.

Existing seed66 output:
`/data2/yb/remote_experiments/osram_meld_flat_width_20261014/width512`.

New output root:
`/data2/yb/remote_experiments/osram_meld_flat_width_20261014_multiseed`.

Each new seed uses its same-seed historical Flat predictions for final mask
parity, not seed66's masks. This is not a deterministic restoration of the
deleted historical baseline weights. Reproduced Flat seeds67/68 are separately
running and must be labeled as reproductions when later compared.

## Independent frozen diagnostic

After starting these runs, implement one task-level four-fork replay with
fixed original target Local, Base/Gap queries and observed keys. Use ONLY
retained reproduced original width1600 Flat seed66 weights from
`osram_meld_write_retention_20261014/seed66_reproduced_b4a3138/baseline`.
Do not use width512 weights, no-skip or legacy models. Distance1 is a causal
boundary check; report distances2/4/8. Normal must reproduce original outputs;
all forks read target pre-write state and use original Flat/head without
training. Task improvement must be tested rather than inferred from source-key
association recovery. Per-source-target populations are diagnostic cohorts,
not whole-test-set model scores.

## Checks

Tests-first seed67/68 acceptance failed under the previous seed66-only guard,
then passed after extending that guard. Three config tests passed locally.
No original training, mask or backbone implementation was edited.

## Launch record

Started on biggpu healthy host GPU2, UUID
`GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`; training code `b4a3138`,
immutable runner worktree
`/data2/yb/remote_experiments/osram_meld_flat_width_20261014_multiseed_code_18a2816`.

| Seed | Persistent parent PID | Training child PID | Output |
|---:|---:|---:|---|
| 67 | 313945 | 313948 | `osram_meld_flat_width_20261014_multiseed/seed67` |
| 68 | 313947 | 313949 | `osram_meld_flat_width_20261014_multiseed/seed68` |

Output paths above are relative to `/data2/yb/remote_experiments`.
Logs are sibling `seed67.log`, `seed68.log`. Each persistent shell runs
training followed by `osram_meld_testbest_ua_20261014.recover`. Its recovery
output is sibling `seed67_testbest_ua`, `seed68_testbest_ua`.

The initial direct-file invocation failed to find the `experiments` package
before creating any run directory or performing training. Original failure
logs remain as `seed67_initial_import_failure.log` and
`seed68_initial_import_failure.log`; corrected launch uses module invocation
(`python -m experiments.osram_meld_flat_width_20261014.run`). Both corrected
jobs have model processes and checked PROVENANCE records.
