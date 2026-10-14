# Width512 three-seed extension — COMPLETE

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

## Completed results

All new runs completed50epochs and retained8best checkpoints. Exact selection
mask recovery completed for all24width512 seed/rate points. ACC/W-F1 reproduce
stored checkpoint selection scores, and UA uses those same checkpoints/masks.
No protocol, batch size, optimizer or missing algorithm changed.

| Seed | Flat1600 ACC | Flat512 ACC | Flat1600 UA | Flat512 UA | Flat1600 W-F1 | Flat512 W-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 66 | 60.675 | 60.766 | 35.436 | 37.359 | 56.807 | 57.150 |
| 67 | 60.603 | 60.330 | 35.830 | 35.961 | 56.889 | 56.300 |
| 68 | 60.800 | 60.915 | 37.498 | 36.411 | 57.438 | 57.116 |
| Mean | 60.693 | 60.670 | 36.255 | 36.577 | 57.045 | 56.855 |

These are per-rate Test-oracle ACC-selected eight-rate means, internal only.
Flat1600 is the new original-code reproduction, not a replacement of archived
historical scores or restoration of the deleted original parameters.

Width512 minus1600: ACC-0.022pp, UA+0.322pp, W-F1-0.189pp. Seed66's UA gain
is not consistent across seeds: +1.923/+0.130/-1.087pp. There is no supported
claim of uniform improvement. High-missing0.5/0.6/0.7 results:

| Width | ACC | UA | W-F1 |
|---:|---:|---:|---:|
| 1600 reproduced | 56.484 | 30.157 | 51.066 |
| 512 | 56.667 | 28.901 | 50.407 |
| Delta512−1600 | +0.183 | -1.256 | -0.658 |

Full new width512 per-rate scores, epochs and hashes:
`SEED67_SUMMARY.json`, `SEED68_SUMMARY.json`; seed66 remains in
`../osram_meld_testbest_ua_20261014/SUMMARY.json` (source ending `/width512`).
Reproduced Flat1600 seed67/68 records are
`../osram_meld_testbest_ua_20261014/FLAT_RESTORED67_SUMMARY.json` and
`FLAT_RESTORED68_SUMMARY.json`. Both baseline restores also completed exact
UA recovery. No model search was launched by the separate Optuna installation.
