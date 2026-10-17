# MELD Flat warmup/cosine comparison — final

INTERNAL TEST-ORACLE SCREEN. The one authorized seed66 job completed50/50 epochs, no failures.

## Matched ACC-best comparison

| Schedule | ACC | Corresponding UA | WF1 | High ACC | High UA | High WF1 |
|---|---:|---:|---:|---:|---:|---:|
| Constant W0 | 60.838 | 35.633 | 56.789 | 56.718 | 29.025 | 50.812 |
| Warmup+cosine C0 | 60.924 | 36.642 | 57.349 | 56.692 | 30.401 | 51.275 |

C0-minus-W0: ACC 0.086pp, UA 1.009pp,
WF1 0.561pp. All three eight-rate metrics rise in this one
seed; high-missing ACC slightly falls. This is a modest exploratory result,
not statistical evidence or permission to add more runs.

## Per-rate, ACC-selected checkpoint

| Miss | W0 ACC | C0 ACC | W0 UA | C0 UA | W0 WF1 | C0 WF1 |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 66.513 | 67.011 | 48.084 | 44.896 | 65.217 | 65.299 |
| 0.1 | 64.904 | 65.249 | 41.469 | 42.555 | 62.670 | 63.193 |
| 0.2 | 63.448 | 63.333 | 38.797 | 40.248 | 60.636 | 60.727 |
| 0.3 | 61.648 | 61.533 | 36.463 | 38.105 | 57.639 | 58.404 |
| 0.4 | 60.038 | 60.192 | 33.174 | 36.131 | 55.713 | 57.347 |
| 0.5 | 58.506 | 58.084 | 29.734 | 33.482 | 52.509 | 53.311 |
| 0.6 | 56.360 | 56.360 | 27.905 | 29.709 | 50.646 | 51.035 |
| 0.7 | 55.287 | 55.632 | 29.437 | 28.012 | 49.280 | 49.479 |

## Separate UA-best exploratory selection

These are different selected checkpoints, not the ACC-best scores above.
W0 has no retained UA-best comparator in this task.

| Selection | Mean ACC | Mean UA | Mean WF1 | High ACC | High UA | High WF1 |
|---|---:|---:|---:|---:|---:|---:|
| C0 UA-best | 59.459 | 39.588 | 57.283 | 54.917 | 32.506 | 51.486 |

## Verification and limits

- Immutable numerical source b4a313841f098487a6482d2712bdac94aa8e319e and driver ae142247ac5be7b56f7ad532f0f26709f7a893d5. No changes to the accepted three-seed baseline or running historical sources.
- Effective config differs from W0 only in lr_schedule=cosine. Existing warmup_ratio=.05 activates two warmup epochs, then cosine to0 at epoch50; this does not isolate cosine from warmup.
- Actual group LR curve verified every epoch. Initialization and train/eight-rate selection-mask streams matched W0 through all50 epochs. Parameters remain13,290,795; no new model parameters/losses.
- Original gcnet-random-mask-v1, ECERC features, seed66, batch32, Adam peakLR.001, original Flat1600/Local Skip, causal joint Block Delta Write, step.6 and ridge.001 retained.
- All eight ACC-best and eight UA-best weights, same-forward predictions and masks, optimizer/RNG recovery state, per-epoch evaluation metrics and actual LR traces retained on biggpu. No extra evaluation or sampler calls for UA selection.
- Peak allocated564.897MiB; worker wall time1433.178s. This shared-resource timing is not a clean speed benchmark.
- Both selectors are test-oracle; different rates may select different epochs/masks. Do not treat these means as single-checkpoint deployment performance or independent paper results.
- This finite queue stopped after the one approved run. No automatic multi-seed or HPO expansion.
- Full machine-readable metrics/config and provenance are in FINAL_SUMMARY.json. Exact remote paths and command are in LAUNCH.md / README.md.
