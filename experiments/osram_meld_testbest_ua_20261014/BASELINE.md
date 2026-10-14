# Current MELD Flat baseline — user accepted 2026-10-14

INTERNAL DIAGNOSTIC ONLY. The user explicitly designated the reproduced
original large Flat three-seed result as the current comparison baseline:

| Seed | Eight-rate ACC | Corresponding UA | W-F1 |
|---:|---:|---:|---:|
| 66 | 60.675 | 35.436 | 56.807 |
| 67 | 60.603 | 35.830 | 56.889 |
| 68 | 60.800 | 37.498 | 57.438 |
| Mean | **60.693** | **36.255** | **57.045** |

Protocol: original `gcnet-random-mask-v1`, 50epochs, batch32, Adam1e-3,
cyclic0.0–0.7, no-JEPA causal Flat, hidden/output1600, original Local Skip.
Per-rate Test-best **ACC** selection; UA/W-F1 use the same selected checkpoint
and its original winning-epoch masks. This is not one single checkpoint's
deployment score. Historical60.498 remains an older archived run, not the
active comparison baseline. MOSI and other dataset baselines are not changed.

## Retained checkpoint and exact selection-mask inventory

Verified on biggpu: all24model files and all24selected-mask prediction files
exist. Each source has8checkpoints, total425,523,784bytes per seed. Never apply
the old discard-screen-checkpoints cleanup to these designated baseline roots.
Weights stay on biggpu; no weights are uploaded to GitHub.

| Seed | Checkpoint root |
|---:|---|
| 66 | `/data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66_reproduced_b4a3138/baseline` |
| 67 | `/data2/yb/remote_experiments/osram_meld_testbest_ua_20261014/flat_restore/seed67/baseline` |
| 68 | `/data2/yb/remote_experiments/osram_meld_testbest_ua_20261014/flat_restore/seed68/baseline` |

Checkpoint files: `best_miss_0p0.pt` through `best_miss_0p7.pt`.

| Seed | Exact selected-mask prediction root |
|---:|---|
| 66 | `/data2/yb/remote_experiments/osram_meld_testbest_ua_20261014/exact/source_0` |
| 67 | `/data2/yb/remote_experiments/osram_meld_testbest_ua_20261014/flat_restore/seed67/testbest_ua/source_0` |
| 68 | `/data2/yb/remote_experiments/osram_meld_testbest_ua_20261014/flat_restore/seed68/testbest_ua/source_0` |

Files: `predictions_miss_0p0.npz` through `predictions_miss_0p7.npz`, containing
labels, predictions and selected availability. These are distinct from the
baseline root's final saved-mask predictions. Do not substitute final masks
or generate new evaluation masks when attempting exact selected-score replay.

Checkpoint hashes and winning epochs are retained in `SUMMARY.json`
(source ending `/baseline`, seed66), `FLAT_RESTORED67_SUMMARY.json` and
`FLAT_RESTORED68_SUMMARY.json`.

## Reproducibility boundaries

**Frozen evaluation reproduced:** all24checkpoint ACC/W-F1 values matched
stored selection scores within1e-10 on the exact original selection masks.
Corresponding UA was calculated in those same evaluations. The separate
fixed-query audit also reproduced all8seed66 original saved classes exactly.

Environment: biggpu `/data2/yb/conda_envs/cmerc-repro/bin/python`,
Torch1.13.1+cu117, NumPy1.26.4, sklearn1.3.2; original training code `b4a3138`.
Feature pickle SHA256:
`a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`.

**Same-seed retraining not proven bit-identical:** old and new configs match,
but training histories already differ at epoch1 and selected epochs differ.
Original code calls `set_random_seed` without strict deterministic algorithms.
This is a possible reproducibility factor, not a localized cause. Do not claim
that from-scratch retraining must produce exactly60.693 or that the0.195pp
old-to-new difference demonstrates a method improvement.

**Exact training resume unavailable from these best weights:** checkpoint keys
are model/config/epoch/selection metadata. They do NOT contain optimizer or
RNG state. They support strict weight restoration and evaluation, not a claim
of uninterrupted-equivalent optimizer/RNG restoration. Starting new training
from them would be a new weight-initialized run, not exact continuation.
