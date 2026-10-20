# MOSI direct-source three-seed interim progress

INTERNAL TEST-ORACLE SCREEN

Checked 2026-10-20T13:22:38.708896+00:00; queue remains training with no reported job failure.
Seed66 is complete150epochs;67/68 are unfinished. Values below are per-rate
best-so-far W-F1, not final three-seed performance. Do not compute/report a
final three-seed mean by mixing these partial budgets.

| Seed | Completed epochs | Direct eight-rate W-F1 | Direct high W-F1 | Completed Flat eight-rate W-F1 |
|---|---:|---:|---:|---:|
| 66 | 150/150 | 79.450 | 73.478 | 79.128 |
| 67 | 37/150 | 67.306 | 62.292 | 78.751 |
| 68 | 34/150 | 75.538 | 69.524 | 78.794 |

## Same completed-budget context

This uses existing Flat training histories only; no new inference/training.
Best W-F1 per rate is selected only within each stated prefix, then averaged.

| Seed | Budget | Flat same-budget eight-rate W-F1 | Direct same-budget W-F1 | Delta pp |
|---|---:|---:|---:|---:|
| 67 | 37 | 74.291 | 67.306 | -6.985 |
| 68 | 34 | 73.957 | 75.538 | 1.580 |

Seed67 is currently materially weaker, including versus its own same-budget
Flat history; seed68 is ahead of its own same-budget Flat history. These are
early seed-dependent trajectories, not evidence of a stable benefit or a proven
implementation bug. No restart, model change, LR change or new experiment was
performed. Both latest gradient checks show14connected/14nonzero source-role
parameter tensors and matching original training mask/update/LR traces.

PROGRESS.json retains all24seed/rate scores and selected epochs. Automatic
completed-only report will run after both seeds reach150 and verify selected
prediction artifacts and matched Flat cohorts/masks. Continue only the two
already authorized runs.
