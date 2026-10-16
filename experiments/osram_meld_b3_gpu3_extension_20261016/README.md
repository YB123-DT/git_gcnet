# GPU3: four additional B3 structures

INTERNAL TEST-ORACLE SCREEN — MELD, seed66 only.

Adds exactly four independent configurations to the original twenty, for 24 combined.
The original controller, SQLite study, queue and running GPU2 jobs are never modified.
Freeze all twenty original configurations, including pending ones, in `REFERENCE.json`;
import completed original results as `reference_only` Optuna trials. These are evidence,
not newly trained runs. Sample four unique remaining structures using immediate TPE
(`n_startup_trials=5`, `constant_liar=True`, `multivariate=False`), no new warm-start jobs.

Only `bottlenecks=1..8` and `rounds=1..4` vary. B3 dimension128, original Flat1600,
Local Skip, CHM, batch32, lr0.001, weight_decay0.00001, dropout0.5, 50epochs and
`gcnet-random-mask-v1` remain unchanged. Test-best ACC and matching UA are distinct
from common-mask paired ACC/UA. Objective remains `0.6*paired_delta_ACC + 0.4*paired_delta_UA`.

Workers run from immutable numerical source `737fdb16e5b4f4ee7a426a0979efbc7f59c5ed8c`;
the separate controller commit is recorded in `PREFLIGHT.json`. Physical GPU3 only,
UUID `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`, up to four total GPU processes.
Reuse the verified same-source largest-structure (8tokens/4rounds) smoke, no new smoke.
Resource admission may reduce concurrency; it never changes batch size or structure.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -m experiments.osram_meld_b3_gpu3_extension_20261016.run
```

Use tmux for persistence. Repeating the same command restores the new study/queue
and never requests more than four new runs. `PLANNED_JOBS.json` records unique jobs
`b3_gpu3_add_01` through `04`. Summaries retain complete worker Test-best, paired
on/off, per-rate, class recall/support, correction/harm, costs and provenance records.
`RESULT.md`, `SUMMARY.json` and `RESULT.csv` are new-run-only; reference trials are
explicitly excluded from completion counts and rankings. No performance gain is
claimed by implementation or launch success.
