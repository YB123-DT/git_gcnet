# MELD B3 structural screening on original Flat

INTERNAL TEST-ORACLE SCREEN — seed66 exploration, not independent test performance.

This finite run adds the existing `mbt_role_bottleneck` before the **original Flat1600**.
The original Local Skip uses unmodified Local. CHM Key/Value, joint Block Write,
read-before-write, Query, task head, loss, and GCNet missing sampler stay unchanged.

The five initial jobs run concurrently on **biggpu physical GPU2**:

| Trial | Bottleneck tokens | Synchronous exchange rounds |
|---|---:|---:|
| 0 | 4 | 2 |
| 1 | 2 | 2 |
| 2 | 8 | 2 |
| 3 | 4 | 1 |
| 4 | 4 | 3 |

All five initial jobs must finish before adaptive TPE begins. The complete budget
is **20 unique configurations total, including these five**, not 25. TPE searches
integer token counts1–8 and rounds1–4, with at most five pending jobs. Model width
is fixed128; lr0.001, weight decay1e-5, dropout0.5, batch32, seed66, and50epochs
stay at the original Flat settings. No B6-tuned training hyperparameters are used.

## Selection and saved evidence

Each trial retains the original per-rate Test-best ACC checkpoint and its exact
same-epoch predictions/masks; UA/W-F1 are calculated from those same predictions.
The fixed common-mask replay is reported separately. TPE maximizes
`0.6 * (paired ACC − Flat paired ACC) + 0.4 * (paired UA − Flat paired UA)` in
percentage points, using the retained original-hyperparameter Flat seed66 summary
from `osram_meld_block2_auto_20261014_v2/screen/Flat/SUMMARY.json`. This reference
is not a newly trained sixth job, and is not the historical three-seed60.693 mean.

The SQLite study, persistent queue, worker logs, effective configs, source hashes,
dataset SHA256, mask hashes, UUID, failures, Test-best metrics, paired on/off
metrics, and checkpoints remain in the independent remote output directory.
`RESULT.md`, `RESULT.csv`, `TRIALS.json`, and `SUMMARY.json` update as jobs finish.
Historical B3 scores are not counted as fresh outcomes.

## Execute (after committing an immutable source snapshot)

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m \
  experiments.osram_meld_b3_structure_hpo_20261016.run \
  --gpus 2 --per-gpu 5 --trials 20 \
  --output /home/qinjinghui/remote_experiments/osram_meld_b3_structure_hpo_20261016
```

Use the project persistent tmux launcher for unattended execution. Startup checks
the dataset hash, protected common masks, original Flat protocol, GPU2 UUID,
committed source, and free disk. One three-update real-batch smoke measures the
largest searched B3 (8tokens/4rounds) before the five formal jobs start. Runtime
capacity can decrease after one auditable OOM retry; batch size and model settings
are never silently reduced. GPU4 and other users' processes are never touched.

Restarting the identical command resumes the same study/queue; conflicting source,
config, or dataset provenance is rejected. No automatic multi-seed expansion.
