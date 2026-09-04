# Target-private rank causal A/B on CMU-MOSI

## Question

Does routing six source-to-target prediction directions through fully shared MMoE experts hurt downstream sentiment performance, despite their shared-expert gradients being mostly cross-target orthogonal?

The control uses `target_private_rank=0`. The treatment uses `target_private_rank=32`, which adds a target-specific low-rank residual inside each expert. No optimizer surgery or other model change is included.

## Locked protocol

- Dataset: CMU-MOSI
- Training: one stratified mixed-rate model per seed
- Seeds: 66, 67, 68, 69, 70
- Test rates: 0.0 through 0.7
- Checkpoint selection: validation weighted F1
- Paired masks: identical within every seed and rate (SHA256 verified)
- Only experimental variable: `target_private_rank` 0 versus 32
- Additional treatment parameters: 49,152

The predeclared sign gate required at least 4/7 positive nonzero-rate means, at least 3/5 positive seed-level nonzero macros, miss-0 mean delta no worse than -0.005, and no collapse.

## Results

All values are five-seed test weighted F1 percentages. Delta is rank32 minus rank0.

| Missing rate | Rank 0 | Rank 32 | Delta (points) | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 85.485 | 85.807 | +0.322 | 4/5 |
| 0.1 | 83.102 | 83.670 | +0.569 | 3/5 |
| 0.2 | 80.922 | 80.776 | -0.146 | 3/5 |
| 0.3 | 79.622 | 79.869 | +0.247 | 4/5 |
| 0.4 | 76.598 | 77.123 | +0.525 | 2/5 |
| 0.5 | 74.819 | 74.148 | -0.671 | 2/5 |
| 0.6 | 74.436 | 73.891 | -0.546 | 2/5 |
| 0.7 | 71.857 | 71.909 | +0.052 | 2/5 |

The seed-level macro deltas over rates 0.1--0.7 were:

| Seed | Delta (points) |
|---:|---:|
| 66 | -1.298 |
| 67 | +0.808 |
| 68 | +0.276 |
| 69 | -2.329 |
| 70 | +2.564 |

Their mean was **+0.004 points** (raw F1 delta `+0.0000418`), with sample standard deviation **1.898 points** and an approximate 95% t interval of **[-2.352, +2.361] points**.

## Verdict

The treatment technically passes the predeclared sign gate: 4/7 nonzero rates have positive means, 3/5 seeds have positive nonzero macros, miss-0 improves by 0.322 points, and no run collapses.

However, the causal effect estimate is effectively zero. The nonzero-rate macro changes from 77.337% to 77.341%, while seed variation is about 450 times larger than the mean gain. The signs also reverse at rates 0.5 and 0.6, where target-private capacity performs worse.

Therefore this A/B **does not support the hypothesis that shared-expert cross-target gradient orthogonality is causing a material downstream performance loss**. The earlier geometry finding is real—same-target shared-expert gradients align more than cross-target gradients—but forcing explicit target-private rank does not reliably improve sentiment prediction. Near-orthogonality is better interpreted as task specialization or weak interaction than as demonstrated destructive interference.

Do not promote rank32 to the main method and do not expand it to other datasets. A parameter-matched capacity control is unnecessary unless a future experiment produces a nontrivial, stable positive effect.

## Artifacts

- Machine-readable paired summary: `target_private_ab/paired_summary.json`
- Per-run configuration, history, metrics, and logs: `target_private_ab/raw/`
- Runner: `scripts/run_mosi_target_private_ab.py`
- Shared-expert direction geometry: `SIX_DIRECTION_GEOMETRY.md`

