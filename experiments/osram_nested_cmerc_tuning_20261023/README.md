# Nested only: MOSEI and IEMOCAP cmerc screen

INTERNAL TEST-ORACLE SCREEN. Finite21 jobs, seed66 only; no automatic expansion.
User requested these two dataset families only. IEMOCAP4 and6 use fold5 for
initial screening, never labelled full fivefold results.

| Method | Graph dim | Depth | Groups | GIN hidden |
|---|---:|---:|---:|---:|
| none (matched Flat control) | — | — | — | — |
| nested_gnn_rooted_evidence | 64 | 3 | 8 | 64 |
| nested_dim32 | 32 | 3 | 8 | 32 |
| nested_dim128 | 128 | 3 | 8 | 128 |
| mosei_nested_depth2 | 64 | 2 | 8 | 64 |
| mosei_nested_mlp128 | 64 | 3 | 8 | 128 |
| mosei_nested_groups4 | 64 | 3 | 4 | 64 |

All aliases already exist; dataset-scoped names do not load another dataset.
Dimension variants also change the existing default GIN hidden width (it equals
graph dim when unspecified); do not call these pure dimension-only changes.
Groups alter graph tokenization, NOT the original8 CHM memory heads.
Original nested residual/zero-initialized decoders remain unchanged. No new module.

MOSEI60 epochs at constant1e-4/MSE; IEMOCAP100 at constant1e-3/CE. Both batch32,
Adam/WD1e-5/cyclic0–.7/new gcnet-random-mask-v1/no-JEPA/Flat1600/LocalSkip.
All original parameters train end-to-end. Missing samplers and CHM math untouched.
Each rate selects its strict TEST-best (MOSEI W-F1; IEMOCAP ACC, same-call UA/W-F1).
Every run retains8BEST, exact selected predictions, masks, history and last.pt.

Use protected Flat records only for config/data/masks, not initialization or paired
scores. MOSEI exact test bank comes from the just-completed cmerc66 control.
IEMOCAP masks are rich selected artifacts from the accepted cmerc66/fold5 run.
Historical IEMOCAP resampled test masks per epoch; this experiment fixes and replays
those selected masks in ALL arms, including a newly trained Flat. Hence compare
new Nested to NEW matched Flat, not a historical score as if trajectory identical.

Torch1.13 UUID-bound device enumeration is repaired using the existing
`repair_uuid_rng` utility in every arm, BEFORE model seeding. This fixes actual
CUDA seeding and checkpoint RNG capture. Historical MOSEI cmerc did not use this
repair; its score is reference only, not a strict optimization-paired control.
No old running snapshot or finished checkpoint is changed by this new experiment.

Run unit tests, two real largest-variant smokes (MOSEI and IEMOCAP6), then deploy a
clean committed immutable worktree. Source hashes and live configs are recorded.
Controller verifies source before every admission. Example persistent command:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_nested_cmerc_tuning_20261023.run \
  --source-commit SOURCE_SHA --output /absolute/fresh/execution_v1 \
  --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e --per-gpu 4
```

Server biggpu; GPU7 preferred; GPU4 forbidden. At most4 of this queue's children,
with >=6GiB reservation per active worker +2GiB margin,64GiB RAM,25GiB free disk.
Conservative admission may run fewer than4. No unrelated task is touched. No OOM
retry or scope expansion. Explicit interrupted-controller recovery requires
`--resume`, identical identity and complete Adam/CPU/CUDA RNG state.

Complete-only RESULT.md/json/PER_RATE.csv rank only new same-source runs. All
conditions and failed/pending jobs remain visible in QUEUE.json. Training cost,
parameters, exact mask hashes and gradient paths are retained. No success claim
follows from smoke; no automatic multi-seed follow-up regardless of score.
