# Nested only: MOSEI and IEMOCAP cmerc screen

LATEST USER SCOPE: IEMOCAP4 cancelled. Only MOSEI and IEMOCAP6 remain,14 jobs
(7 each), seed66 only. IEMOCAP6 uses fold5, not a fivefold mean. The original
21-job launch receipt remains historical evidence, NOT the current plan.

INTERNAL TEST-ORACLE SCREEN. No automatic expansion. A scope-only supervisor
adopts the original live MOSEI worker and keeps the old immutable worker source
3777ab1 for all14 retained jobs. No model, optimizer, RNG, masks or weights are
changed/restarted. Cancelled4-class artifacts are preserved and excluded from
ranking. See IEMOCAP4_CANCELLED.md for the operational handoff.

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

MOSEI60 epochs at constant1e-4/MSE; IEMOCAP6 100 at constant1e-3/CE. Both batch32,
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

Complete-only RESULT.md/json/PER_RATE.csv rank only the14 retained same-source runs. All
conditions and failed/pending jobs remain visible in QUEUE.json. Training cost,
parameters, exact mask hashes and gradient paths are retained. No success claim
follows from smoke; no automatic multi-seed follow-up regardless of score.

Preflight note: the first IEMOCAP6 smoke stopped at a too-strict bitwise-output
assertion (max log-probability difference7.1525574e-7, zero changed classes).
Read-only hooks verified original Flat Adapter/LocalSkip inputs exactly equal
(max difference0, identical strides), identical original parameter digests and
Nested input_change_norm=0. Matching requires_grad flags did not remove roundoff.
The check now retains strict original parameter/input-correction/class equality,
allows float32 prediction tolerance atol=rtol=1e-6 and records BOTH measured
max error and whether bytes actually match. It never reports approximate parity
as exact. No model math, optimizer, loss or mask code was changed by this fix.
