# MOSEI Flat128 three-seed control and finite Nested screen

Status: DESIGN ONLY; no new training submitted by this plan.
INTERNAL TEST-ORACLE SCREEN.24 result slots,23 fresh training runs,1 reused.

## Fixed protocol

All models use Flat hidden128 (output still1600), cmerc-repro,60epochs,
constant LR1e-4 for all parameters,batch32,Adam,WD1e-5,MSE,cyclic missing0–.7,
gcnet-random-mask-v1, original accepted data/splits/exact test masks, original
UUID CUDA RNG repair, LocalSkip and original8x64 CHM, write_step0.6.
No local LR, scheduler, new loss, source-role block, Gate or JEPA.
Nested retains the existing input-residual route and zero decoders.

The21 Nested candidates are all seed66 only. Flat128 controls cover66/67/68.
Do not average one seed of Nested against three Flat seeds and call it a paired
improvement; screen each Nested against Flat128 seed66. No automatic multi-seed
promotion or additional search is included in these24 slots.

## Three Flat result slots

| ID | Model | Seed | Action |
|---|---|---:|---|
| F66 | Flat128 | 66 | reuse completed current cmerc run |
| F67 | Flat128 | 67 | fresh training |
| F68 | Flat128 | 68 | fresh training |

F66: `/data2/yb/remote_experiments/osram_mosei_capacity_cmerc_20261024/execution_v1/runs/f128`.
60epochs complete;4,746,977 parameters; ACC83.606219,W-F1 83.387044,
high-missing W-F1 80.011427. Preserve all source weights/predictions/masks.

## Twenty-one Nested result slots

First18 form a3x2x3 factorial: token dim32/64/128, graph depth2/3,
graph groups2/4/8; MLP hidden128 fixed. Three additional MLP anchors permit
within-dimension comparisons without silently coupling hidden width to dim.

| ID | Dim | Depth | Graph groups | MLP hidden | Nested params | Total params |
|---|---:|---:|---:|---:|---:|---:|
| N01 | 32 | 2 | 2 | 128 | 96,258 | 4,843,235 |
| N02 | 32 | 2 | 4 | 128 | 96,386 | 4,843,363 |
| N03 | 32 | 2 | 8 | 128 | 96,642 | 4,843,619 |
| N04 | 32 | 3 | 2 | 128 | 108,771 | 4,855,748 |
| N05 | 32 | 3 | 4 | 128 | 108,899 | 4,855,876 |
| N06 | 32 | 3 | 8 | 128 | 109,155 | 4,856,132 |
| N07 | 64 | 2 | 2 | 128 | 191,234 | 4,938,211 |
| N08 | 64 | 2 | 4 | 128 | 191,490 | 4,938,467 |
| N09 | 64 | 2 | 8 | 128 | 192,002 | 4,938,979 |
| N10 | 64 | 3 | 2 | 128 | 216,131 | 4,963,108 |
| N11 | 64 | 3 | 4 | 128 | 216,387 | 4,963,364 |
| N12 | 64 | 3 | 8 | 128 | 216,899 | 4,963,876 |
| N13 | 128 | 2 | 2 | 128 | 381,186 | 5,128,163 |
| N14 | 128 | 2 | 4 | 128 | 381,698 | 5,128,675 |
| N15 | 128 | 2 | 8 | 128 | 382,722 | 5,129,699 |
| N16 | 128 | 3 | 2 | 128 | 430,851 | 5,177,828 |
| N17 | 128 | 3 | 4 | 128 | 431,363 | 5,178,340 |
| N18 | 128 | 3 | 8 | 128 | 432,387 | 5,179,364 |
| N19 | 32 | 3 | 8 | 32 | 65,667 | 4,812,644 |
| N20 | 64 | 3 | 8 | 64 | 159,235 | 4,906,212 |
| N21 | 128 | 3 | 8 | 256 | 662,403 | 5,409,380 |

Counts: existing `NestedSweepGNN` + `TokenAdapter` instances were constructed
on remote CPU with CUDA hidden; exact module parameter counts measured and
added to verified original Flat128 count. These are not estimated training
times or claimed model scores. All total budgets remain below pureFlat256
(5,508,961), so the screen stays compact.

Graph groups are NOT CHM memory heads. CHM remains8 heads of64 dimensions.
Groups2/4 merge4/2 contiguous actual heads for graph tokens, preserving the
complete512-dimensional forward read when decoded. No truncate/pad shortcut.
Groups2/4 reduce the number of explicit rooted graph passes, but their changed
graph granularity is an experimental factor, not a free speed-only change.

## GPU plan

- biggpu only; physical GPU0 and1 UUID-bound, maximum6 own active workers per
  card,12 globally. GPU4 forbidden; no foreign process signals.
- Count still-running capacity jobs against the six-per-card ceiling. At
 15:46:49UTC GPU0 was free; GPU1 still had3 old capacity workers. Do not blindly
  add6 to those3. Recheck allocation and RAM/disk before every admission.
- Suggested initial GPU0 slots: F67,F68,N19,N20,N01,N02.
- Suggested initial GPU1 slots: N03,N04,N05,N06,N07,N08, subject to old jobs
  finishing/resource gates. Remaining N09–N18/N21 fill released slots; no
  unnecessary barrier between first and second waves.
- Resource reservation must use measured full training/evaluation peaks plus
  margin. Previous6144MiB reservations cannot admit6 on32GiB cards; never
  merely change the process-count flag and ignore memory. Validate lightweight
  group2/4 and largest N21 peak before safely choosing per-job reservations.
- Keep batch32/precision/protocol fixed; reduce concurrency if needed. RAM
  >=64GiB available, disk>=25GiB before admission. Only84GiB disk was free at
  the precheck: account for retained BEST/last artifacts; do not delete old runs.
- Do not promise every run completes in2hours. Shared-load timing and different
  root counts make runtime heterogeneous. Measure after admission and update ETA.

## Implementation/verification handoff

Reuse existing immutable cmerc worker callbacks. Add explicit dataset-scoped
registrations for these existing NestedSweep configurations only; do not change
Nested's mathematics, vectorize active jobs or reuse MELD-labeled training configs.
Preserve defaults and verify unchanged original parameters/RNG. Full same-width
Flat128 initialization must match per seed; zero decoder initial outputs must
match the corresponding untrained Flat128. Preserve actual gradient-path checks,
single Memory scan, checkpoint/Adam/CPU+CUDA RNG recovery and selected prediction
hashes. Keep old running source snapshots immutable.

Reuse F66 only after config/source/data/mask/environment/RNG parity checks.
Save a24-slot result manifest separating reused results, new pending/running/
failed/completed results and physical GPU/PID/log paths. Report parameter count,
actual epoch time, ACC/W-F1/MSE per rate, eight-rate and high-missing means,
and paired seed66 deltas vs Flat128; only completed runs enter rankings.
Rank completed candidates by eight-rate W-F1, with high-missing W-F1 and then
lower parameter count as tie-breakers. Preserve all outcomes, not just winners.
Future selection/claims remain exploratory test-oracle, not independent evidence.

## Current Nested results motivating the plan

At15:46:49UTC, same cmerc/newMissing/LR1e-4/seed66:

| Model | Epoch | ACC | W-F1 | High W-F1 |
|---|---:|---:|---:|---:|
| Flat128 | 60 | 83.606 | 83.387 | 80.011 |
| Flat256 | 60 | 83.596 | 83.374 | 80.092 |
| Flat256+Nested64 | 60 | 83.479 | 83.285 | 79.992 |
| Flat256+Nested128 | 58 | 83.304 | 83.148 | 79.811 |
| Flat512+Nested128 | 58 | 83.304 | 83.194 | 79.968 |
| Flat512+Nested256 | 56 | 83.407 | 83.297 | 79.932 |

The earlier completed LargeFlat1600+Nested32 had ACC83.664695,
W-F1 83.549017,high W-F1 80.132940. This is a different Flat width,
not a prediction of N19's performance. Current small-Flat Nested results do
not yet demonstrate a gain. This plan tests a compact new operating point,
not an already-established bottleneck or guaranteed improvement.
