# MELD Block2 finite screen and automatic training HPO

INTERNAL TEST-ORACLE SCREEN. User-authorized exploratory test selection, not
an independent-test paper result. Initial inspected HEAD `e0bb6b4`; every
formal run uses the new shared immutable implementation commit, not an old
backbone snapshot. Old protected Flat three-seed checkpoints are untouched.

## Fixed scope

Seven from-scratch seed66 jobs: Flat and B1–B6. MELD official ECERC features,
50epochs, batch32, Adam1e-3/wd1e-5/dropout0.5, emotion-only, cyclic0.0–0.7,
original GCNet random-mask-v1. Eight-head64dim causal CHM, write0.6,
Flat1600, original Local Skip/head/task loss all retained and train jointly.
Only one existing `MeaningfulInputAdapter` mechanism is enabled at a time.
Read `METHOD_AUDIT.md` for actual mechanisms, initial identity, Local update
scope, and historical MOSI deduplication. These are six transfers, not six
new research inventions.

## Distinct evaluations

**A: Test-best.** The default-off `preserve_selection_artifacts` trainer flag
collects predictions/logits/labels/masks/IDs/speaker directly during the already
scheduled evaluation. No extra forward or RNG draw. Every winning checkpoint
references its own immutable epoch artifact; UA/recalls/W-F1/CE come from that
same ACC-selected point. The legacy final random retest is still performed
as configured but **never** used to supply these selected metrics.

**B: PAIRED_MASK_REPLAY.** Each per-rate selected weight is evaluated using
the same protected seed66 winning-epoch mask NPZ from `exact/source_0`, checked
against immutable file hashes. Labels/order/utterance positions/speakers are
validated. Original canonical files lack IDs/speaker metadata; identities are
reconstructed from the same official test loader and checked against this
run's identity-bearing selection artifact. Block2-on/off share **one** CHM
scan: capture original input before Block2, then compute an off Flat readout
from those same raw reads/current Local. No second trajectory, no retraining,
and Local Skip is never fed modified Local. Flat on/off are identical.

Random-mask availability groups are not persistent-missing experiments.
Dataset SHA256 must be exactly:
`a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`.

## Pre-registered gate and Optuna

Paired eight-rate mean relative to the NEW Flat: exclude dACC <−0.30pp or
dUA <−0.50pp. Rank score0.6*dACC+0.4*dUA; require positive score and at least
one positive delta. Top2maximum, IDtie-break. Record Pareto front and conflicts.
If no positive signal, stop HPO and report failures; never force a winner.

Each winner20trials (CLI permits16–20), SQLite TPE seed66. Five explicit
startup configurations finish one at a time before batches of <=2pending
trials/study. Every trial really trains50epochs. Search ONLY lr3e-4–2e-3log,
wd1e-6–1e-3log, dropout0.2–0.6. This first round is **training HPO only**:
no module-specific structure parameters are exposed or claimed as searched.
Failed/unfinished trials do not rank; every failure and parameter is saved.
Test-selected best scores have selection bias; future paper performance
requires new training with validation selection and independent seeds.

## Immutable deployment and resource admission

Run on `ssh biggpu`, Python `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Current source is committed then checked out to an immutable remote worktree.
All formal workers share that commit and source hashes. GPU0/2/3 default,
UUID-pinned logicalcuda:0, hostGPU4 prohibited; no external jobs changed.
Max3cards,4taskprocesses/card,12global; screen only7jobs. Peaks measured using
three real GPU optimizer steps on the32longest training conversations (.7
smoke mask activates maximum active evidence slots). CUDA identity/zero-bridge
gradient/updates are checked; smoke is not a performance result.

The queue admits measured peaks plus1GiBmargin and live free memory, reserves
disk, reduces concurrency once for auditable OOM retry with unchanged HP.
It records PID birth/command identity to avoid duplicate submission on restart.
Epoch-boundary recovery includes full model/optimizer/RNG/sampler/config/source
identity. Unreferenced recovery versions are pruned only in THIS new run.
Completed HPO jobs retain all8BEST weights and exact prediction artifacts;
only their temporary full recovery cache is released to respect disk limits.
Protected baseline24checkpoints never enter cleanup.

Torch1.13 can incorrectly enumerate0devices under a GPU UUID although CUDA
runtime sees1. A logged process-local compatibility shim corrects enumeration
before seeding and RNG recovery; all methods AND fresh Flat use the same shim.
Do not assume these new training results must exactly equal archived seed66.

## Unified command

From the new immutable source worktree:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m \
  experiments.osram_meld_block2_auto_20261014.run \
  --phase all --gpus 0,2,3 --per-gpu 4 --seed 66 \
  --top-k 2 --trials-per-model 20 \
  --output /data2/yb/remote_experiments/osram_meld_block2_auto_20261014
```

Use nohup/tmux (controller and detached children survive SSH). Reinvoke the
same command/source/output to recover; don't change parameters or code in
place. `--phase summarize` refreshes the report without starting training.

Artifacts: PREFLIGHT.json, RUNNER.json, queue/QUEUE.json and logs, smoke/,
screen/{Flat,B1..B6}/SUMMARY.json, best_miss_*.pt, selected_predictions_*.npz,
paired_miss_*.npz, history/config; optuna/*.sqlite3, all trial records/weights;
RESULT.md, SUMMARY.json/CSV. Only source/config/text/small statistics go to Git.

At implementation time full training is not yet complete. See real run records
for statuses; no performance is inferred from CPU tests or GPU smoke.
