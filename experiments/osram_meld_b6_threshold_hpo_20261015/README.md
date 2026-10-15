# B6 threshold initialization and independent learning-rate HPO

INTERNAL TEST-ORACLE SCREEN. Authorized2026-10-15; MELD seed66 only.

## Finite preregistration

20total trials =5fixed initial +15adaptive proposals, not25. All train from
scratch50epochs, batch32; no old weights are resumed. Fix rank2/steps14from the
previous highest-ACC trial3(61.216,UA35.935), taking the higher-UA configuration
of the Test-best ACC tie. This starting choice was stated before execution.

Main AdamLR0.0003433234536467373,weight_decay0.0000044857894231052665,
dropout0.22489055023388654remain frozen. Preserve original CHM,Flat1600,LocalSkip,
GCNet random-mask-v1,cyclic0.0–0.7,no-JEPA,per-rate Test-best ACC selection.

Exactly2searched quantities:

* `threshold_init`:0.01..1.0log. Initializes learned threshold coefficients
  `theta_t = threshold_init * 0.8**t`,t1..14. Keep decay0.8unchanged. This does
  NOT alter the separate SVD initializer's0.1*scale clipping.
* `b6_lr_multiplier`:0.1..30log. Only the B6core threshold/step/gamma tensors
  receive mainLR*multiplier. Shared encoder,CHM,Flat and head retain their prior
  optimizer rates. Weight decay,clipping and loss unchanged. No extra gate/loss.

Initial5: (0.1,1), (0.03,1), (0.3,1), (0.1,0.1), (0.1,10). The first repeats
the previous setting at the new immutable source. Do not reuse an old score as
a new completed control. Thresholds,steps and gamma remain trainable; only
their threshold initialization and optimizer rate are searched. Gamma starts0.

After5complete, independent TPE with constant-liar handling fills available
slots. Latest user resource override: **physical hostGPU2only**,at most6total
compute processes including external jobs; neverGPU4. No batchsize changes to
fit more. Global pending cap6;15trials remain after initialization and are
proposed incrementally as jobs finish. Do not expand the budget to fill slots. Reuse measured
memory admission with reserve and lower concurrency after one audited OOM retry.

## Evaluation and stopping

Keep exact winning-epoch predictions/masks,all8BEST weights,full active optimizer
and RNG recovery. Report Test-best and existing common-mask on/off replay
separately. TPE objective stays0.6*deltaACC+0.4*deltaUAversus screen Flat on common
masks; separately list highest Test-best ACC. Test-selected results are not
independent paper evidence. Stop after20trials,retain failures and negative results.
Do not automatically add structural variants,other datasets or multi-seed runs.

An increased score need not demonstrate direct inference contribution: previous
on/off differences were tiny. Keep the same on/off diagnostic rather than claim
that tuning makes PCA effective by definition.

## Runtime expectation and execution

Previous20trial batch took4h14m26s end-to-end. Per-run training46–140minutes,
median82minutes; post-training replay/loading/statistics20–101seconds,median44.
TPE proposing parameters is cheap; model training dominates. Same-card6way
sharing can increase each run's latency and need not improve total throughput.
The prior4–6hour estimate assumed3cards. Under the latest single-cardGPU2scope,
use a rough5–8hour expectation and recalibrate from actual epoch throughput;
this is not a promise. Record actual cost.

Use existing biggpu SSH and `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Immutable source and output are separate from all previous experiments. New
output lives under `/home` because `/data2`has little free space.

```bash
python -m experiments.osram_meld_b6_threshold_hpo_20261015.run \
  --output /home/qinjinghui/remote_experiments/osram_meld_b6_threshold_hpo_20261015 \
  --gpus 2 --per-gpu 6 --trials 20
```

Implementation: tests first → default-compatible threshold plumbing and exact
B6optimizer group → shared bounded controller → immutable source → remote CPU
checks and real batch smoke at threshold1/multiplier30 → detached queue.
SQLite filename `STRUCTURE.sqlite3` is inherited from the existing controller;
the actual study searches only threshold initialization and B6LR,not rank/depth.
