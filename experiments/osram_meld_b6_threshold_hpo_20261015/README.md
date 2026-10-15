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

## Resource-only extension, 2026-10-15

User subsequently authorized using the underutilized GPU3/5 in addition to GPU2.
The standalone `resume_resources.py` changes only future job placement to the
UUID-validated whitelist2/3/5. Keep20total trials,6global pending,6total compute
processes/card including unrelated tasks, and the five-completed-starts barrier.
Do not invent extra starts just to occupy these cards. GPU4 remains prohibited.

Training workers remain at immutable source
`9fa9f571e7095d5b09f9c256d888c31a76c82744`, including subsequent trials. The
standalone controller checks original source hashes, frozen configs and saved
Optuna protocol before acquiring the same exclusive controller lock. Existing
worker PID/command/start-time identities are recovered by the original scheduler;
workers must not be stopped, migrated, restarted or reconfigured. Only the old
controller is handed off, never its process group. No existing job is replaced.

Keep original PREFLIGHT, masks, outputs and study database untouched; append the
new resource provenance as `RESOURCE_RESUME.json` and reference it from RUNNER.
The initial five configurations were already running before this change, so
there are no adaptive sampler RNG draws to recover at handoff. Later interrupted
resumes do not promise identical future proposals unless sampler RNG is saved.

```bash
python /home/qinjinghui/remote_experiments/osram_meld_b6_threshold_resume_resources_20261015.py \
  --source-root /home/qinjinghui/remote_experiments/osram_meld_b6_threshold_code_9fa9f57 \
  --gpus 2,3,5 --check-only
# After the old controller releases its lock, omit --check-only to adopt workers.
```

This resource extension is not a model change, new method, extra trial budget,
or performance claim. Unit tests cover allowed GPUs, immutable study scope and
original worker source routing; remote check-only verifies the real saved study.

Handoff verified2026-10-15 08:56UTC: new controllerPID136664 adopted all five
unchanged trainingPIDs4052358/4052381/4052518/4052528/4052536 at26/50epochs.
Queue capacities now include2/3/5; additional cards have not yet received jobs
because all five initial trials must finish first. No extra model training has
been launched at this handoff. Runtime journal is `RESOURCE_RESUME.json`.
Local targeted regression:12passed,2Optuna-dependent tests skipped; remote
check-only and live PID/start-time/queue ownership checks passed. The pre-existing
unused torch-sparse/libcusparse warning remains; no training error was observed.

## Subsequent immediate-launch override

User explicitly revoked waiting and requested GPU0 with fewer jobs on GPU3/5.
`immediate_burst.py` therefore consumes five of the existing remaining trials
immediately: GPU0three, GPU3one, GPU5one. Existing GPU2workers remain untouched;
no future jobs are added to GPU2 by this controller. GPU4 is still forbidden.
At this burst preserve all already-running trials across the four explicitly
requested cards. Added capacity is3/1/1 on0/3/5, not a replacement for existing
jobs on these cards; actual total queue capacities include preserved jobs.

The requested override permits immediate proposals without waiting for startup
feedback, but the actual initial five completed during preflight: the former
controller had already issued six feedback-informed trials(5..10). The new burst
consumes another five remaining slots(11..15); Optuna records completed-feedback
count for each. Therefore these actual proposals can use five completed trials;
do not describe them as feedback-free draws. No change to training code or total
budget20. On restart use the saved absolute issued-target16, not another five.
Collect the active burst before the original adaptive loop resumes for trials
16..19. Later pending cap remains6. Save actual change in
`IMMEDIATE_BURST_V2.json`; keep the no-launch first handoff journal separately.
This is a user-directed search scheduling change, explicitly recorded, not new
method evidence. Do not hide any earlier warning about feedback availability.

```bash
python /home/qinjinghui/remote_experiments/osram_meld_b6_threshold_immediate_burst_v2_20261015.py \
  --source-root /home/qinjinghui/remote_experiments/osram_meld_b6_threshold_code_9fa9f57
```

Verified actual launch: controller475303; new trial11PID475783/GPU0,
trial12PID475965/GPU0,trial13PID476011/GPU3,trial14PID476374/GPU5,
trial15PID476513/GPU0. Existing sixPIDs456480/456633/456648/456798/456972/457449
unchanged. At this checkpoint study states5COMPLETE+11RUNNING, not twenty
completed. No queue delay for these five starts; resource checks remain mandatory.
