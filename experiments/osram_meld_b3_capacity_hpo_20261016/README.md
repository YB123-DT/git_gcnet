# MELD B3: conditional capacity and optimizer search

INTERNAL TEST-ORACLE SCREEN — seed66 exploration only.

The previous 20 GPU2 runs plus four GPU3 extensions searched **only token count
and exchange rounds**. This independent, finite extension trains **four additional
models**, for **28 total** across the three experiments. It does not cancel,
overwrite, or modify the old studies or their numerical source.

## Scope and search space

B3 remains the existing `mbt_role_bottleneck` input adapter between CHM reads and
the original Flat1600. Original raw Local Skip, classifier, CHM reads/writes and
Query are unchanged. All trials retain MELD seed66,50epochs,batch32,global Adam
lr0.001,weight decay1e-5,dropout0.5, and `gcnet-random-mask-v1`. B6 tuning values
are not imported. The bottleneck has no additional internal dropout in this task.

| Parameter | Search values | Historical default |
|---|---|---:|
| Bottleneck tokens | Integer1–8 | varied previously |
| Exchange rounds | Integer1–4 | varied previously |
| Internal dimension | 64,128,256 | 128 |
| Internal attention heads | 2,4,8 | 4 |
| FFN hidden dimension | 128,256,512 | 256 |
| B3-only LR multiplier | 0.1,0.3,1.0 | 1.0 |

Every new job must change at least one of the last four parameters. Additionally,
jobs1–4 must respectively change dimension,attention heads,FFN width,and B3-only
learning-rate multiplier, guaranteeing coverage of all four requested axes.
The other parameters remain TPE-selected. This is a **conditional expansion of
the search**, not a controlled one-factor ablation; several parameters can change
in a single trial, and causal attribution from four scores would be unjustified.
CHM's eight memory heads are not the searched B3 internal attention heads.

The B3 learning-rate multiplier applies to the entire existing
`osram.meaningful_block` parameter group,including tokenizer,interaction core,and
raw decoders. The backbone,Flat and classifier retain the original global rate.
No extra loss or second module is enabled. All dimension/head combinations in the
space are divisible and valid. Effective FFN width is explicit rather than a
hidden dimension-dependent change.

## Bayesian evidence, uniqueness and evaluation

The controller copies completed original outcomes read-only into a new SQLite
study as `reference_only` observations. These have the actual historical defaults
for all four newly exposed knobs. It does not falsely claim those knobs were
searched previously. All24 old full configurations are reserved, including failed
or incomplete jobs. A token/round pair may repeat **only if another knob differs**.
Because every old observation has default values for the newly exposed knobs,
these four new jobs remain exploratory: the existing history does not establish
well-optimized settings for the new capacity or learning-rate axes.
Complete six-parameter tuples are unique. Sampling uses independent TPE with five
startup observations,constant liar,and seed266; at least five completed references
are required, so new jobs use TPE rather than four fresh random-start trials.
Conditional rejection enforces the nondefault requirement and uniqueness; after
64 rejected proposals an explicit deterministic feasible fallback is recorded.

The objective stays `0.6 × paired ΔACC + 0.4 × paired ΔUA` against the retained
original-hyperparameter Flat seed66 reference. Protected common GCNet masks are
unchanged. Per-rate Test-best ACC,UA,W-F1 are saved separately, with their exact
winning-epoch predictions/masks/checkpoints. UA is never mixed with a different
epoch or newly drawn mask. Test use during search introduces selection bias.

## Resource and reproducibility contract

Run only on biggpu physical GPU3,UUID
`GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`,at most four total GPU processes including
old jobs and smoke. The controller waits for the previous four GPU3 jobs to finish
without signalling them. Then one real-batch,three-update smoke tests the largest
search configuration:tokens8,rounds4,dim256,heads8,FFN512. Its gradient/update checks
and memory peak must succeed at the **new committed numerical source**; old
dim128 smoke results are not reused. The resource-aware persistent scheduler may
lower concurrency rather than altering batch/model settings.

The controller and workers use the same immutable current checkout,not the old
`737fdb1` numerical source. Historical default equivalence tests must pass before
launch. `PREFLIGHT.json` records source/module hashes,config,data SHA,mask hashes,
environment,GPU UUID,and new smoke. `REFERENCE.json` freezes actual source artifacts
and their hashes. SQLite,queue state,logs,config,checkpoints and results stay in
the independent remote output. Existing protected checkpoints are untouched.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m \
  experiments.osram_meld_b3_capacity_hpo_20261016.run \
  --output /home/qinjinghui/remote_experiments/osram_meld_b3_capacity_hpo_20261016
```

Launch from the committed immutable source under tmux. Identical restarts recover
the same four jobs/study; source/config/reference changes are rejected. Outputs
include `PLANNED_JOBS.json`,`CAPACITY.sqlite3`,`TRIALS.json`,`SUMMARY.json`,
`RESULT.md`,and `RESULT.csv`. Failed jobs are retained and not ranked as successes.
No automatic multi-seed expansion or further search is authorized here.

## Targeted verification

```bash
python -m unittest experiments.osram_meld_b3_capacity_hpo_20261016.tests.test_capacity
```

Tests cover fixed original hyperparameters,valid parameter space,full-tuple
uniqueness,no token-only trials,per-axis coverage,actual Optuna finite4job budget,
and restart without duplicate training. Optuna integration runs in the approved
remote environment; it is explicitly skipped where Optuna is absent.
