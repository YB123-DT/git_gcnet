# GCNet Research Execution Contract

This file applies to the entire GCNet repository. It supplements the parent
workspace instructions. Its purpose is to prevent repeated baselines,
validation inflation, and infrastructure work from delaying scientific runs.

## 1. Default execution path

For each new GCNet module, execute these stages in order:

1. **Reuse audit (target: 5 minutes)**
   - Locate the target module and existing experiment arm.
   - Locate an existing compatible Original result.
   - Reuse the existing runner, mask bank, summarizer, and result layout.
2. **Method lock (target: 10 minutes)**
   - Write the paper anchor, exact replacement point, equation, reason, and
     unchanged components in the experiment MD.
   - Define the single scientific variable before editing code.
3. **Implementation (target: 20 minutes)**
   - Prefer one module file plus minimal model/train wiring.
   - Do not create a per-method runner, result loader, manifest system, or mask
     validator when an existing one can express the experiment.
4. **Fast gate (target: 5 minutes)**
   - Run only method-level mathematics and one integration wiring test.
   - Run a one-epoch archive test only when train/save code changed.
   - Do not run full test discovery as a precondition for a research job.
5. **Formal launch (target: 5 minutes)**
   - Freeze the commit and launch candidate jobs immediately.
   - Formal execution must not contain smoke jobs.
6. **Evidence and decision**
   - Verify completed jobs, paired masks, metrics, and parameter counts.
   - Update the experiment MD with per-rate results and accept/reject decision.

The 40-minute pre-launch target is an operating budget, not permission to skip
a real blocker. If a stage exceeds its target, record the blocker and continue
the shortest root-cause path. Do not respond by adding general infrastructure.

## 2. Original inheritance is mandatory by default

Do not rerun Original merely because the Git commit, filename, result directory,
or candidate module changed.

An Original result is reusable when all of these are equal:

- dataset and feature files;
- train/validation/test split and fold;
- missing rates, seeds, and mask SHA256 for every paired job;
- optimizer, epochs, losses, batch size, hidden size, dropout, and graph setup;
- the selected Original computation path, its parameters, RNG trajectory, and
  default output behavior.

Unselected newly instantiated parameters do not invalidate a baseline when a
test proves they do not change the selected path or RNG.

Original may be rerun only if at least one item above differs. Before launching,
the experiment MD must contain a `Baseline rerun justification` naming the
changed scientific input or computation. An empty or commit-only justification
forbids the run.

For IEMOCAP-6 fold 5 under the locked 8-rate/5-seed protocol, reuse the verified
Original at:

```text
/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/
protocol_recovery_v1_biggpu/formal/original
```

## 3. Job budget

The default IEMOCAP-6 candidate budget is exactly:

```text
1 candidate x 8 missing rates x 5 seeds x fold 5 = 40 formal jobs
```

Every job beyond this budget must be listed in the experiment MD with:

- the scientific question it answers;
- why existing evidence cannot answer it;
- the additional job count and expected wall time.

If these fields are absent, do not launch the extra jobs.

For a dataset without any compatible Original, one first baseline run is
allowed and becomes the reusable baseline for all later modules.

## 4. Validation layers must remain separate

### Layer A — method unit tests (no training)

Prove only the paper-facing mathematical claims: shape, mask mapping, limiting
behavior, and gradients required by the method.

### Layer B — integration tests (no training)

Prove the replacement point, default-path identity, and single-variable wiring.
Do not repeat method-level equivalence at every model layer.

### Layer C — archive integration (optional one epoch)

Run once only after changing argument parsing, training control flow, archive
naming, or NPZ contents. Do not rerun it after changes limited to tests, docs,
summaries, or remote scheduling.

### Layer D — formal experiment

Run only the frozen candidate grid. Do not mix smoke, debugging, code cleanup,
or baseline reruns into this stage.

## 5. Prohibited expansion

Unless a concrete failure proves it necessary, do not add:

- another per-method runner or summarizer;
- adversarial NPZ/path/manifest/process tests;
- repeated CPU/GPU equivalence tests at multiple layers;
- source-text inspection tests;
- full-suite execution before every remote launch;
- new mask banks when compatible immutable banks already exist;
- concurrent runners that can exceed three workers per requested GPU.

Fix observed root causes with the smallest local change. Do not generalize a
one-off failure into a new framework during an active research experiment.

## 6. Required time and execution record

Every experiment MD must record UTC timestamps for:

- task received or first durable design record;
- code frozen;
- formal training launched;
- formal training completed;
- result summary completed.

It must separately report:

- necessary candidate jobs;
- inherited baseline jobs;
- newly run baseline jobs;
- smoke/debug jobs;
- failed or wasted jobs;
- total wall time and avoidable wall time.

## 7. Stop conditions

Stop adding code and launch the formal candidate when:

- the formula and replacement point are locked;
- the default path is unchanged;
- the fast gate passes;
- data, masks, environment, and output filename are valid.

Stop promoting a method when the locked multi-rate result is negative or
unstable. Preserve it as an ablation/negative result instead of adding unplanned
mechanisms during the same experiment.

