# Completed TASR handoff

INTERNAL DIAGNOSTIC ONLY. Four approved frozen-adapter jobs completed, no new
multi-seed/fivefold runs or search. See RESULT.md and COMPARISON.json/CSV.

## Source and experiment identity

- Parent: ea27c084ad21d38a01025ed0f8fbaccb6b1dcb49, feature/ecerc-features.
- Structure:1a56a21; mathematical/real-GPU verification and runner:5595fd0.
- Immutable training source:38251f4b5ca1f1a64ea698eb3cc3a4fb9acf2a65.
- Independent read-only analysis:ce3c055b8e705668724be1cb6e4af675197cd3c9.
- User destination:https://github.com/YB123-DT/git_gcnet, remote `github`;
  NEVER upstream `origin`. No checkpoint/feature/NPZ files enter Git.
- Server biggpu, UUID-bound physicalGPU0. Other users' processes untouched;
  broken physicalGPU4 never allocated. GPU5 used only earlier real smoke.
- Runtime `/data2/yb/conda_envs/cmerc-repro/bin/python`, Python3.10.13,
  Torch1.13.1+cu117, NumPy1.26.4, sklearn1.3.2.
- Run root `/home/qinjinghui/remote_experiments/osram_tasr_20261018/runs_v1`.
  Queue complete; all four workers exit0, one attempt, zero failures/retries.
- Summary root `/home/qinjinghui/remote_experiments/osram_tasr_20261018/summary_v1`.
  The temporary SSH banner outage prevented observation, not a demonstrated
  training failure. No jobs were restarted/duplicated because of that outage.

## Changes and key implementation locations

- `gcnet_missing_m3/tasr.py`: strict_past_trajectory(line13),
  JointSourceTracker(line39), TASRAdapter(line96). One shared151→64→1 score,
  zero output initialization,59,423 parameters, no new query or task loss.
- `gcnet_missing_m3/osram.py`: opt-in flags/guards(line873), unchanged joint
  block_write with optional same-solve intermediate return(line1523), source
  decay/read/update in _scan, tracker construction(line1853), Gap-only correction
  before original Flat(line1932). Runtime Gap ablations cannot be bypassed.
- `gcnet_missing_m3/model.py` and `train_gcnet.py`: default-off model/config/CLI
  plumbing: osram_source_tracking, osram_tasr, osram_tasr_source_mode.
- `tests/test_tasr_{primitives,integration,protocol,runner,summary}.py`:
  algebra, causality, mask/RNG/legacy parity, gradient/update/reload, controls,
  split, bounded scope, paired identity/metrics and fail-closed reporting.
- Experiment code:protocol.py(fixed protected anchors/splits/patterns),
  runner_helpers.py(strict load/RNG/replay/recovery), run.py(two-view frozen
  training/validation/final evaluation), launch.py(four-job persistent queue),
  smoke.py(real GPU checks), summarize.py(read-only prediction recomputation).
- Records:DESIGN.md, README.md, VERIFICATION.md, SMOKE_MELD.json,
  SMOKE_IEMOCAPSix.json, RESULT.md, COMPARISON.json/CSV and this HANDOFF.md.

## Data and forward/gradient path

Original cfg84 CHM/ObservedSetEncoder/Local/Skip/Base/queries/Flat/head and original
GCNet random sampler remain unchanged. Alpha decay precedes causal reads;
A/T/V still participate in ONE joint block update. H_A/H_T/H_V reuse its exact
Kbar/Vbar/P, including availability×sqrt(beta), source injection AND shared
erasure. The original M remains authoritative. Reading the same residualized
Gap query from decayed sources reconstructs original Gap within roundoff.
Only missing-target forward Gap receives bounded source-conditioned correction;
backward half zero, no-history/padding/observed-target corrections zero.

F1 presents G/3 three times with the same Local, trajectory and parameter budget;
F2 presents actual source reads. Neither changes Base or original Local Skip.
One fixed protected original miss=.3 checkpoint per dataset for every condition;
original checkpoint/prediction/mask/data hashes are retained in COMPARISON.json.
No historical baseline or protected file was modified.

Original network frozen and eval throughout. Only59,423 TASR parameters train;
original frozen Flat/norm/head still transmit task derivatives to the Adapter.
View1 uses original GCNet random mask; independent View2 uses COMPLETE training
features and uniform six-pattern Persistent masks, not deletion from View1.
Each starts zero Memory and sources. Loss=.8CE_random+.2CE_persistent, backward
both before one gradient clip/Adam step. No auxiliary objective or projector.

Both controls:Adam3e-4/wd1e-5/batch32/seed66/30epochs; one adapter checkpoint
selected only using validation random8 ACC then lower CE. MELD official dev;
IEMOCAP6 seed66/fold5 uses deterministic10% original training-conversation holdout
because inherited official validation aliases test. The frozen backbone
previously saw this holdout: do not claim independent end-to-end validation.
Historical anchors are old Test-oracle; paired F0 is NOT old per-rate-best mean.

## Completion, costs and tests

| Dataset | Control | Selected epoch | Optimizer steps | Peak MiB | Total observed seconds |
|---|---|---:|---:|---:|---:|
| MELD | F1 |3|990|267.9|7877.8|
| MELD | F2 |20|990|267.7|7875.9|
| IEMOCAP6 fold5 | F1 |5|120|979.8|720.2|
| IEMOCAP6 fold5 | F2 |26|120|980.4|720.0|

Times include setup/training/validation/final paired evaluations on a shared
server, not controlled kernel/inference speed benchmarks. Do not attribute the
long MELD wall time to a specific fault without resource traces. Adapter-only
best/last.pt include optimizer/progress/RNG/config and immutable anchor identity;
no duplicate backbone saved. Full predictions/masks remain in remote evaluation
NPZ files. COMPARISON.json records class recall/support, all15 conditions,
corrections/harms, gamma/read3×3 matrices, coverage, Gap norm ratio, validation
curves, mask hashes and actual missing/pattern distributions.

95 focused tests passed before launch; all52 TASR-specific tests passed again
after result collection. The complete paired analysis was rerun read-only into
`summary_verified`, independently of training, and completed successfully.
One old unrelated write-intervention test
is unsupported by Torch1.13(with_kwargs). It also failed on pre-TASR ea27c08 and
was not represented as passing. Real MELD/IEMOCAP6 GPU smoke passed baseline
strict replay/zero-on exact logits/CPU+CUDA RNG/frozen hashes/three updates/reload.
After zero-init, first step has2 nonzero-gradient parameter tensors and later
steps12/12, all finite. Complete analysis recomputed predictions, verified both
controls'30epoch mask traces, budget, selected checkpoint, F0 logits and cohorts.

Downloaded unmodified remote artifact SHA256 (verified before local RESULT.md
received the explicit final assessment):

- COMPARISON.json:2129874a4c9f871c9c61c848b10e0944a49d52af6a59b7b8a9e320f72b108698
- COMPARISON.csv:c43c5a3e6cb6436431109a43472ffbd56f766f515cf660bfd86f5d941a393f7d
- Original generated RESULT.md:3281b79d6e18f7cf0bae7d6ea1d80a4cfbf5d64ab50094726ed7a29f6fe8b4ca

## Reproduce and interpretation boundary

Use README.md's actual commands in the immutable training snapshot. Reporting:

```bash
python -m experiments.osram_tasr_20261018.summarize \
  --runs /home/qinjinghui/remote_experiments/osram_tasr_20261018/runs_v1 \
  --output /absolute/new/read_only_summary
```

Source algebra passed. IEMOCAP6 fold5 shows modest Persistent extra value over
F1 but harms V-only and does not universally improve random UA. MELD remains
essentially unchanged in task accuracy despite nonzero Gap correction. The
engineering random-ACC/W-F1 guard passes in both datasets, not a generalization
guarantee. These results alone do not justify pure-modality semantics, a resolved
MELD history bottleneck, or a robust HAMR contribution. No automatic expansion.
