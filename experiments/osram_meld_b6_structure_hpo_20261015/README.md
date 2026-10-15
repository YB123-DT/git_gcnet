# B6 internal HPO at fixed trial16 training parameters

INTERNAL TEST-ORACLE SCREEN. Authorized 2026-10-15. MELD seed66 only.

## Preregistered scope

20 actual training trials, each from scratch for50epochs, batch32. Fix original
trial16 learning_rate=0.0003433234536467373,
weight_decay=0.0000044857894231052665, dropout=0.22489055023388654.
Preserve official ECERC data, `gcnet-random-mask-v1`, cyclic0.0–0.7, no-JEPA,
original CHM/Flat1600/Local Skip and ACC-selected per-rate BEST checkpoints.
No new multi-seed, losses, gates, or resumed trial16 weights.

Search **only** two B6 structural parameters:

* Factor rank1/2/3. A group's effective rank is capped at its active role count.
* Learned unrolling steps4 through28, even increments; original14 included.

Rank is an approximation dimension, not a number of memory heads. A higher rank
may leave little sparse residual when few roles are active; it is not assumed
better. Step-specific thresholds and update sizes remain trainable, initialized
by the original formula at the chosen vector length. Gamma still starts at0.
No separate search of threshold/gamma initializations is included.

Initial5unique configurations: (1,14), (1,8), (1,24), (2,14), (3,14), run in
parallel. The first is a same-source control, not reuse of an old score.
After these complete, TPE with constant-liar pending-trial handling proposes
new, nonduplicate pairs. Use completed feedback as jobs finish; maintain no
more than12pending/running trials, at most4total compute processes per GPU
(including other queues). Source snapshots of old jobs are untouched.

## Evaluation and ranking

Preserve exact selected-epoch predictions/masks for all8rates. Also replay the
existing protected seed66 common masks, with second block on/off. Report both
separately. Optimize the previously registered paired objective
`0.6*delta_ACC + 0.4*delta_UA` against the current screen Flat; rank highest
Test-best ACC separately. No test-selected score is independent test evidence.

Save SQLite study, trial configs/failures, all BEST weights, exact prediction
artifacts, code/data/mask hashes, time/peak memory, CSV/JSON/RESULT.md.
Keep all prior accepted Flat24checkpoints untouched. Outputs go on `/home`,
not the nearly-full `/data2` filesystem.

## Execution

Use existing SSH `biggpu` and Python
`/data2/yb/conda_envs/cmerc-repro/bin/python`; compute-helper is unavailable.
GPU0/2/3 only, bind by verified UUID, never GPU4 or its UUID. A real worst-length
batch smoke of rank3/steps28 must pass before formal launch. No other tasks are
killed or paused. Shared GPU timing is not isolated benchmark throughput.

```bash
python -m experiments.osram_meld_b6_structure_hpo_20261015.run \
  --output /home/qinjinghui/remote_experiments/osram_meld_b6_structure_hpo_20261015 \
  --gpus 0,2,3 --per-gpu 4 --trials 20
```

Implementation order: regression tests → default-compatible config plumbing →
bounded asynchronous controller and provenance → immutable commit → CPU/GPU
checks → detached launch → observed epoch progress. No performance claim until
completed artifacts are verified.
