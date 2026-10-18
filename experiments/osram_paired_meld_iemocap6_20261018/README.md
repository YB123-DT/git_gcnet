# Paired history task-only transfer: MELD and IEMOCAP6

INTERNAL TEST-ORACLE SCREEN. Approved finite seed66 experiment, not a new module
or HPO sweep. Reuse the original MOSI A configuration: history_drop_prob=.2,
history_task_view2_weight=.5, history_contrast_weight=0.

For each train conversation, View1 uses the original GCNet cyclic random masks.
View2 deletes only observed modalities and restores one original observation if
needed. Both complete causal scans start with zero Memory and share parameters.
Loss = .5 * CE(View1) + .5 * CE(View2), supervising ALL valid utterances in both
views. Anchor selection does not limit task CE; InfoNCE has zero loss weight.
The existing inert training projector/computation is retained for exact reuse,
has no task-only gradient, and is never called by ordinary test inference.

No changes to Local, CHM read-before-write/joint Block Write/Key/Value/queries,
Base/Gap, Flat1600, Local Skip, classifier, data, environment or GCNet missing.
No persistent-pattern training, Nested, JEPA, completion, Gate, auxiliary loss,
additional seeds or additional configurations.

| Dataset | Seed | Folds | Epochs | Adam LR | Weight decay | Batch |
|---|---:|---|---:|---:|---:|---:|
| MELD | 66 | official one split | 50 | 1e-5 | 1e-5 | 32 |
| IEMOCAP6 | 66 | all five LOSO folds | 100 | 1e-3 | 1e-5 | 32 |

Configuration is copied from the accepted MELD BASELINE.json and the completed
IEMOCAP Flat config, then only the four paired settings above are overwritten.
Only two values differ from those defaults (paired enabled; contrast weight0).
Numerics remain immutable MELD b4a3138 and IEMOCAP785a012. Runtime is biggpu
/data2/yb/conda_envs/cmerc-repro/bin/python (Torch1.13.1+cu117/NumPy1.26.4).
Data SHA and original feature paths are verified. Existing baseline weights
and all prior result directories are read-only.

Six independent training jobs share healthy physical GPU0/3, at most3 jobs/GPU,
UUID-bound. GPU4 is forbidden. One actual first training epoch of MELD and of
IEMOCAP6 fold5 provides integration/gradient/saving and peak-memory validation;
remaining four folds automatically start once both pass, subject to resource
admission. No separate disposable training smoke. Initial6GiB/job is a
conservative estimate above the prior 2.1GiB single-view IEMOCAP measurement,
not a newly measured paired cost. Actual peaks override admission upward.

Reuse the persistent scheduler's ownership, locking, failure log, complete
epoch model/optimizer/RNG recovery and at most one lower-concurrency OOM retry.
The approved batch size/hyperparameters never change to accommodate concurrency.
Do not signal other tasks. Retain all eight BEST + exact selected artifacts and
last_training.pt for each new run; only unreferenced temporary cache versions
inside this experiment's own training_versions may be compacted by the inherited
state helper, never historical checkpoints.

MELD's original trainer requires the existing TrainingCapture to preserve
same-forward ACC-selected masks/logits/labels for UA/W-F1. IEMOCAP uses its native
preserve_selection_artifacts=True. Final randomly resampled evaluation is NOT
reported beside earlier best ACC. The inert projector is excluded, and only it
is excluded, when checking original MELD backbone initialization identity.
Actual MELD View1 train/test mask traces are compared to the accepted run each
epoch; no such full-stream equality is claimed without trace evidence for IEMOCAP.

After each complete training run, automatically evaluate seven persistent
observed sets A,T,V,AT,AV,TV,ATV. Singles use .7 BEST, pairs .3, full .0. Replay
the retained original selected logits exactly before frozen pattern inference;
projector invocation is prohibited. This is evaluation-only, not retraining
seven models. IEMOCAP scores are reported only after all five folds complete,
with equal fold weighting and the matching seed66 original Flat as control.

```bash
python -u experiments/osram_paired_meld_iemocap6_20261018/run.py \
 --output /home/qinjinghui/remote_experiments/osram_paired_meld_iemocap6_20261018/runs_v1 \
 --commit <actual-committed-driver-sha>
```

Run within tmux from the immutable deployed snapshot. Same-source recovery adds
--resume; never mix modified source/configs into an existing run. Queue logs,
per-epoch EPOCH_STATUS.json, FIRST_EPOCH_INTEGRATION.json, exact selection files,
full states and persistent predictions remain remote. On finite completion,
RESULTS.json and RESULT.md compare random8/high3 and Persistent7/macro6/macro7
against the existing seed66 Flat. This is one seed, not three-seed evidence;
source Test-oracle selection does not establish independent generalization.

Existing paired unit/training tests were verified in BOTH original numerical
checkouts (10 passed each, CPU). They include isolated RNG, nonempty subset masks,
Memory reset and loss weighting; those small training tests use MOSI, not a
substitute for this launch's real classification integration epoch.
