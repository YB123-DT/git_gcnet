# Stage1 Teacher information-exit audit

Completed 2026-09-15. No backbone/Student training, no Test evaluation.
Five supervised Stage1 Teachers, seeds 66–70, their existing validation-selected
best.pt checkpoints. Only downstream ridge probes were fitted.

## Locked protocol

- Complete A/T/V on train and validation; frozen model in eval/no-grad.
- Extract the trained ONLINE `observed_set.projectors` outputs, not dormant EMA
  `teacher.*`; hook the same forward's fused node; use returned pre-classifier hidden.
- Train: 1,284 utterances, 52 conversations. Validation: 229 utterances,
  10 conversations. Train/validation conversation sets are disjoint.
- Fit StandardScaler and Ridge(alpha=10, cholesky) only on train. Fixed alpha
  inherited from the existing complete-state audit; no validation hyperparameter search.
- Fit continuous MOSI sentiment including zero targets; report binary Accuracy
  and weighted F1 excluding zero labels (216 validation utterances), threshold >0.
  MAE in raw JSON includes zero labels.
- No MMoE/EMA forward, optimizer step or parameter update. Whole-model state hash
  is unchanged for all seeds. The original head applied to extracted hidden
  reproduces Teacher validation W-F1 for all five checkpoints.

## Results

All values below are percentages; SD is sample SD across five seeds, not an
uncertainty estimate across independent validation datasets.

| Representation | Dimension | Validation ridge W-F1 mean ± SD | Accuracy mean |
|---|---:|---:|---:|
| Audio projector | 256 | 53.07 ± 1.89 | 54.63 |
| Text projector | 256 | 85.55 ± 0.82 | 85.56 |
| Visual projector | 256 | 55.70 ± 6.38 | 55.74 |
| Complete fused node | 256 | 83.87 ± 1.65 | 83.89 |
| Complete final hidden | 700 | 83.96 ± 1.63 | 83.98 |

For reference, the original jointly trained Teacher head has validation W-F1
85.83%. This is not another ridge-probe result and is not Test performance.

## Interpretation and limits

The proposed pattern "all modality projectors weak, final hidden much stronger"
does not appear under this fixed linear-probe protocol. Text already exposes
strong sentiment information, while Audio/Visual have much weaker standalone
linear readability. A blanket claim that the Teacher knowledge exit is wrong
is not supported, nor is switching to final hidden justified by these probes.

This does not prove Audio/Visual contain no emotion information, that multimodal
fusion is harmful, or that Text can be predicted from Audio/Visual. Ridge measures
linear readability with one fixed regularization setting, not all usable information.
Final hidden also has 700 dimensions versus 256 for the other exits and includes
conversation context. The new ridge head is not equivalent to the original
jointly trained task head; its lower score does not imply information was destroyed.

Crucially, target informativeness and cross-modal predictability are separate:
the Text target is useful, but this audit does not establish whether the MMoE can
recover its sample-specific content when Text is missing. That remains untested
in this audit. No second/third-layer audit or replacement model was launched.

The Teachers were already selected using this validation set, so this is a
representation diagnostic, not independent generalization evidence. No label-
shuffled baseline, dimension-matched probe or significance claim is included.

## Reproduction and evidence

`information_audit.py` runs all five seeds using the existing remote paths and
Python environment. Output refuses to overwrite an existing audit directory.
JSON artifacts: [information_audit_results](information_audit_results/summary.json).
Individual checkpoint hashes, selected epochs, split IDs, per-seed metrics and
integrity checks are recorded in `seed_66.json` through `seed_70.json`.
Per-sample validation predictions remain in remote `information_audit/*.npz`.

Tests: two probe tests pass (train-only scaling/fixed alpha; validation labels
cannot change predictions). Real five-checkpoint extraction and integrity checks
pass. Python compilation and `git diff --check` pass. No model/loss code changed.
