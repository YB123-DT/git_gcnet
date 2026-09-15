# Predicted Text readability versus affine-coordinate mismatch

Completed 2026-09-15. Two linear diagnostics only; no neural-model training,
architecture changes, loss changes, Test evaluation or checkpoint reselection.

## Locked setup

Reuse five frozen Student miss=.5 checkpoints, seeds 66–70, and their fixed
supervised Teachers. Validation predictions/targets/labels are loaded directly
from the preceding Text-transfer audit NPZs: identical cohorts and outputs.
Natural train missing=.5, schedule epoch=0; one frozen extraction per training
conversation. Current A, V and AV cohorts are fitted separately. Contextual
MMoE `reg_predictions` and original source averaging are unchanged.

1. **Predicted probe:** StandardScaler + Ridge(alpha=10, intercept enabled), fit
   on train predicted Text and continuous sentiment; evaluate validation
   predicted Text. Unlike the previous direct transfer, this fits a new decoder
   in prediction coordinates.
2. **Label-free alignment:** StandardScaler + multi-output Ridge(alpha=10,
   intercept enabled), fit train predicted Text → real Teacher Text pairs. No
   sentiment labels enter this fit. Feed mapped validation predictions to the
   unchanged Teacher Text probe fitted on all 1,284 real train Text latents.

All scalers are train-only. No validation regularization selection or threshold
calibration. Binary W-F1 excludes zero labels; regression fits include them.
This map is affine, not constrained to a pure orthogonal rotation. The Teacher
targets/probe themselves are task-trained; "label-free" refers only to fitting
the alignment map, not the provenance of the entire pipeline.

## Validation results

W-F1 percentages; five-seed mean ± sample SD. Columns compare identical samples
within each seed and pattern.

| Sources→Text | Real Text + original probe | Direct predicted transfer | New predicted probe | Label-free alignment + original probe |
|---|---:|---:|---:|---:|
| A→T | 86.44 ± 5.30 | 46.88 ± 13.45 | 52.08 ± 9.84 | 54.94 ± 6.11 |
| V→T | 87.06 ± 2.94 | 55.44 ± 8.67 | 53.15 ± 8.17 | 54.39 ± 7.89 |
| AV→T | 83.90 ± 7.79 | 54.56 ± 10.69 | 47.83 ± 7.69 | 54.79 ± 8.20 |

| Pattern | Train counts, seeds 66–70 | Validation counts | New probe train W-F1 mean |
|---|---|---|---:|
| A | 191/213/248/224/205 | 31/41/34/40/46 | 75.13 |
| V | 212/218/227/205/241 | 37/42/50/29/36 | 74.33 |
| AV | 153/158/146/160/137 | 34/29/37/29/32 | 70.51 |

## Conclusion

Neither diagnostic restores performance near real Teacher Text's 84–87%.
An affine correction helps A→T relative to its weak direct-transfer baseline,
but leaves a large gap. V and AV do not show substantial recovery. The data do
not support the strong explanation "sentiment is already well preserved and
only needs a simple coordinate alignment" under these fixed protocols.

However, do not claim the predictions contain zero sentiment information.
The new probes fit train at 71–75% and generalize to validation at 48–53%: limited
generalizable linear readability, finite sample effects and train/validation
distribution differences remain possible. With 137–248 training examples per
pattern against 256 latent dimensions, this single fixed-ridge audit is not an
information-theoretic impossibility proof or an exhaustive decoder search.
It also cannot identify whether weak prediction originates in the source inputs,
context, predictor optimization, task losses, or missing-target uncertainty.

Validation cohorts are small (29–50), Teachers were validation-selected and
Students previously Test-oracle-selected. This is internal diagnostic evidence,
not an independent generalization or significance claim. No further models,
regularization sweeps, nonlinear probes or new experiments were launched.

## Verification and artifacts

- Same Student checkpoint/Teacher hashes as the preceding audit.
- Whole-model state unchanged and predictor-on/off logits exactly identical.
- Train/validation conversation sets disjoint.
- Teacher probe's real and direct-transfer cached validation scores reproduced.
- Alignment API accepts only predicted and target latent arrays, no labels.
- Six related probe/alignment tests pass; compileall and diff check pass.

Code: `text_alignment_audit.py`; test: `tests/test_text_alignment_audit.py`.
Raw per-seed results and summary: `text_alignment_results/`.
Per-sample scores are kept remotely in `text_alignment_audit/*.npz`.
No model/trainer/loss files were modified.
