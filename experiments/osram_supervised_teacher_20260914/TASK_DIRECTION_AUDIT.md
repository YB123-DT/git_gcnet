# Text task-direction accessibility audit

Completed 2026-09-15. Five existing frozen Students/Teachers; no model training,
Test evaluation, checkpoint reselection or inference changes. Natural miss=.5,
same validation A/V/AV cohorts as the prior transfer audit.

## Exact direction

The train-fitted Text probe is StandardScaler + Ridge(alpha=10). Convert to raw
latent coordinates before forming the scalar:

`w_raw = ridge.coef_ / scaler.scale_`

`b_raw = ridge.intercept_ - scaler.mean_ @ w_raw`

`r_star = z_teacher @ w_raw`, `r_hat = z_prediction @ w_raw`.

Then `r + b_raw` exactly represents the original probe decision score. MAE and
Pearson correlation compare the r scalars; binary decisions retain b_raw, not
the incorrect threshold r=0. This is one fixed probe's sentiment direction,
not an assertion that all task information occupies one dimension.

## Observed-input reference

Separate StandardScaler + Ridge(alpha=10) fitted on each train pattern:

- A: `[z_A; Base; Gap_Text]` → r_star (1,280 input dimensions).
- V: `[z_V; Base; Gap_Text]` → r_star (1,280 dimensions).
- AV: `[z_A; z_V; Base; Gap_Text]` → r_star (1,536 dimensions).

These are actual observed Student latents and actual causal pre-write reads.
No missing Text latent or complete target enters the input. Full Text is used
only to produce training supervision r_star. The ridge does not fit sentiment
labels directly; the fixed Teacher Text probe was previously fit with labels.
No nonlinear MLP, hyperparameter tuning or feature selection was added.

This bypasses the original predictor's context projection and source-plus-context
compression and is an empirical linear-accessibility reference, **not a proven
upper bound** on available information or achievable predictor performance.

## Validation results

Five-seed means. W-F1 is percentage; MAE is on the original probe-score scale.

| Pattern | Teacher Text W-F1 | MMoE direction correlation | MMoE r MAE | MMoE W-F1 | Input ridge correlation | Input ridge r MAE | Input ridge W-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 86.44 | .112 | 1.157 | 46.88 | .063 | 1.367 | 54.04 |
| V | 87.06 | .202 | 1.234 | 55.44 | .014 | 1.353 | 51.71 |
| AV | 83.90 | .137 | 1.075 | 54.56 | −.0004 | 1.452 | 52.82 |

W-F1 sample SD across seeds: MMoE A/V/AV = 13.45/8.67/10.69 pp;
input ridge = 7.95/9.92/14.86 pp. Small cohort uncertainty is material.

Eight within-pattern target permutations; positive values indicate a real-pair
advantage. These are descriptive averages, not significance tests.

| Pattern | MMoE real−shuffle correlation | MMoE shuffle−real MAE | Input ridge real−shuffle correlation | Input ridge shuffle−real MAE |
|---|---:|---:|---:|---:|
| A | .088 | .050 | .071 | .065 |
| V | .178 | .061 | −.020 | .0001 |
| AV | .127 | .056 | −.001 | .054 |

MMoE/Teacher sign agreement (using the original intercept) is 59.63/57.67/58.84%
for A/V/AV; input ridge agreement is 51.74/55.70/50.75%. This differs from sentiment
accuracy because the reference sign is the Teacher probe, not the true label.

## Train-versus-validation gap

| Pattern | Input ridge train correlation | Train W-F1 | Validation W-F1 |
|---|---:|---:|---:|
| A | .899 | 87.48 | 54.04 |
| V | .877 | 86.51 | 51.71 |
| AV | .972 | 88.92 | 52.82 |

Train counts are A 191–248, V 205–241, AV 137–160; validation 29–50 per pattern.
The reported context dimensions include causal backward zeros, which remain
constant under the train-fitted scaler. Even excluding those zero columns,
the input dimension is large relative to each training cohort.

## Interpretation

The weak MMoE reconstruction is not confined to the other 255 latent directions:
prediction of the fixed sentiment direction itself is weak. However, the direct
observed-input ridge also fails to generalize that direction well. There is no
positive evidence here that a simple linear predictor could recover strong
Teacher sentiment while the MMoE alone discards it.

Do not conclude A/V/context contain no information. The strong train fit and poor
validation performance highlight estimation/generalization limitations; finite
samples, representation quality, conditioning, regularization and source-target
ambiguity remain unresolved. A negative result for one fixed ridge is not a
mathematical upper bound or a proof of cross-modal impossibility.

The MMoE r-derived W-F1 necessarily reproduces the earlier direct-transfer W-F1:
it is the same fixed probe score rewritten as a scalar. The new evidence is the
scalar correspondence and the observed-input accessibility reference, not a new
independent F1 experiment. No new architecture is justified automatically.

## Verification and artifacts

- Same Student and Teacher checkpoint hashes; frozen state unchanged.
- Predictor-on/off classification logits identical.
- Same validation labels and cached probe scores reproduced numerically.
- Standardization/intercept conversion tested; train/validation IDs disjoint.
- Eight related probe tests pass; compileall and `git diff --check` pass.
- Code: `task_direction_audit.py`; tests: `tests/test_text_task_direction.py`.
- JSON results: `task_direction_results/`; per-sample scalar NPZs remain remote.

Teachers were validation-selected and Students previously Test-oracle-selected.
This remains an internal diagnostic, not independent held-out evidence.
