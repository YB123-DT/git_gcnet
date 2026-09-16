# Fixed-teacher Stage-2: regression-only latent auxiliary

## Hypothesis

With the supervised frozen Teacher and the causal OSRAM / eta=.6 / Flat /
structured contextual MMoE Stage-2 unchanged, removing the InfoNCE term from
the missing-latent objective makes the regression auxiliary more suitable for
the emotion task.

This is a one-variable training-objective ablation, not a new architecture.

## Single change

Original frozen-teacher Stage-2:

```text
L_old = L_emotion + 0.05 * L_reg + 0.05 * L_NCE
```

Candidate:

```text
L_new = L_emotion + 0.05 * L_reg
```

The regression target, target mask, regression aggregation, rate weighting,
teacher source, Student initialization, OSRAM settings, MMoE, data, masks and
checkpoint-selection rule are unchanged. The contrastive output head remains
initialized and receives a forward pass, but its loss contribution is zero.

Implemented by:

```text
--training-objective joint-reg-only
```

The candidate uses the same frozen-Teacher two-stage runner as the original
joint configuration.

## Controls

| ID | Stage-2 objective | Role |
|---|---|---|
| N | `emotion-only` | fixed-Teacher no-JEPA control |
| R+NCE | `joint` | original regression + InfoNCE |
| R-only | `joint-reg-only` | candidate, regression only |

All three use the same frozen Teacher projector export, same Student
initialization seed, same data, same missing masks, same cyclic mixed-rate
budget and the same per-seed x per-rate Test-oracle selection rule.

The existing inherited `osram_causal_nojepa_20260910` result may be reported
as a secondary no-JEPA reference, but the fixed-Teacher `emotion-only` run is
the primary control for this comparison.

## Correctness checks before training

- `missing_m3_loss(..., include_contrastive=False).total == 0.5 * regression`
  and its contrastive value is exactly zero.
- regression gradient reaches shared Student parameters.
- frozen Teacher is not updated; inference forward is unchanged.
- `joint` remains exactly the old regression + contrastive behavior.

## Decision rules

1. Candidate is stably better than both R+NCE and no-JEPA: keep it and then
   investigate why this supervision is more suitable.
2. Candidate is better than R+NCE but close to no-JEPA: only claim that the
   simplified auxiliary has no obvious extra benefit; do not claim JEPA works.
3. Candidate has no stable advantage: stop this loss-simplification round.
   Do not immediately compensate with variance, target, predictor or memory
   changes.

This is a candidate-selection experiment, not yet a paper-level method claim.
