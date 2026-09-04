# MOSI six-direction shared-expert geometry

## Question

Two confounds were tested before starting the Target-Private A/B:

1. Are near-zero direction cosines merely caused by Top-K routes selecting
   disjoint experts?
2. Are same-target direction pairs more compatible than cross-target pairs?

## Exact scope

- Existing Missing-M3 MOSI all-rates checkpoint, seed 66, best epoch 44.
- All non-zero missing rates, using the full training split.
- Only singleton observed patterns (A, T, V) are included. In these samples a
  prediction is exactly one model direction rather than an average of two
  source predictions.
- Gradients contain only `missing_predictor.mmoe.experts.*`. Target heads,
  gates, source/target embeddings and the rest of the predictor are excluded.
- Each direction uses the model's SmoothL1 and InfoNCE target objective.

## Aggregate result

| group | pairs | shared-expert cosine | common-expert cosine | routing overlap | gradient-mass overlap |
|---|---:|---:|---:|---:|---:|
| same target | 3 | +0.263 | +0.263 | 0.838 | 0.898 |
| cross target | 12 | +0.021 | +0.021 | 0.789 | 0.878 |

The common-expert cosine is effectively identical to the global cosine. Thus
the near-zero cross-target result is not created by padding disjoint expert
blocks with zeros. Both groups share most routing and gradient mass.

## Same-target comparison by missing rate

| rate | same-target cosine | cross-target cosine | same-target routing overlap | cross-target routing overlap |
|---:|---:|---:|---:|---:|
| 0.1 | -.022 | -.002 | .843 | .778 |
| 0.2 | +.137 | +.000 | .852 | .789 |
| 0.3 | +.082 | +.016 | .836 | .785 |
| 0.4 | +.181 | +.021 | .839 | .786 |
| 0.5 | +.214 | +.029 | .840 | .788 |
| 0.6 | +.211 | +.024 | .836 | .786 |
| 0.7 | +.203 | +.004 | .836 | .793 |

Rate 0.1 contains few singleton samples and is the sole exception. From 0.2
through 0.7, same-target gradients are consistently more aligned.

## Individual aggregate pairs

The three same-target pairs are:

- `A->T | V->T`: +0.139
- `A->V | T->V`: +0.330
- `T->A | V->A`: +0.320

Most cross-target pairs are near zero. The lowest is
`V->A | V->T = -0.069`; this is a same-source, different-target pair.

## Verdict

1. Top-K routing does provide mild implicit specialization: same-target routes
   overlap about 0.049 more than cross-target routes.
2. It does **not** separate directions into nearly disjoint expert sets. Route
   overlap remains high for both groups, and all shared experts carry gradient.
3. The predictor has clear target-conditioned gradient geometry:
   `C_same-target > C_cross-target` is supported on MOSI seed 66.
4. This supports target-aware parameter sharing as a mechanism hypothesis, but
   does not yet prove that cross-target near-orthogonality reduces emotion F1.
   That performance claim still requires the paired Target-Private A/B.

Raw values are stored in `mosi_six_direction_geometry.json`.
