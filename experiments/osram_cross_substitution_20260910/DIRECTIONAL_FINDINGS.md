# Go / No-Go: NO-GO for a specially suppressed Base–Gap difference direction

Five fixed MOSI causal Flat eta=.6 checkpoints, exactly-one-missing AT/AV/TV,
nonzero labels; rates .1–.7. No training, model changes, CKA or new checkpoint
selection. The frozen readout's raw regression score is the differentiated output.

## Controls and execution

- Real-D: `[-D_i,+D_i]`, D=(B-G)/2.
- Shuffled-D: another nonzero-D sample from the same seed/rate/pattern, with
  no self donor; randomized cyclic derangements, 8 fixed deterministic draws.
- Random-antisymmetric: `[-r,+r]`, r only in the forward256 dimensions.
- Symmetric: `[+r,+r]` using the same r as its antisymmetric control.
- All four **joint two-slot directions have unit norm**. Rescaling donor/random
  D to norm(D_i) before normalizing cancels algebraically; the implementation
  directly constructs the identical unit directions. No perturbation reaches the
  never-observed zero backward half.
- 315 of 7,714 eligible seed/rate/sample records have zero D and are excluded from
  all four directional comparisons. They have no defined Real-D unit direction.
  All remaining 7,399 records have at least one other nonzero donor in their group.
- Random controls are empirical or support-restricted directions; shuffling a
  real direction onto a different sample is **not proof of an on-manifold path**.

Sensitivity is `abs(grad_B f · v_B + grad_G f · v_G)` for jointly unit-normalized v.
There is no emotion/JEPA loss here and no gradient to any model parameters.
Eight-draw control sensitivities are averaged per sample before group aggregation.

The previous magnitude cache lacked Local. One normal evaluation pass per batch
therefore captured Local, verifying cached B/G bitwise before use. The actual
audit then evaluates **only the frozen emotion readout**, in FP64. OSRAM is not
part of the differentiated graph; Local is a fixed constant.

## Main result

All sensitivities below are displayed in **10^-3 raw-score units per unit input
direction**. Rates are averaged equally within seed, then seeds equally. For All,
patterns are sample-weighted within each seed/rate.

| Pattern | Real-D | Shuffled-D | Random-antisymmetric | Symmetric | Real / random |
|---|---:|---:|---:|---:|---:|
| All | 3.352 | 3.409 | 0.877 | 2.035 | 3.82× |
| AT | 3.199 | 3.267 | 1.026 | 2.334 | 3.12× |
| AV | 3.170 | 2.961 | 0.836 | 2.140 | 3.79× |
| TV | 3.626 | 3.948 | 0.772 | 1.646 | 4.70× |

Real-D exceeds random-antisymmetric in **105/105 seed/rate/pattern group means**,
including TV **35/35**. All-pattern Real/Shuffled ratio is 0.983; TV ratio is
0.918, not an order-of-magnitude suppression. TV seed69 even has higher Real-D
than Shuffled-D sensitivity. There is no evidence for S_D much less than both
matched real and random controls, or for a broadly insensitive difference subspace.

## TV across seeds

Same 10^-3 units; rate averages within each seed.

| Seed | Real-D | Shuffled-D | Random-antisymmetric | Symmetric |
|---|---:|---:|---:|---:|
| 66 | 2.945 | 3.272 | 0.736 | 1.535 |
| 67 | 4.352 | 5.537 | 0.880 | 1.920 |
| 68 | 3.428 | 3.660 | 0.805 | 1.718 |
| 69 | 5.837 | 5.530 | 0.967 | 2.212 |
| 70 | 1.566 | 1.741 | 0.469 | 0.846 |

Symmetric Gaussian directions are more sensitive than antisymmetric Gaussian
directions overall, but **less** sensitive than the empirical Real-D direction.
Thus there is no basis to claim that only the common/symmetric evidence matters.

## Numerical verification

- Centered FD uses **epsilon=1e-3 and 5e-4**, fixed before the run, for Real-D and
  draw0 of all three controls on every included sample.
- Maximum absolute FD/autograd sensitivity discrepancy: **2.1282e-11** across
  the 105 groups. Group means/maxima saved in CSV. FD columns are draw0 whereas
  main control columns average 8 draws; do not compare these as if identical draws.
- FP64 frozen-readout score versus original FP32 inference: maximum absolute
  difference **6.6517e-7**.
- Direction unit norm, antisymmetric/symmetric signs, zero backward support,
  no-self shuffle and linear analytic FD verified in a red-to-green test: **1 passed**.
- Frozen readout state dictionaries loaded strictly, all parameters have
  requires_grad=False and no parameter gradients after differentiation.
- `git diff --check` passed. Existing audit/model behavior remains intact.

## Decision

**NO-GO for the proposed specially suppressed / near-null difference direction
story on this setup. Stop this representation-analysis line here.**

The score is responsive to genuine Base–Gap differences, more than to a generic
matched antisymmetric direction. Prior W-F1 stability therefore cannot establish
that D is ignored. It is compatible with nonzero score changes mostly not crossing
the classification threshold, as the earlier interpolation sign-flip audit showed.
This experiment alone does not prove a complete boundary-distance explanation or
globally rule out other insensitive directions. No new module or loss is proposed.

## Artifacts

- `directional.py`: Local capture and readout-only directional evaluation.
- `directional_results/per_sample.csv`: four mean sensitivities, control draw SD,
  raw-score distance from zero, D norm, first donor index, two-step FD values.
- `per_seed_rate_pattern.csv`: all 105 groups, zero-D counts, sensitivities,
  sample-wise paired order frequencies, and verification errors.
- `per_seed_summary.csv`: separate AT/AV/TV and pooled summaries for all five seeds.
- `PROVENANCE.json`: checkpoint hashes, seeds/draw policy, normalization and FD definition.
- Existing normal-eval seed JSONs preserve mask/epoch provenance.
- Local arrays and normal predictions remain on biggpu under
  `/data2/yb/remote_experiments/osram_directional_20260910/`; Base/Gap cached under
  `/data2/yb/remote_experiments/osram_magnitude_20260910/contexts/`.
- Reproduce readout audit without another OSRAM pass using the remote s0 Python:
  `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python directional.py analyze ROOT` (use
  the full script path from the repo root and the existing remote ROOT).

**INTERNAL DIAGNOSTIC ONLY.** Historical eight-rate-mean Test-selected checkpoints
are fixed. No formal significance, equivalence or population-null-space claim.
