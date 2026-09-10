# Base–Gap differences are substantial in magnitude, especially TV

Evaluation-only capture from the same five frozen MOSI causal Flat eta=.6
checkpoints. No training, prediction patch, memory alteration, Jacobian,
random-direction control or CKA. Scope matches the preceding interpolation:
AT/AV/TV exactly-one-missing samples, nonzero labels, rates .1–.7.

Each statistic is computed per sample/group, then averaged over rates within seed
and over seeds equally. These are representation units, not percentage points.

| Pattern | norm B | norm G | norm(B-G) | norm(B+G) | cosine | centered cosine | relative difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| AT | 10.290 | 7.023 | 7.625 | 15.869 | 0.646 | 0.552 | 0.449 |
| AV | 10.501 | 6.950 | 7.282 | 16.190 | 0.650 | 0.754 | 0.454 |
| TV | 10.390 | 3.102 | 9.795 | 11.892 | 0.314 | 0.234 | 0.731 |

Relative difference is **mean of per-sample norm(B-G)/(norm B + norm G)**,
not a ratio of the displayed means. Centered cosine independently subtracts the
B and G sample mean vectors within each seed × rate × pattern before computing
paired cosine; there is no cross-pattern or cross-seed centering.

## Across-seed stability

| Pattern | Range of five seed-mean relative differences | Range of seed-mean centered cosine |
|---|---:|---:|
| AT | 0.341–0.500 | 0.409–0.729 |
| AV | 0.332–0.565 | 0.658–0.857 |
| TV | 0.614–0.825 | 0.041–0.383 |

The result is not driven by a single seed: TV has a large relative difference
and low centered alignment across all five seeds. Its difference includes both
lower Gap magnitude and directional mismatch. It is not merely a pair of almost
identical vectors whose exchange would be trivially harmless.

Combined with the earlier fixed-sum interpolation (TV t=0 versus Normal W-F1
86.861 versus 86.848%), this supports the bounded observation: **a substantial
Base–Gap difference can be removed along this path with little aggregate task
performance change**. Some individual scores and decisions still change.

Do not yet conclude that the difference carries useful semantic information,
lies in a task-null space, or is uniquely suppressed by the network. Local may
dominate decisions, classifier scale may reduce sensitivity in many directions,
and score shifts may stay away from the decision boundary. A directional versus
matched-random perturbation comparison would distinguish some of these explanations,
but is **not executed in this audit**.

## Zero handling and precision

- Context tensors have 512 dimensions with an exactly zero 256-d backward half
  in the causal model. This is retained, matching the actual readout input.
- Mean zero-norm fractions are AT 4.81%, AV 3.81%, TV 3.84% for both Base/Gap.
- Zero-norm cosine is undefined and excluded, not assigned 0/1; valid counts are
  retained. Norm means include zero vectors. Centered means include the full group,
  including original zero vectors; centered cosine has its own valid counts.
- Captured tensors are FP32; magnitude/cosine calculations use FP64.
- Centered cosine quantifies paired residual alignment, not CKA or independence.

## Evidence and reproduction

- `magnitude.py collect ROOT`: one normal evaluation forward per batch, detached
  copies of Base and active Gap. Old diagnostic runner only gains an optional
  observer; default behavior and production model/trainer are unchanged.
- `magnitude.py analyze ROOT`: NumPy-only magnitude and centered-pair analysis.
- Red-to-green unit test **1 passed**, checking identical/opposite vectors,
  offset-invariant centering, undefined zero/singleton cases.
- 40/40 normal prediction arrays, labels, masks and sample order exactly match
  preceding saved outputs; all five checkpoint hashes match.
- Raw context shape and zero backward half verified across all 40 artifacts;
  all representation values finite. `git diff --check` passed.
- `magnitude_results/`: per-sample measurements, per-seed/rate/pattern
  mean/median/P90/counts, grouped summaries with seed SD, checkpoint metadata,
  source-context hashes and `RESULT.md`.
- Raw context vectors remain at
  `/data2/yb/remote_experiments/osram_magnitude_20260910/contexts/` on biggpu;
  the larger raw arrays are not duplicated into GitHub. Analysis values and hashes
  are uploaded. They can support subsequent approved audits without re-extraction.

**INTERNAL DIAGNOSTIC ONLY.** Existing historical eight-rate-mean Test-selected
checkpoints are fixed; no new epoch or hyperparameter selection is performed.
