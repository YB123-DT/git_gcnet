# Pre-launch validation

INTERNAL TEST-ORACLE SCREEN. Implementation checks are not performance evidence.

Baseline source: `515d785d2fb8e5407c1afc5192bb6ec9faa3184a`.

CPU regression command uses the existing `multimodalerc310` environment with
pytest from the installed `msa_extract` site-packages; no dependency installed.
Focused suite: **65 passed in 8.33 seconds**:

- 13 core second-read tests, 9 integration tests, 7 runner tests;
- existing paired V3 unit/integration, older connection integration, and runner
  regression tests (36 additional tests).

Checked disabled forward/backward/RNG and strict legacy state restore against
the baseline source; initial enabled predictions/RNG exact with both a fresh
zero Flat exit and a nonzero trained-like Flat exit; Local path once, CHM scan
once, unchanged writes/raw Gap, actual alpha-decayed pre-write state, strict
causality, no-history/padding, inactive-query exclusion and rank-deficient
projection gradients. Both variants share all initial parameters and RNG.

The span projection is differentiable `Q @ pinv(Q) @ q`, with `rtol=1e-5` and
`atol=1e-7`. No gradient is stopped into the original active query space. The
same thresholds define diagnostics in both modes. The numerical rank may
change discontinuously at the cutoff; finite gradients are verified on test
cases, not guaranteed for every possible training trajectory.

Added parameters: **429,062** in each default variant (166,406 Query MLP +
262,656 output projection). Only forward Base is corrected. No source tracker,
new classifier, new loss, extra Local computation or changed Memory write.

GPU smoke and formal training have not yet completed at this validation stage.
Their actual outcomes belong in the separate launch/results records, not in
this pre-launch CPU report.
