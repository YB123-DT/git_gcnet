# Feature-only verification

Remote regression suite:59 passed (initial new test version); expanded feature-only
test file:8 passed on biggpu with CUDA_VISIBLE_DEVICES=6. Existing PyG deprecation
warning only. Tests verify original/full-gate initialization compatibility,
Level2 never executes (including softmax), altered Level2 weights have no effect,
active reweight1, masks/padding, finite gradients, joint updates and emotion-only
train_epoch loss. CLI and model builder carry the new flag.

Actual cfg84 dimensional check, using original seed66 config: train and eval max
prediction difference0.0 at initialization; CUDA RNG equal. Three finite joint
Adam steps updated original query/Flat/Local skip and feature output weights.
This small verifier ran on GPU0 before the user changed formal run placement.
All formal runs are bound to requested host GPU6; GPU4 never used.

No performance gain inferred from passing tests. No feature-only training had
been launched when the placement change arrived. Existing Flat and two-level
gate runs remain untouched. Source snapshot excludes unrelated local edits.
