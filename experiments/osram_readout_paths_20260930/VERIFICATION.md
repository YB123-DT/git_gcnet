# Verification

Six local tests passed: cache/formula/masks/full Skip bias/model-state and
layout equivalence; MOSI neutral filtering and weighted-F1 transitions; stable
classification CE; nested prediction artifacts and neutral-only groups.
Remote first five tests passed before the final nested-artifact test was added.

Real checkpoint smoke: seed66, rate.7, official validation10conversations,
229utterances,13 settings. Complete with exact identity logits and equal model
state hashes before/after. CPU analysis of its artifact also completed.

Initial smoke failed strict equality by max2.3841858e-7. Boundary audit showed
adapter output/pre-norm sum/hidden values identical; transposed-mask torch.where
produced noncontiguous hidden, selecting a different CUDA Linear calculation.
Preserving safe masking then making hidden contiguous restores the original
layout and bitwise equality. Regression test failed before the fix and passed
after. Identity tolerance was not relaxed. Original failed smoke remains remote.

No weights changed, no optimizer, no model-core modifications. No test inference.
Smoke is interface validation, not evidence for choosing any coefficient.
IEMOCAP independent-validation limitation documented in README.
