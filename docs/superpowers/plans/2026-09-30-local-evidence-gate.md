# Local-Conditioned Evidence Gate implementation plan

## Approved scope

Keep cfg84 no-JEPA random-missing Flat and all original parameters trainable.
Add an independent default-off gate before the original Flat history inputs.
No formal training is launched by this implementation task.

For each Base / Gap-A / Gap-T / Gap-V evidence, use shared learned projections
of Local and context to 128 dimensions. Concatenate these projections, their
product and absolute difference, their L2 norms, an 8-dimensional evidence-type
embedding and the three availability bits. A shared 128-hidden ELU MLP outputs
one scalar per evidence: `g = 1 + 0.2 * tanh(logit)`.
Scale original context, not the projected representation. Only the final MLP
layer is zero initialized. Preserve baseline initialization RNG.

Base is active for valid utterances; Gap is active only for a missing modality.
Use already ablated emotion contexts and safe `where` masking, including padding.
Regularizer is the mean `(g-1)^2` over active, valid evidence; multiply by
`lambda=1e-3` exactly once in training. No additional objectives or freezing.

## Execution and acceptance

1. Add failing tests in `tests/test_local_evidence_gate.py` before implementation.
2. Implement module and Flat integration in `gcnet_missing_m3/osram.py`; expose
   default-off configuration through model, trainer and fixed-pattern builder.
3. Integrate differentiable regularization and detached per-type diagnostics.
4. Verify off-path preservation, initialization identity, nonfinite inactive
   Gap/padding isolation, ablation isolation, regularizer reduction and finite
   gradients/parameter updates. Run existing related regression tests.
5. Verify on biggpu using its existing Python environment and host GPU0 only;
   no use of broken host GPU4 and no migration of model experiments.
6. Review scoped diff, archive exact verification evidence, commit with Lore
   trailers and push current branch to `github`, never upstream `origin`.

## Scientific limits

This changes four independent evidence strengths, unlike the earlier shared
history coefficient. Identity initialization and regularization do not ensure
improved downstream scores. No test-label oracle artifacts enter training.
Performance remains untested until a separately authorized matched experiment.
