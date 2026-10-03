# Independent feature-reasoning review

Date: 2026-10-03. Reviewer: decision_impl (independent of the implementation owner).
Verdict: **APPROVE after the NODE leaf-numbering correction**. No remaining
blocking finding in the reviewed TabNet/NODE/common implementation.

## Scope and source contract

Read `meaningful_blocks_feature_reasoning.py`, `meaningful_blocks_common.py`,
`tests/test_meaningful_feature_reasoning.py`, and the accepted feature cards.
Compared the distinguishing cores with the pinned primary implementations:

- TabNet: `google-research/google-research` revision
  `e49bbfe381c9c0e564b937f1c4e163a2273c65cc`, `tabnet/tabnet_model.py`.
- NODE: `Qwicen/node` revision
  `3bae6a8a63f0205683270b6d566d9cfa659403e4`, `lib/odst.py` and
  `lib/nn_utils.py`.

TabNet preserves GLU value-times-sigmoid gating, square-root-one-half residual
scaling, the full-input warmup followed by three decision passes, eligible-only
sparsemax of prior-times-attention logits, `prior *= 1.5 - mask`, and accumulated
ReLU decision features. Per-utterance LayerNorm replacing batch statistics is
the documented adaptation, not an original-benchmark reproduction claim.

NODE preserves shared sparse feature scores, depth-four path products, vector
leaf responses, and three densely connected ensembles. Mask-dependent raw
feature eligibility and train-only private-RNG initialization are documented
adaptations. Common tokenization sanitizes inactive input before projections
and hard-masks projected inactive output.

## Finding and accepted correction

The initial NODE implementation complemented the pinned source's leaf index:
it selected `entmoid(z)` for bit one rather than bit zero. With responses equal
to leaf IDs 0 through 15 and all split logits 0.7, it returned 11.0968247716;
the source convention returns 3.9031752284. These are representationally
equivalent only after also reversing the response leaf axis.

The owner swapped the two `torch.where` arms and added
`test_node_leaf_numbering_matches_source_bin_codes`. Independent rerun confirms
the corrected convention. This was a source-equivalence issue, not evidence
of training divergence or performance differences.

## Independent numerical evidence

Read-only CPU probes checked more than the initial unit-test fixtures:

- Two-class entmax1.5 versus the source's explicit entmoid1.5 formula:
  maximum float64 error `2.220446049250313e-16`.
- Masked sparsemax/entmax1.5 at widths 2, 7, 579 and input scales 0.01, 1, 100,
  compared with their analytic support Jacobians: maximum gradient error
  `3.552713678800501e-15` in float64 and `3.4570693969726562e-06` in float32.
  Maximum simplex-mass errors were `3.4416913763379853e-15` and
  `3.5762786865234375e-07`, respectively. Excluded outputs were exact zero.
- NODE initialization with four rows spanning two eligibility patterns,
  independently reconstructed using `torch.quantile` over all four rows:
  threshold error `4.440892098500626e-16`; maximum-distance log-temperature
  error `3.3306690738754696e-16`.
- The first training initialization preserved global Torch, Python, and NumPy
  RNG states. A poisoned excluded simplex coordinate had zero output and
  zero input gradient while valid gradients remained finite.

Verification command (environment Python, no dependency installation):

```text
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_feature_reasoning tests.test_meaningful_set_context -v
```

Observed after the correction: **13 tests passed in 2.764 seconds**.

## Remaining nonblocking coverage recommendation

Persist the broader randomized simplex/Jacobian, mixed-pattern all-row
quantile, three-library RNG, and manual TabNet-chain fixtures as regression
tests. The review probes establish current behavior but are not a substitute
for durable coverage. CUDA, full trainer recovery, and performance were outside
this feature-core review. No production files were edited by the reviewer.
