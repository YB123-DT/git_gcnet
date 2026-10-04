# Screening forty additional leads, not forty qualified experiments

LITERATURE SCREEN ONLY — NO NEW TRAINING

On 2026-10-04, forty source leads were screened: **9 retained for design review, 17 held, 14 rejected**. The request for forty meaningful, non-overlapping qualified methods is **not yet fulfilled**. None is approved for implementation or training. Existing jobs and the cap of sixty distinct trained methods are unchanged.

Deduplication includes the [existing forty](osram40_inventory.md), the [mechanism registry](../experiments/osram_method_registry.json), the [withdrawn basic twenty](../experiments/osram_readout20_20261003/CANDIDATES.zh.md), and [earlier reconsideration](../experiments/osram_readout20_reconsidered_20261003/RECONSIDERATION.zh.md), plus earlier gate/GRN/relation, completion and dual-view work.

## Contract

Use only existing Local256 and forward512 Base/Gap reads with masks. Do not change OSRAM read/write/query, task head or task objective. No extra memory, cross-batch support, completion or invented head/time/pixel geometry. Renaming, changing placement or width, swapping a solver, or wrapping a primitive in an MLP does not create a new method.

A design-review label is not a performance or final novelty verdict. Source fidelity differs: DPPy is an independent reference library and TensorFlow Lattice is a later author-project implementation. Source inspection is not runtime verification. No candidate package was installed or executed.

## Full screening ledger

|#|Source|Status|Reason|
|---:|---|---|---|
|1|[OptNet: Differentiable Optimization as a Layer in Neural Networks](https://proceedings.mlr.press/v70/amos17a.html)|held|Hold: no grounded coupled constraints; KKT alone does not distinguish it from prior implicit optimization.|
|2|[Determinantal Point Processes for Machine Learning](https://arxiv.org/abs/1207.6083)|design_review|Review: determinant-based joint subsets, not independent marginal gates.|
|3|[SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver](https://proceedings.mlr.press/v97/wang19e.html)|held|Hold: Boolean semantics and auxiliary-variable interpretation are not established.|
|4|[Robust Aggregation for Federated Learning](https://doi.org/10.1109/TSP.2022.3153135)|held|Hold: overlaps energy-based aggregation; a new robust potential alone is insufficient.|
|5|[Differentiable Submodular Maximization](https://arxiv.org/abs/1803.01785)|held|Hold: exact source code and gradient path without subset supervision unresolved.|
|6|[Differentiable Learning of Submodular Models](https://papers.neurips.cc/paper_files/paper/2017/hash/192fc044e74dffea144f9ac5dc9f3395-Abstract.html)|held|Hold: grid TV code does not establish the non-grid semantic-graph transfer.|
|7|[DSAC - Differentiable RANSAC for Camera Localization](https://arxiv.org/abs/1611.05705)|rejected|Reject: geometry solver and expected-risk objective do not fit the interface.|
|8|[SparseMAP: Differentiable Sparse Structured Inference](https://proceedings.mlr.press/v80/niculae18a.html)|held|Hold: no independent structure space beyond prior QP/OT mechanisms.|
|9|[Differentiable Dynamic Programming for Structured Prediction and Attention](https://proceedings.mlr.press/v80/mensch18a.html)|rejected|Reject: no valid ordered path; heads are not time steps.|
|10|[Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731)|rejected|Reject: changes training risk and cross-batch state.|
|11|[Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)|design_review|Review: base-point-dependent gyrovector algebra; hierarchy remains hypothetical.|
|12|[PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures](https://proceedings.mlr.press/v108/carriere20a.html)|held|Hold: differentiable feature-to-diagram pipeline is incomplete.|
|13|[BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation](https://arxiv.org/abs/2106.10994)|design_review|Review with high family-overlap risk: prior graph diffusion and new scattering are adjacent.|
|14|[Diffusion Scattering Transforms on Graphs](https://arxiv.org/abs/1806.08829)|design_review|Review: fixed multiscale wavelets with cascaded modulus paths, not one linear filter.|
|15|[Learning with Holographic Reduced Representations](https://arxiv.org/abs/2109.02157)|held|Hold: fixed keys can collapse to a linear map; dynamic keys need stronger novelty evidence.|
|16|[Direct Parameterization of Lipschitz-Bounded Deep Networks](https://proceedings.mlr.press/v202/wang23v.html)|design_review|Review: coupled parameterization bounds a branch, unlike independent norm clipping.|
|17|[Building Deep Networks on Grassmann Manifolds](https://doi.org/10.1609/aaai.v32i1.11725)|held|Hold outside shortlist: unjustified basis invariance; source also includes AFEW.|
|18|[Neural-Kernel Conditional Mean Embeddings](https://proceedings.mlr.press/v235/shimizu24a.html)|held|Hold outside shortlist: kernel objective and support structure violate the contract.|
|19|[KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)|design_review|Review: edge functions and nested sums, not an activation swap or invertible coupling.|
|20|[Deep Lattice Networks and Partial Monotonic Functions](https://arxiv.org/abs/1709.06680)|design_review|Review: calibrated multivariate interpolation and composition; no sentiment monotonicity claim.|
|21|[Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)|design_review|Review with logic-family proximity: truth-table operator selection, not NLM quantifiers.|
|22|[An evidential classifier based on Dempster-Shafer theory and deep learning](https://arxiv.org/abs/2103.13549)|held|Hold: inspected author code disagrees with standard ignorance-mass combination.|
|23|[Higher-Order Factorization Machines](https://arxiv.org/abs/1607.07195)|rejected|Reject: returns to withdrawn low-rank interaction family; exact code provenance unresolved.|
|24|[Tensorizing Neural Networks](https://arxiv.org/abs/1509.06569)|rejected|Reject: weight compression alone is not a new evidence mechanism.|
|25|[Density Modeling of Images using a Generalized Normalization Transformation](https://arxiv.org/abs/1511.06281)|rejected|Reject: normalization alone is too small; density learning requires a new objective.|
|26|[Iterative Normalization: Beyond Standardization towards Efficient Whitening](https://arxiv.org/abs/1904.03441)|rejected|Reject: normalization-family overlap and source batch dependence.|
|27|[Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505)|rejected|Reject: scalar quantization with generic wrappers is below the requested mechanism threshold.|
|28|[Vector Neurons: A General Framework for SO(3)-Equivariant Networks](https://arxiv.org/abs/2104.12229)|rejected|Reject: latent coordinates have no justified SO(3) action.|
|29|[HyperNetworks](https://arxiv.org/abs/1609.09106)|held|Hold: dynamic source LSTM uses scaling; full dynamic-matrix transfer not established.|
|30|[Dynamic Filter Networks](https://arxiv.org/abs/1605.09673)|rejected|Reject: no spatial support; dense reinterpretation merges with HyperNetworks.|
|31|[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)|design_review|Review: finite-time initial-value flow, not a DEQ equilibrium or extra history.|
|32|[Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439)|held|Hold: same dynamical family as ODE; conductance model cannot be reduced to a gate.|
|33|[N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)|held|Hold: double residual chain may still amount to a residual-MLP variant.|
|34|[Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://proceedings.mlr.press/v119/sun20b.html)|rejected|Reject: requires self-supervision and test-time parameter updates.|
|35|[Differentiable plasticity: training plastic neural networks with backpropagation](https://proceedings.mlr.press/v80/miconi18a.html)|rejected|Reject: relies on an additional writable associative state.|
|36|[An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity](https://doi.org/10.1162/NECO_a_00949)|rejected|Reject: changes credit assignment and local learning.|
|37|[Credit Assignment in Neural Networks through Deep Feedback Control](https://arxiv.org/abs/2106.07887)|rejected|Reject: changes training updates; inference cannot use unknown targets.|
|38|[B-cos Networks: Alignment is All We Need for Interpretability](https://arxiv.org/abs/2205.10268)|held|Hold: code inspected but full primary method pending; must exclude cosine-gate degeneration.|
|39|[Janossy Pooling: Learning Deep Permutation-Invariant Functions for Variable-Size Inputs](https://arxiv.org/abs/1811.01900)|held|Hold: full permutation averaging differs, but the relevant invariance is unmotivated.|
|40|[Learning Structured Text Representations](https://aclanthology.org/Q18-1005/)|held|Hold: tree marginals differ from softmax, but structural-attention family and tree prior remain issues.|

## Remaining overlap and evidence gaps

The nine provisional leads are DPP joint subsets, hyperbolic gyrovector networks, Bernstein graph filtering, diffusion scattering, Lipschitz sandwich networks, full KAN function composition, deep lattice composition, differentiable logic circuits, and finite-time neural ODEs.

They are not nine unrelated broad directions. Bernstein filtering and scattering share a spectral-graph family. Logic circuits are adjacent to the existing NLM family, and ODEs are adjacent to prior iterative/implicit designs. A broad-family interpretation of non-overlap requires further merging or rejection.

Important unresolved points:
- OptNet lacks task-grounded coupled constraints; KKT differentiation alone does not escape the prior DEQ/EA family.
- PersLay does not by itself close the differentiable path from current features to persistence diagrams.
- The inspected E-CNN author code applies three combination terms to the ignorance coordinate as well; this differs from the standard singleton-plus-ignorance rule. It is held, not silently repaired.
- B-cos author code was inspected but the complete primary method was not successfully retrieved.
- Fixed-key HRR binding/unbinding can collapse to a linear operator.
- FSQ, Tensor Train and HOFM do not meet the user's complete-mechanism threshold merely by adding generic wrappers.
- Sandwich bounds apply only to a correctly parameterized branch, not automatically to the unchanged Flat anchor, normalization, classifier or discrete mask changes.
- None of these mechanisms creates information absent from both the current input and the observed history.

## Handoff

Resolve concrete mathematical/source gaps before admitting any experiment. If a method only works by relaxing the loss, memory or data contract, keep it out rather than silently broadening scope. Forty qualified methods have not been found by this screen; this is not a proof that none exist.

The [paper bank](osram_next40_paper_bank.json) contains forty unique IDs, author arrays, primary URLs, code URLs, nearest prior methods and explicit statuses. Detailed evidence: [optimization](osram_next40_optimization.json), [geometry](osram_next40_geometry.json), [representation](osram_next40_representation.json), [conditional computation](osram_next40_conditional.json), [supplement](osram_next40_supplement.json). Benefits remain hypotheses; no new scores are reported.

Inventory baseline: `42d4904`, branch `feature/osram-uniform-forced-text`. Only literature records were added. Training source, checkpoints, accepted-method registry and remote queues were untouched.
