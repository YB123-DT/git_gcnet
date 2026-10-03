# Grouping implementation audit

Date: 2026-10-03. Scope: `meaningful_blocks_grouping.py` and its dedicated tests.
Status: four complete grouping cores implemented; 21 CPU tests pass. This is
structural/numerical verification, not evidence of task performance or permission
to launch training. TDD and verification-before-completion were used.

## Interface and shared boundaries

`build_grouping(method, latent_dim, num_heads, value_dim)` returns a module with
`output_dim=128`. Its forward accepts Local `[N,D]`, current evidence `[N,4,H*V]`,
boolean active `[N,4]`, and availability metadata `[N,3]`. Active is authoritative;
availability does not generate an additional mask. The outer wrapper remains
responsible for excluding padding/no-history and for the zero-initialized affine
Flat bridge (weight **and bias**, followed by the outer history guard).

The owned tokenizer retains one Local and all real heads: with `D=256,H=8,V=64`,
33 typed storage positions, ordinarily 9/17/25 active. It uses LayerNorm, a 64d
Local projection, per-evidence-type projections and Local conditioning, type/head
embeddings, and GELU. Inactive values are sanitized before affine operations and
re-masked after biased memory operations. Empty rows bypass the core. There is no
temporal operation, memory call, external state cache, batch statistic, dropout,
label input, decoder, or auxiliary objective. The modules return learned features,
not object/pose estimates or completed missing modalities. Routing executes in
float32 with autocast disabled; original input dtype is restored on return.

## Source-to-implementation match

### `capsule_dynamic_routing`

Primary paper: [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829),
Procedure 1 and equations 1–3. Official code inspected:
[Sarasra routing/squash](https://github.com/Sarasra/models/blob/984fbc754943c849c55a57923f4223099a1ff88c/research/capsules/models/layers/layers.py),
revision `984fbc754943c849c55a57923f4223099a1ff88c` (Apache-2.0).

Kept: input/parent-specific 64→32 vote matrices, four parents, initial zero logits,
parent softmax, weighted sums, vector squash, and accumulated dot-product agreement
through three attached-gradient rounds. Output concatenates four 32d capsules.
Adaptations: typed read heads, zero-safe denominator, no optional routing bias,
visual backbone, margin loss, or reconstruction. Independent looped Procedure 1
matches numerically; a one-round counterexample differs, so this is not one-pass
attention under another name.

### `slot_attention`

Primary paper: [Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055),
Algorithm 1. Official code inspected:
[Google SlotAttention](https://github.com/google-research/google-research/blob/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/slot_attention/model.py#L43),
revision `e49bbfe381c9c0e564b937f1c4e163a2273c65cc` (Apache-2.0).

Kept: shared Gaussian initialization, four 64d slots, bias-free Q/K/V,
slot competition followed by input-normalized weighted means, GRU state updates,
LayerNorm/residual 64→128→64 MLP, and three refinements. The test expands GRU gates
independently instead of calling the same GRU cell. Adaptations: PyTorch GRU
parameterization, LayerNorm epsilon 1e-5, zero initial log-sigma, task-only symmetric
mean/max feature readout, and explicit RNG isolation. No set-prediction/image loss.

Training draws use only a CPU generator seeded by
`(torch.initial_seed()+104729) mod 2^63`; its byte state is a persistent buffer.
`configure_seed(seed)` is an optional pre-training setter, never a post-load hook.
Strict reload preserves the next draw. Evaluation uses a persistent seed-1729
noise buffer and mutates neither private state nor global RNG. All-inactive calls
also leave private state unchanged. Slot states themselves reset every utterance.

### `otke`

Primary paper: [A Trainable Optimal Transport Embedding for Feature Aggregation
and its Relationship to Attention](https://arxiv.org/abs/2006.12065), Definition 3.1.
Official code inspected: [OTKernel](https://github.com/claying/OTK/blob/2a4d3be10d305e76d55251d8f01c7f2ea7d9bf8c/otk/layers.py)
and [Sinkhorn](https://github.com/claying/OTK/blob/2a4d3be10d305e76d55251d8f01c7f2ea7d9bf8c/otk/sinkhorn.py),
revision `2a4d3be10d305e76d55251d8f01c7f2ea7d9bf8c`.
No repository license was verified; this implementation independently expresses
the mathematical adaptation and does not copy upstream code.

Kept: nonlinear normalized features, four learned supports, uniform source/target
masses, 30 differentiable log-Sinkhorn rounds at epsilon 0.5, and support-specific
aggregation. Paper scaling is `sqrt(4)*T.T@features`, not source code's globally
rescaled per-bin mean. Flattened bins project to 128d. Adaptations: neural ReLU map
instead of exact Gaussian Nyström features, no position encoding/pretraining.
Masked batched log-mass `-inf` gives the same plan as packing each utterance's
active rows; a dedicated NaN/backward test verifies equivalence. Tests also use an
independent probability-domain scaler and distinguish unconstrained row softmax.

### `capsule_variational_bayes`

Primary paper: [Capsule Routing via Variational Bayes](https://doi.org/10.1609/aaai.v34i04.5785),
Algorithm 1. Official code inspected:
[matrix votes](https://github.com/fabio-deep/Variational-Capsule-Routing/blob/78fb69dc10c71210fad8f456f1f2ce97766c4bb3/src/layers.py)
and [posterior/routing](https://github.com/fabio-deep/Variational-Capsule-Routing/blob/78fb69dc10c71210fad8f456f1f2ce97766c4bb3/src/vb_routing.py),
revision `78fb69dc10c71210fad8f456f1f2ce97766c4bb3` (Apache-2.0).

Kept: per-type 4×4 matrices and activations, input/parent-specific matrix votes,
four parents, activation-weighted full-covariance sufficient statistics,
Dirichlet/Gaussian-Wishart updates, expected-log/Mahalanobis responsibilities, and
three posterior rounds. Fixed priors are alpha=kappa=1, zero mean, inverse scale I,
nu=17. Cholesky solves use 1e-6 jitter; no explicit inverse in production. Final
means and entropy-derived activations form 68 values projected to 128d.

Disclosed source-code variant: activation is
`sigmoid(beta_a-exp(Elogpi)*entropy-beta_u)`; it is not claimed identical to paper
equation 14. Source post-routing BatchNorm is omitted for row independence. Tests
independently calculate full scatter/logdet and use explicit inverse only as an
oracle for three-round responsibilities. No ELBO, KL term, sampling, VAE, or
source spread loss is added. Original EM capsules were rejected at research time
because original-author code was not verified; this is the distinct VB substitute.

## Verification evidence

Environment: `/home/yangbin/miniconda3/envs/multimodalerc310/bin/python`,
PyTorch `2.2.2+cu121`, CPU, one intra-op thread. No GPU training launched.

TDD evidence: the initial 20 tests failed with the explicit missing-module
assertion before production code existed. The batched OT extension initially
failed on its unsupported mask argument, then passed after implementation.

Command:

```bash
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful_grouping.py -v
```

Result: **21 tests, 1.893 s, OK**. `py_compile` for the two Python files also passed.
Coverage includes seven availability patterns, poisoned inactive NaNs, zero
masked gradients, finite input/parameter gradients, active-zero and empty inputs,
zero empty rows after biased parameters change, batch independence, Local/Base
conditioning, repeated evaluation, strict load, CPU autocast, actual equations,
and private Slot RNG state/next-draw restoration.

Additional real-size synthetic check: seed 66, seven rows covering seven observed
modality patterns, `L256/H8/V64`, inactive reads set to NaN, three Adam steps at
1e-3 on mean-square output. Every parameter tensor changed; all outputs and all
input/parameter gradients were finite; inactive-read gradients were exactly zero.
These unit-check timings are not training-throughput estimates.

| ID | Parameters including tokenizer | Changed parameter tensors | Three CPU steps (s) |
| --- | ---: | ---: | ---: |
| capsule_dynamic_routing | 321,536 | 25/25 | 0.084430 |
| slot_attention | 109,696 | 45/45 | 0.070221 |
| otke | 88,512 | 29/29 | 0.076520 |
| capsule_variational_bayes | 67,677 | 49/49 | 0.090194 |

Remaining integration checks belong to the root lane: wrapper no-history/pad
guards after learning, unchanged OSRAM scan/read count, Flat-anchor/default-off
RNG equivalence, and full trainer checkpoint selection/resume. CUDA execution,
distributed RNG state behavior, task accuracy, and large-scale throughput are not
claimed verified here. The fixed-step Sinkhorn plan approximates its source row
marginal; the tested bounded toy problem meets tolerance, not a universal exact
convergence guarantee.
