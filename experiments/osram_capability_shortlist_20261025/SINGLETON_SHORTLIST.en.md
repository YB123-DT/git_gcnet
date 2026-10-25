# Singleton-capable shortlist: research only

2026-10-25; repository anchor 4643a97. No implementation or training.
Literature Trace and Idea Convergence retain two provisional options for user selection.

## Scope and exclusions

GCNet random-mask-v1 rate .7 has one currently observed modality. A candidate must retain its core computation with singleton input, without requiring new raw features. This does not imply recovery of unobservable information.
The previous missing-aware ReconBoost sketch degenerates to ordinary task loss for singleton input.
TRAIN exemplar retrieval already exists: osram_exemplar_switch_20261023/RESULT.md reports label-aware ON minus original Flat +0.006pp mean W-F1 and -0.391pp high-missing. Do not relabel retrieval as untested.
PCD has an interesting distributional alignment formulation, but its official README still says code is coming soon. It is not a verified runnable implementation.
Generic MoE and target-private PLE are already present. The latter targets JEPA missing-target prediction; distinguish this from missing-pattern task representation specialization without claiming the whole family is new.

## Option 1: privileged-deviation-guided expert specialization

Paper: https://arxiv.org/abs/2606.30355
Code: https://github.com/seunghub/MARS
Reviewed SHA: f94cf9337082a763452974cf9aac3fd7a0dbb73b.
File: CASIA-SURF/classification/models/our_model.py.

MARS uses full/partial representation differences to guide training-time routing. Experts process partial features; a feature-only router imitates privileged routing for deployment. The reviewed NoisyGate uses only feature routing when residual is None, and detaches the privileged logits in its KL target.

Caveats: the paper uses full-minus-partial, the reviewed file uses partial-minus-full. The code evaluates all experts before weighting, so top-k does not imply sparse compute. Evaluation computes an unused full anchor; a transfer must remove that path and verify no dependence. Its modality sampler/adaptive sampling must not replace GCNet masks.

Proposed OSRAM transfer, NOT an implementation: task-side Local256 -> nonlinear expert block -> Local256 -> existing Flat with unchanged Base/Gap. Only a training privileged branch encodes full current input. Original missing Local still feeds CHM Key/Value/Query; no full-input writes or second scan. Whether task Local Skip follows the transformed Local must be locked before implementation and shared by both controls. Singleton A still activates experts from its A-derived Local.

Control: identical experts/capacity/interface/budget, ordinary feature router versus privileged-deviation guidance. Evaluate deployed feature routing, not privileged routing. Failure to beat ordinary MoE rejects the added guidance for this transfer. Full-input deviation may not be predictable from singleton input; regression and fixed-feature transfer remain unvalidated.

## Option 2: missing-conditioned parameter specialization

Paper: https://arxiv.org/abs/2511.11460
Code: https://github.com/Jiahuiqu/mamol
Reviewed SHA: 6d7dd482f8302d12b34a36d76976ec7013aa19ea.
File: momal_3modality/modules/vision_transformer_moe_3.py.

MaMOL combines shared/specialized lightweight experts with missing-conditioned routing. Transfer would give A-only and AT condition-specific task transforms rather than reweight already-read Base/Gap. This remains active for a singleton.

Not drop-in: the reviewed implementation thresholds a learned missing embedding, not raw availability bits, for static expert activation; it also checks None after tensor conversion. Audit semantics before adoption. Strong conceptual overlap with existing PLE/Soft MoE lowers novelty priority.

Control: capacity-matched shared versus availability-conditioned adapter, identical insertion and training/freezing. Optimization conflict is a hypothesis, not an established cause.

## Decision boundary

Prefer discussing option 1; retain option 2 as a conservative alternative. Await user choice. No model, sampler, environment, job or historical artifact was changed.
