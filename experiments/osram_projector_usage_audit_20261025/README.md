# Projector usage audit

INTERNAL DIAGNOSTIC ONLY. No training or parameter recycling authorized here.

Scope: MOSI seed66 accepted cmerc-repro Flat, new GCNet random-mask-v1,
one fixed protected miss0.3 BEST checkpoint. Four training batches (batch32)
at each of eight rates, same conversation cohort, original sampler including
host/guest and padded rectangle. No test labels or test inference.

Capture actual fc2 inputs (GELU outputs; dropout disabled in eval), obtain
gradients of sum valid-utterance MSE by autograd.grad. These gradients include
effects on later utterances through CHM; pattern groups label the SOURCE
utterance, not each downstream loss target. This is not an independent
per-utterance Jacobian nor a direct measure of causal utility.

Compute mean absolute activity and gradient for each channel/availability.
Normalize each by its own condition's channel mean. Report descriptive joint
low-use thresholds .01/.05/.1 and intersection across all conditions containing
the modality, requiring at least32 sample slots per condition. No condition
support means N/A, not zero. Repeated masks of a conversation are not independent
observations. Single-checkpoint results do not establish persistent inactivity.

Checks: no-hook/hooked logits bitwise equal; parameters unchanged; no parameter
grad accumulation; hooks consume no Torch CPU/CUDA RNG; rate0 all observed,
rate.7 exactly one observed; padding zero; synthetic statistics tests.

Entry: `python -m experiments.osram_projector_usage_audit_20261025.run --output UNIQUE_PATH --gpu-uuid HEALTHY_UUID --source-commit SHA`

Use `/data2/yb/conda_envs/cmerc-repro/bin/python` on biggpu, physical GPU0
if still free at launch, never GPU4. Snapshot code before running; no existing
jobs or checkpoints modified. Full JSON includes all channel/condition means,
configuration, environment, checkpoint hash, mask hashes and conversation IDs.
Masks stay in the independent remote output; only small JSON/Markdown tracked.
