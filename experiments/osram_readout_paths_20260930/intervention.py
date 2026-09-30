"""Frozen, paired Flat readout interventions; never rerun the memory trajectory."""
from dataclasses import dataclass
from typing import Callable

import torch


SETTINGS = tuple((level, level, memory) for level in (.8, 1., 1.2)
                 for memory in (.8, 1., 1.2)) + (
    (.8, 1., 1.), (1.2, 1., 1.), (1., .8, 1.), (1., 1.2, 1.))
IDENTITY_INDEX = SETTINGS.index((1., 1., 1.))


@dataclass(frozen=True)
class FlatReadoutCache:
    adapter_input: torch.Tensor
    skip_output: torch.Tensor
    valid: torch.Tensor
    original_logits: torch.Tensor


def _validate_model(model):
    if any(module.training for module in model.modules()):
        raise ValueError('Diagnostic requires model.eval(), including every submodule')
    backbone = model.osram
    if backbone.osram_readout_fusion != 'flat':
        raise ValueError('Only original Flat is supported')
    if any(getattr(backbone, name, 'full') != 'full'
           for name in ('osram_ablation', 'osram_emotion_ablation')):
        raise ValueError('Existing readout ablations must be disabled')
    for owner, names in ((backbone, ('osram_post_grn', 'osram_history_input_gate',
                                   'osram_local_evidence_gate', 'osram_hierarchical_evidence_gate')),
                         (model, ('local_context_residual', 'classification_completion',
                                  'pretrained_completion'))):
        if any(getattr(owner, name, False) for name in names):
            raise ValueError('Readout adaptations/completion must be disabled')
    if getattr(model, 'readout_type', 'shared') != 'shared':
        raise ValueError('Original shared task head is required')
    return backbone


@torch.no_grad()
def capture_flat_readout(model, forward_call: Callable, umask: torch.Tensor) -> FlatReadoutCache:
    """Run the supplied original forward exactly once and detach its Flat inputs.

    ``umask`` is [batch, length]. ``forward_call`` returns the usual model tuple
    whose first element is logits. Hooks are removed even on forward failure.
    No parameters, requires_grad flags, train/eval flags or buffers are changed.
    """
    backbone = _validate_model(model)
    inputs, skips = [], []
    handles = [backbone.emotion_adapter.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone())),
        backbone.local_skip.register_forward_hook(
        lambda module, args, output: skips.append(output.detach().clone()))]
    try:
        output = forward_call()
    finally:
        for handle in handles:
            handle.remove()
    if len(inputs) != 1 or len(skips) != 1:
        raise ValueError('Expected exactly one original Flat adapter and skip call')
    logits = output[0] if isinstance(output, (tuple, list)) else output
    valid = umask.T.bool().detach().clone()
    if inputs[0].shape[:2] != valid.shape or logits.shape[:2] != valid.shape:
        raise ValueError('Flat cache and [batch,length] umask shapes do not match')
    return FlatReadoutCache(inputs[0], skips[0], valid, logits.detach().clone())


@torch.no_grad()
def replay_predictions(model, cache: FlatReadoutCache) -> torch.Tensor:
    """Return [13,length,batch,outputs], preserving complete skip output/bias.

    Padded hidden is zero before the original task head; its bias is preserved
    in padded logits exactly as in the original model. Metrics must use valid.
    """
    backbone = _validate_model(model)
    local = cache.adapter_input[..., :backbone.latent_dim]
    context = cache.adapter_input[..., backbone.latent_dim:]
    predictions = []
    for alpha, beta, mu in SETTINGS:
        adapter_input = torch.cat((alpha * local, mu * context), dim=-1)
        hidden = backbone.emotion_norm(beta * cache.skip_output + backbone.emotion_adapter(adapter_input))
        # The transposed mask can make where return batch-major strides. Keep
        # original Flat's contiguous head input: otherwise CUDA Linear chooses
        # a different reduction path despite bit-identical hidden values.
        hidden = torch.where(cache.valid[..., None], hidden, torch.zeros_like(hidden)).contiguous()
        predictions.append(model.smax_fc(hidden))
    result = torch.stack(predictions)
    torch.testing.assert_close(result[IDENTITY_INDEX], cache.original_logits,
                               rtol=0, atol=0, msg='Identity replay differs from original Flat')
    return result
