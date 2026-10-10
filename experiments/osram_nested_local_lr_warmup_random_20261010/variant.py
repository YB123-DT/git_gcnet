"""Isolated Nested initialization and optimizer policy; original modules unchanged."""
import math


def random_nested_adapter(latent_dim, num_heads, value_dim):
    from torch import nn
    from gcnet_missing_m3 import meaningful_input_new40 as module
    from gcnet_missing_m3.meaningful_new40_structure import NestedGNN
    # The sealed original has no zero_decoder argument. Preserve its constructor
    # and RNG draw order; only replace the decoder factory during construction.
    original = module.zero_linear
    try:
        module.zero_linear = nn.Linear
        return module.TokenAdapter(NestedGNN(), latent_dim, num_heads, value_dim, dim=64)
    finally:
        module.zero_linear = original


def separate_nested_group(groups, provenance, model, target_lr):
    if not math.isfinite(target_lr) or target_lr <= 0:
        raise ValueError('Nested LR must be finite and positive')
    nested = [p for n, p in model.named_parameters()
              if n.startswith('osram.meaningful_block.') and p.requires_grad]
    ids = {id(p) for p in nested}
    if not ids:
        raise ValueError('No trainable Nested parameters')
    original = [p for group in groups for p in group['params']]
    if len(original) != len(set(map(id, original))) or not ids <= set(map(id, original)):
        raise ValueError('Original optimizer partition is incomplete or duplicated')
    result = []
    for group in groups:
        remaining = [p for p in group['params'] if id(p) not in ids]
        if remaining:
            result.append(dict(group, params=remaining))
    result.append({'params': nested, 'lr': target_lr, 'name': 'nested'})
    count = sum(p.numel() for p in nested)
    records = {name: dict(value) for name, value in provenance.items()}
    records['backbone']['parameter_count'] -= count
    records['nested'] = dict(learning_rate=target_lr, parameter_count=count,
        multiplier=target_lr/records['backbone']['learning_rate'])
    return result, records


def apply_nested_lr(optimizer, epoch_index, target_lr, warmup_epochs):
    if epoch_index < 0 or warmup_epochs < 1 or not math.isfinite(target_lr) or target_lr <= 0:
        raise ValueError('Invalid Nested warmup configuration')
    groups = [g for g in optimizer.param_groups if g.get('name') == 'nested']
    if len(groups) != 1:
        raise ValueError('Expected exactly one independent Nested optimizer group')
    groups[0]['lr'] = target_lr * min(epoch_index+1, warmup_epochs)/warmup_epochs
