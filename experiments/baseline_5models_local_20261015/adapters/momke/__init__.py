"""MoMKE Original / Causal with shared local feature and missing-mask adapters.

Parameter names/shapes match upstream 2dceb57. Only attention visibility differs
between versions; no experts, router, fusion or classifier are added or removed.
"""
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .attention import Attention
from .model import MoMKE

UPSTREAM_COMMIT = '2dceb57f9e6f4bc84c2fc8f4eb7c761ea64c9efc'


class AdaptedMoMKE(MoMKE):
    """Batch-first interface. ``padding=True`` denotes a valid utterance."""

    def forward(self, a, t, v, availability, padding, first_stage=False):
        if a.ndim != 3 or availability.shape != (*a.shape[:2], 3):
            raise ValueError('Expected BxLxD features and BxLx3 availability')
        if padding.shape != a.shape[:2]:
            raise ValueError('Expected BxL validity mask')
        available = availability.bool() & padding.bool().unsqueeze(-1)
        # Missing raw features must never enter projection, routing, or residuals.
        # masked_fill also removes arbitrary values in padded storage.
        features = torch.cat([x.masked_fill(~available[:, :, i:i+1], 0)
                              for i, x in enumerate((a, t, v))], dim=-1)
        hidden, logits, out_a, out_t, out_v, weights = super().forward(
            features.transpose(0, 1), available.transpose(0, 1),
            padding.bool(), first_stage)
        return {'logits': logits, 'hidden': hidden,
                'auxiliary_logits': (out_a, out_t, out_v), 'routing_weights': weights}


def build_model(config, dims, nclasses, nspeakers=None):
    """Build version from config['causal']; dims is (A,T,V) or {'a','t','v'}."""
    if isinstance(config, dict):
        cfg = dict(config)
    else:
        cfg = vars(config).copy()
    if isinstance(dims, dict):
        dims = tuple(dims[k] for k in ('a', 't', 'v'))
    hidden = cfg.get('hidden', 256)
    heads = cfg.get('num_heads', cfg.get('heads', 2))
    if hidden % heads:
        raise ValueError('MoMKE hidden must be divisible by attention heads')
    dropout = cfg.get('drop_rate', cfg.get('dropout', 0.5))
    args = SimpleNamespace(device=cfg.get('device', 'cpu'), drop_rate=dropout)
    model = AdaptedMoMKE(args, *dims, hidden, nclasses,
                        depth=cfg.get('depth', 4), num_heads=heads, mlp_ratio=1,
                        drop_rate=dropout,
                        attn_drop_rate=cfg.get('attn_drop_rate', 0.), no_cuda=True)
    model.causal = bool(cfg.get('causal', True))
    for layer in model.modules():
        if isinstance(layer, Attention):
            layer.causal = model.causal
    return model


def stage_loss(output, labels, padding, first_stage):
    """Official IEMOCAP loss: sum of A/T/V CE in Stage 1, fusion CE in Stage 2.

    Stage 1 receives complete training modalities, just as upstream. The runner
    controls missing augmentation, fixed test masks, and immutable checkpoints.
    """
    valid = padding.bool()
    if not valid.any():
        raise ValueError('Cannot compute MoMKE loss on an empty batch')
    if first_stage:
        return sum(F.cross_entropy(x[valid], labels[valid]) for x in output['auxiliary_logits'])
    return F.cross_entropy(output['logits'][valid], labels[valid])
