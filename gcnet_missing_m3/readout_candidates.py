"""Fixed external-mechanism screen; all variants supplement the intact Flat.

These are masked, per-utterance adaptations, not full paper reproductions.
The common128-dimensional interface and zero-initialized output are experiment
controls, not claims about the original papers. No Memory operation occurs here.
"""
from __future__ import annotations

import torch
from torch import nn


CANDIDATE_METHODS = (
    'film', 'se', 'eca', 'cbam', 'gct', 'simam', 'sk',
    'mlb', 'mfb', 'mutan', 'block', 'mcb', 'ban',
    'dcnv2', 'cin', 'autoint', 'din', 'dlrm', 'aff', 'nonlocal',
)


def build_operator(method: str, dim: int = 128) -> nn.Module:
    if method in CANDIDATE_METHODS[:7]:
        from .readout_candidates_recalibration import build_recalibration
        return build_recalibration(method, dim)
    if method in CANDIDATE_METHODS[7:13]:
        from .readout_candidates_bilinear import build_bilinear
        return build_bilinear(method, dim)
    if method in CANDIDATE_METHODS[13:]:
        from .readout_candidates_interactions import build_interaction
        return build_interaction(method, dim)
    raise ValueError(f'Unknown external readout candidate: {method}')


class ExternalReadoutResidual(nn.Module):
    """Sanitize forward evidence, apply one method, add a zero-start residual.

    The caller retains all original Flat parameters. The module is initialized
    inside the caller's fork_rng, and no new stochastic operation runs forward.
    Type embeddings start at zero and apply only to active tokens. Entire rows
    without preceding valid history are excluded before running the operator.
    """

    def __init__(self, latent_dim, context_dim, output_dim, method):
        super().__init__()
        if method not in CANDIDATE_METHODS:
            raise ValueError(f'Unknown external readout candidate: {method}')
        if min(latent_dim, context_dim, output_dim) <= 0 or context_dim % 2:
            raise ValueError('Positive dimensions and paired forward/backward context required')
        self.method = method
        self.forward_dim = context_dim // 2
        self.output_dim = output_dim
        self.local_project = nn.Linear(latent_dim, 128, bias=False)
        self.memory_project = nn.Linear(self.forward_dim, 128, bias=False)
        self.type_embedding = nn.Parameter(torch.zeros(5, 128))
        self.operator = build_operator(method, 128)
        self.output = nn.Linear(128, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_diagnostics = {}

    def forward(self, local, base, gap, availability, umask, flat_anchor):
        valid = umask.T.bool()
        prefix = tuple(valid.shape)
        if (tuple(local.shape[:2]) != prefix
                or tuple(base.shape) != (*prefix, 2 * self.forward_dim)
                or tuple(gap.shape) != (*prefix, 3, 2 * self.forward_dim)
                or tuple(availability.shape) != (*prefix, 3)
                or tuple(flat_anchor.shape) != (*prefix, self.output_dim)):
            raise ValueError('External readout input dimensions do not match')
        av = availability[valid]
        if not torch.all((av == 0) | (av == 1)) or not torch.all(av.bool().any(-1)):
            raise ValueError('Every valid utterance needs a nonempty binary availability')
        has_history = valid & ((valid.long().cumsum(0) - valid.long()) > 0)
        active_memory = torch.cat((valid[..., None], valid[..., None] & ~availability.bool()), -1)
        active_memory &= has_history[..., None]
        safe_local = torch.where(has_history[..., None], local, torch.zeros_like(local))
        evidence = torch.cat((base[..., None, :self.forward_dim], gap[..., :self.forward_dim]), 2)
        evidence = torch.where(active_memory[..., None], evidence, torch.zeros_like(evidence))
        residual = flat_anchor.new_zeros(flat_anchor.shape)
        if has_history.any():
            q = self.local_project(safe_local[has_history])
            k = self.memory_project(evidence[has_history])
            history_active = active_memory[has_history]
            active = torch.cat((torch.ones_like(history_active[:, :1]), history_active), -1)
            tokens = torch.cat((q[:, None], k), 1) + self.type_embedding
            tokens = torch.where(active[..., None], tokens, torch.zeros_like(tokens))
            representation = self.operator(tokens, active)
            if representation.shape != (q.shape[0], 128):
                raise RuntimeError('Candidate must return exactly128 features per utterance')
            residual[has_history] = self.output(representation)
        with torch.no_grad():
            n = int(valid.sum())
            norm = residual[valid].norm(dim=-1)
            self.last_diagnostics = {
                'method': self.method,
                'valid_count': n,
                'history_count': int(has_history.sum()),
                'residual_norm': float(norm.mean()) if n else 0.,
                'residual_anchor_norm_ratio': float((norm / flat_anchor[valid].norm(dim=-1).clamp_min(1e-8)).mean()) if n else 0.,
                'active_memory_count_mean': float(active_memory.sum(-1)[valid].float().mean()) if n else 0.,
            }
        return residual
