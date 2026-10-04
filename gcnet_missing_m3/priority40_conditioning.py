"""M11--M15: conditioning at the explicitly assigned input/readout boundaries.

Independent mathematical adaptations; no image/time coordinates are invented.
Primary papers: FiLM arXiv:1709.07871 Eq. 1--2; Cross-stitch
arXiv:1604.03539 Sec. 3; MMTM arXiv:1911.08670 Sec. 3;
GCT arXiv:1909.11519 Eq. 2--5; Dynamic ReLU arXiv:2003.10027 Sec. 3.
Core code inspected: ethanjperez/film vr/models/filmed_net.py FiLM.forward;
haamoon/mmtm mmtm.py MMTM.forward; z-x-yang/GCT PyTorch/GCT.py;
microsoft/DynamicHead dyhead/dyrelu.py DYReLU.forward (later author variant,
not the original ECCV release; here the paper's sigmoid is used).
No author Cross-stitch code confirmed: independent paper-equation adaptation.
MMTM's author reimplementation uses sigmoid; 2*sigmoid here is an explicit
identity-centered adaptation, with zero final projections. Squeeze axes are
already absent from raw vectors; their concatenation is the joint descriptor.

The caller retains OSRAM reads/writes, the untouched Local skip, and task head.
"""
from __future__ import annotations

import torch
from torch import nn

from .readout_candidates_recalibration import GCTReadout, _masked_mean, _safe_tokens


def _zero_linear(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _input_evidence(local, evidence, active, latent_dim, forward_dim):
    if local.ndim != 2 or local.shape[1] != latent_dim:
        raise ValueError('local must have shape [N, latent_dim]')
    if evidence.shape != (local.shape[0], 4, forward_dim):
        raise ValueError('evidence must have shape [N, 4, forward_dim]')
    if active.shape != evidence.shape[:2] or active.dtype != torch.bool:
        raise ValueError('active must be boolean [N, 4]')
    if evidence.device != active.device or local.device != evidence.device:
        raise ValueError('inputs and active must share a device')
    return torch.where(active[..., None], evidence, torch.zeros_like(evidence))


class HistoryFiLM(nn.Module):
    """History-conditioned affine map of adapter Local only; no-history is no-op."""

    def __init__(self, latent_dim=256, forward_dim=512):
        super().__init__()
        self.latent_dim, self.forward_dim = latent_dim, forward_dim
        self.conditioner = nn.Sequential(
            nn.Linear(4 * forward_dim, 64), nn.ReLU(),
            _zero_linear(nn.Linear(64, 2 * latent_dim)))

    def forward(self, local, evidence, active, availability):
        history = _input_evidence(local, evidence, active, self.latent_dim, self.forward_dim)
        gamma, beta = self.conditioner(history.flatten(1)).chunk(2, dim=-1)
        adapted = (1 + gamma) * local + beta
        adapted = torch.where(active.any(1, keepdim=True), adapted, local)
        return adapted, history


class MMTMInput(nn.Module):
    """Joint 64-D squeeze, five separate channel excitations, identity startup."""

    def __init__(self, latent_dim=256, forward_dim=512):
        super().__init__()
        self.latent_dim, self.forward_dim = latent_dim, forward_dim
        self.squeeze = nn.Sequential(nn.Linear(latent_dim + 4 * forward_dim, 64), nn.ReLU())
        self.local_gate = _zero_linear(nn.Linear(64, latent_dim))
        self.evidence_gates = nn.ModuleList([
            _zero_linear(nn.Linear(64, forward_dim)) for _ in range(4)])

    def forward(self, local, evidence, active, availability):
        history = _input_evidence(local, evidence, active, self.latent_dim, self.forward_dim)
        joint = self.squeeze(torch.cat((local, history.flatten(1)), dim=-1))
        adapted_local = local * (2 * self.local_gate(joint).sigmoid())
        gates = torch.stack([2 * head(joint).sigmoid() for head in self.evidence_gates], dim=1)
        adapted_history = torch.where(active[..., None], history * gates, torch.zeros_like(history))
        return adapted_local, adapted_history


class CrossStitchReadout(nn.Module):
    """Five typed streams; per-channel 5x5 stitch between nonlinear stages."""

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.before = nn.ModuleList([nn.Linear(dim, dim) for _ in range(5)])
        self.after = nn.ModuleList([nn.Linear(dim, dim) for _ in range(5)])
        initial = torch.eye(5) * .875 + torch.ones(5, 5) * .025
        self.stitch = nn.Parameter(initial[None].repeat(dim, 1, 1))

    def forward(self, tokens, mask):
        x = _safe_tokens(tokens, mask, self.dim)
        x = torch.stack([layer(x[:, i]).relu() for i, layer in enumerate(self.before)], dim=1)
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        x = torch.einsum('cij,njc->nic', self.stitch, x)
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        x = torch.stack([layer(x[:, i]).relu() for i, layer in enumerate(self.after)], dim=1)
        return _masked_mean(x, mask)


class DynamicReLUReadout(nn.Module):
    """DY-ReLU-B: group-conditioned, channel-specific two-piece activation."""

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.hypernetwork = nn.Sequential(
            nn.Linear(dim, max(1, dim // 4)), nn.ReLU(),
            _zero_linear(nn.Linear(max(1, dim // 4), 4 * dim)))

    def forward(self, tokens, mask):
        x = _safe_tokens(tokens, mask, self.dim)
        coefficients = 2 * self.hypernetwork(_masked_mean(x, mask)).sigmoid() - 1
        a1, b1, a2, b2 = coefficients.chunk(4, dim=-1)
        first = (1 + a1[:, None]) * x + .5 * b1[:, None]
        second = a2[:, None] * x + .5 * b2[:, None]
        return _masked_mean(torch.maximum(first, second), mask)


READOUT_METHODS = {
    'm12_cross_stitch': CrossStitchReadout,
    'm14_gct': GCTReadout,
    'm15_dynamic_relu': DynamicReLUReadout,
}
INPUT_METHODS = {'m11_film': HistoryFiLM, 'm13_mmtm': MMTMInput}


def build(method, dim=64):
    if method not in READOUT_METHODS:
        raise ValueError(f'unknown conditioning readout: {method}')
    return READOUT_METHODS[method](dim)


def build_input(method, latent_dim=256, forward_dim=512):
    if method not in INPUT_METHODS:
        raise ValueError(f'unknown conditioning input adapter: {method}')
    return INPUT_METHODS[method](latent_dim, forward_dim)
