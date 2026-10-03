"""Seven paper-inspired readouts over one utterance's five typed evidence slots.

Independent mathematical adaptations, not copied visual-model implementations.
Sources, fixed settings, licenses and deviations are recorded in
experiments/osram_readout20_20261003/literature/recalibration.json.
Only the parent wrapper owns memory, evidence projection and output residuals.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _safe_tokens(tokens, active, dim):
    if tokens.ndim != 3 or tokens.shape[1:] != (5, dim):
        raise ValueError('tokens must have shape [N, 5, dim]')
    if active.shape != tokens.shape[:2] or active.dtype != torch.bool:
        raise ValueError('active must be boolean with shape [N, 5]')
    if active.device != tokens.device or not tokens.is_floating_point():
        raise ValueError('tokens must be floating point on the active mask device')
    return torch.where(active[..., None], tokens, torch.zeros_like(tokens))


def _masked_mean(tokens, active):
    safe = torch.where(active[..., None], tokens, torch.zeros_like(tokens))
    count = active.sum(1, keepdim=True).clamp_min(1).to(tokens.dtype)
    return safe.sum(1) / count


def _masked_max(tokens, active):
    maximum = tokens.masked_fill(~active[..., None], -torch.inf).amax(1)
    return torch.where(active.any(1, keepdim=True), maximum, torch.zeros_like(maximum))


def _excitation(dim):
    return nn.Sequential(nn.Linear(dim, max(1, dim // 16)), nn.ReLU(),
                         nn.Linear(max(1, dim // 16), dim))


class FiLMReadout(nn.Module):
    """FiLM Eq. (2): Local-conditioned affine modulation of memory tokens."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conditioner = nn.Sequential(nn.Linear(dim, max(1, dim // 16)),
                                         nn.ReLU(), nn.Linear(max(1, dim // 16), 2 * dim))

    def forward(self, tokens, active):
        x = _safe_tokens(tokens, active, self.dim)
        condition = x[:, 0]
        for layer in self.conditioner:
            condition = layer(condition)
            condition = torch.where(active[:, :1], condition, torch.zeros_like(condition))
        delta_gamma, beta = condition.chunk(2, dim=-1)
        memory = (1 + delta_gamma[:, None]) * x[:, 1:] + beta[:, None]
        memory = torch.where(active[:, 1:, None], memory, torch.zeros_like(memory))
        return _masked_mean(torch.cat((x[:, :1], memory), dim=1), active)


class SEReadout(nn.Module):
    """SE Eq. (3)-(4), with active evidence slots as the squeeze axis."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.excitation = _excitation(dim)

    def forward(self, tokens, active):
        x = _safe_tokens(tokens, active, self.dim)
        gate = self.excitation(_masked_mean(x, active)).sigmoid()
        return _masked_mean(x * gate[:, None], active)


class ECAReadout(nn.Module):
    """ECA Eq. (9): the 1-D filter spans feature coordinates, never time."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        size = int(abs((math.log2(dim) + 1) / 2))
        self.kernel_size = size if size % 2 else size + 1
        self.channel_filter = nn.Conv1d(1, 1, self.kernel_size,
                                        padding=self.kernel_size // 2, bias=False)

    def forward(self, tokens, active):
        x = _safe_tokens(tokens, active, self.dim)
        descriptor = _masked_mean(x, active)
        gate = self.channel_filter(descriptor[:, None]).squeeze(1).sigmoid()
        return _masked_mean(x * gate[:, None], active)


class CBAMReadout(nn.Module):
    """CBAM channel then spatial gates, adapting spatial to five typed slots.

    Slot-kernel adjacency is an explicit adaptation. No BatchNorm is retained.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.excitation = _excitation(dim)
        self.slot_filter = nn.Conv1d(2, 1, 7, padding=3, bias=False)

    def forward(self, tokens, active):
        x = _safe_tokens(tokens, active, self.dim)
        channel_gate = (self.excitation(_masked_mean(x, active))
                        + self.excitation(_masked_max(x, active))).sigmoid()
        x = x * channel_gate[:, None]
        descriptor = torch.stack((x.mean(-1), x.amax(-1)), dim=1)
        slot_gate = self.slot_filter(descriptor).squeeze(1).sigmoid()
        return _masked_mean(x * slot_gate[..., None], active)


class GCTReadout(nn.Module):
    """GCT Eq. (2)-(5): L2 embedding and normalized, identity-centered gate."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(dim))
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.epsilon = 1e-5

    def forward(self, tokens, active):
        x = _safe_tokens(tokens, active, self.dim)
        embedding = (x.square().sum(1) + self.epsilon).sqrt() * self.alpha
        scale = (embedding.square().mean(-1, keepdim=True) + self.epsilon).sqrt()
        gate = 1 + torch.tanh(self.gamma * embedding / scale + self.beta)
        return _masked_mean(x * gate[:, None], active)


class SimAMReadout(nn.Module):
    """SimAM Figure 3 over active slots; singleton rows use identity."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.e_lambda = 1e-4

    def forward(self, tokens, active):
        x = _safe_tokens(tokens, active, self.dim)
        centered = x - _masked_mean(x, active)[:, None]
        squared = torch.where(active[..., None], centered.square(), torch.zeros_like(centered))
        count = active.sum(1, keepdim=True)
        variance = squared.sum(1) / (count - 1).clamp_min(1).to(x.dtype)
        gate = (squared / (4 * (variance[:, None] + self.e_lambda)) + .5).sigmoid()
        gate = torch.where((count > 1)[..., None], gate, torch.ones_like(gate))
        return _masked_mean(x * gate, active)


class DenseSKReadout(nn.Module):
    """SK-inspired channel-wise selection of two dense current-token branches.

    Dense transformations replace spatial kernels; this is not SK convolution.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        hidden = max(dim // 16, 32)
        self.branches = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.squeeze = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.selectors = nn.ModuleList([nn.Linear(hidden, dim) for _ in range(2)])

    def forward(self, tokens, active):
        x = _safe_tokens(tokens, active, self.dim)
        branches = []
        for branch in self.branches:
            value = branch(x)
            value = torch.where(active[..., None], value, torch.zeros_like(value))
            branches.append(value.relu())
        summary = self.squeeze(_masked_mean(branches[0] + branches[1], active))
        weights = torch.stack([selector(summary) for selector in self.selectors], dim=1).softmax(1)
        mixed = sum(value * weights[:, index, None] for index, value in enumerate(branches))
        return _masked_mean(mixed, active)


def build_recalibration(method, dim=128):
    """Build one fixed-setting readout returning [N, D] from [N, 5, D]."""
    if type(dim) is not int or dim <= 0:
        raise ValueError('dim must be a positive integer')
    classes = {'film': FiLMReadout, 'se': SEReadout, 'eca': ECAReadout,
               'cbam': CBAMReadout, 'gct': GCTReadout, 'simam': SimAMReadout,
               'sk': DenseSKReadout}
    if method not in classes:
        raise ValueError('unknown recalibration method: ' + str(method))
    return classes[method](dim)
