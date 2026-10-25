"""Recent NoArea descriptors and centered corrections to original causal reads."""
import torch
from torch import nn

from experiments.osram_signature_noarea_iemocap68_20261025.block import NoAreaSignatureBlock


class _HistoryDescriptor(NoAreaSignatureBlock):
    """Reuse the exact descriptor without constructing an unused readout bridge."""
    def __init__(self, window=8):
        nn.Module.__init__(self)
        if window < 1:
            raise ValueError('window must be positive')
        self.window = window
        self.slots = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 8), nn.GELU()) for _ in range(3)])


class SerialNoAreaBlock(nn.Module):
    """Four role-specific forward-read deltas from a shared centered interaction.

    ``history`` is an optional projected-history counterfactual, making p=0 an
    explicit and testable identity even after the decoders have been updated.
    """
    def __init__(self, window=8):
        super().__init__()
        self.descriptor = _HistoryDescriptor(window)
        self.history_projection = nn.Sequential(nn.Linear(273, 64, bias=False), nn.GELU())
        self.local_projection = nn.Linear(256, 64)
        self.read_projection = nn.Linear(512, 64)
        self.roles = nn.Embedding(4, 8)
        self.interaction = nn.Sequential(
            nn.LayerNorm(203), nn.Linear(203, 128), nn.GELU(), nn.Linear(128, 64))
        self.decoders = nn.ModuleList([nn.Linear(64, 512, bias=False) for _ in range(4)])
        for decoder in self.decoders:
            nn.init.zeros_(decoder.weight)

    def history_descriptor(self, frame):
        descriptor = self.descriptor.descriptor(frame)
        return torch.cat((descriptor[..., :270], descriptor[..., -3:]), -1)

    def forward(self, frame, history=None):
        descriptor = self.history_descriptor(frame)
        projected = self.history_projection(descriptor) if history is None else history
        reads = torch.cat((frame.base[..., None, :512], frame.gap[..., :512]), -2)
        local = self.local_projection(frame.local).unsqueeze(-2).expand(*reads.shape[:-1], 64)
        read = self.read_projection(reads)
        p = projected.unsqueeze(-2).expand(*reads.shape[:-1], 64)
        a = frame.availability.to(read).unsqueeze(-2).expand(*reads.shape[:-1], 3)
        roles = self.roles.weight.expand(*reads.shape[:-2], 4, 8)
        context = (local, read)
        positive = self.interaction(torch.cat((*context, p, a, roles), -1))
        counterfactual = self.interaction(torch.cat((*context, torch.zeros_like(p), a, roles), -1))
        centered = positive - counterfactual
        delta = torch.stack([decoder(centered[..., role, :])
                             for role, decoder in enumerate(self.decoders)], -2)
        has_history = descriptor[..., -3:].sum(-1) > 0
        active = frame.valid & has_history
        roles_active = torch.cat((torch.ones_like(active[..., None]), ~frame.availability), -1)
        return delta * (active[..., None] & roles_active).unsqueeze(-1)

    def updated_reads(self, frame, delta=None):
        delta = self(frame) if delta is None else delta
        base = torch.cat((frame.base[..., :512] + delta[..., 0, :], frame.base[..., 512:]), -1)
        gap = torch.cat((frame.gap[..., :512] + delta[..., 1:, :], frame.gap[..., 512:]), -1)
        return base, gap
