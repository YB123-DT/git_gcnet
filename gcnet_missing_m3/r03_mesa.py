"""Mesa cumulative ridge memory adapted to OSRAM's historical read contract.

Paper: arXiv:2506.05233v2, equations 7–8. G/H retain weighted historical
associations. This short-conversation adaptation uses an exact positive-ridge
solve instead of the paper's CG/chunk kernels, static trainable OSRAM-style
gates, jointly observed modality slots, and reads before the current write.
The configured write_step scales input weights (default protocol: 0.6).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .core20_storage import SequenceStorage


class MesaStorage(SequenceStorage):
    def __init__(self, config, osram):
        super().__init__(osram.num_heads, osram.key_dim, osram.value_dim, osram.latent_dim)
        self.write_step = float(getattr(config, 'osram_write_step', osram.write_step))
        self.ridge_floor = float(getattr(config, 'r03_mesa_ridge_floor', 0.25))
        ridge_initial = float(getattr(config, 'r03_mesa_ridge', 1.0))
        if not math.isfinite(self.write_step) or not 0 <= self.write_step <= 1:
            raise ValueError('Mesa input write_step must be finite and within [0, 1]')
        if (not math.isfinite(self.ridge_floor) or self.ridge_floor <= 0
                or not math.isfinite(ridge_initial) or ridge_initial <= self.ridge_floor):
            raise ValueError('Mesa requires positive ridge_floor and ridge_initial > floor')
        # Own parameters: integration freezes the unused original gates.
        self.alpha_logits = nn.Parameter(osram.alpha_logits.detach().clone())
        self.beta_logits = nn.Parameter(osram.beta_logits.detach().clone())
        inverse_softplus = math.log(math.expm1(ridge_initial - self.ridge_floor))
        self.ridge_logits = nn.Parameter(osram.alpha_logits.new_full(
            (self.num_heads, self.key_dim), inverse_softplus))

    @property
    def ridge(self):
        return F.softplus(self.ridge_logits) + self.ridge_floor

    def initial_state(self, reference, batch):
        # Keep sufficient statistics in FP32 under low-precision training;
        # preserve FP64 for numerical oracle tests.
        dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
        return {'G': torch.zeros(batch, self.num_heads, self.value_dim, self.key_dim,
                                 device=reference.device, dtype=dtype),
                'H': torch.zeros(batch, self.num_heads, self.key_dim, self.key_dim,
                                 device=reference.device, dtype=dtype)}

    def read(self, state, queries, active):
        # Exactly one forgetting event per valid utterance, including one
        # without observations. Padding changes neither sufficient statistic.
        with torch.autocast(device_type=queries.device.type, enabled=False):
            gamma = self.alpha_logits.sigmoid().to(state['G'])[None, :, None, None]
            enabled = active[:, None, None, None]
            decayed = {name: torch.where(enabled, tensor * gamma, tensor)
                       for name, tensor in state.items()}
            diagonal = torch.diag_embed(self.ridge.to(state['H']))[None]
            q = torch.where(active[:, None, None, None], queries, 0.).to(state['H'])
            # Solve all four query types as simultaneous right-hand sides.
            solved = torch.linalg.solve(decayed['H'] + diagonal, q.permute(0, 2, 3, 1))
            output = (decayed['G'] @ solved).permute(0, 3, 1, 2)
        return output.to(queries.dtype), decayed

    def write(self, state, keys, values, observed, active, queries):
        del queries  # Query/label information does not enter history statistics.
        with torch.autocast(device_type=keys.device.type, enabled=False):
            observed = observed.bool() & active[:, None]
            mask = observed[:, None, None, :]
            k = torch.where(mask, keys, 0.).to(state['H'])
            v = torch.where(mask, values, 0.).to(state['G'])
            weights = (self.write_step * self.beta_logits.sigmoid().to(k))[None, :, None, :]
            kt = k.transpose(-1, -2)
            candidate = {'G': state['G'] + (v * weights) @ kt,
                         'H': state['H'] + (k * weights) @ kt}
            enabled = active[:, None, None, None]
            return {name: torch.where(enabled, tensor, state[name])
                    for name, tensor in candidate.items()}


R03MesaMemory = MesaStorage
