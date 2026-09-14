"""Training-only next-utterance prediction from post-write causal memory."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class FutureStateJEPA(nn.Module):
    def __init__(self, encoder, local_path, hidden_dim, latent_dim,
                 num_heads, key_dim, value_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = num_heads * value_dim
        self.query_parameters = num_heads * key_dim
        # Exact online parameter budget of CompleteViewLocalStateJEPA.
        self.parameter_budget = (2 * hidden_dim + 8 * 16
                                 + (hidden_dim + 17) * latent_dim
                                 + (latent_dim + 1) * latent_dim)
        constant = self.query_parameters + 2 * self.input_dim + latent_dim
        ideal = (self.parameter_budget - constant) / (self.input_dim + latent_dim + 1)
        candidates = {max(1, math.floor(ideal)), max(1, math.ceil(ideal))}
        self.width = min(candidates, key=lambda w: (
            abs(self.parameter_count(w) - self.parameter_budget), w))
        self.state_query = nn.Parameter(torch.randn(num_heads, key_dim))
        self.predictor = nn.Sequential(
            nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, self.width),
            nn.GELU(), nn.Linear(self.width, latent_dim))
        self.teacher_encoder = copy.deepcopy(encoder).requires_grad_(False).eval()
        self.teacher_local = copy.deepcopy(local_path).requires_grad_(False).eval()
        self.register_buffer('ema_updates', torch.zeros((), dtype=torch.long))
        self.records = {}
        self._template = None
        self._valid = None

    def parameter_count(self, width):
        return (self.query_parameters + 2 * self.input_dim + self.latent_dim
                + width * (self.input_dim + self.latent_dim + 1))

    def train(self, mode=True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_local.eval()
        if not mode:
            self.clear()
        return self

    def clear(self):
        self.records = {}
        self._template = self._valid = None

    def begin(self, node, valid):
        self.clear()
        self._template = node.new_zeros(*node.shape[:2], self.input_dim)
        self._valid = valid.detach().clone()
        return self.observe

    def observe(self, time_index, memory, active):
        """Read non-detached post-write state; never update it."""
        query = F.normalize(self.state_query, dim=-1).to(memory.dtype)
        compact = (memory @ query[None, ..., None]).squeeze(-1).flatten(1)
        self.records[time_index] = compact.masked_fill(~active[:, None], 0)

    def compact_states(self):
        if self._template is None:
            raise ValueError('future-state requires a preceding training forward')
        return torch.stack([self.records.get(t, self._template[t])
                            for t in range(self._template.shape[0])])

    @torch.no_grad()
    def target(self, features, umask):
        valid = umask.T.bool()
        availability = valid[..., None].expand(-1, -1, 3).to(features.dtype)
        node, _ = self.teacher_encoder(features, availability, umask)
        return (node + self.teacher_local(node)).masked_fill(~valid[..., None], 0)

    def loss(self, features, umask):
        if self._valid is None or not torch.equal(self._valid, umask.T.bool()):
            raise ValueError('future-state loss requires matching training forward/mask')
        selected = self._valid[:-1] & self._valid[1:]
        count = int(selected.sum().item())
        if not count:
            return self.state_query.sum() * 0, 0
        states = self.compact_states()[:-1][selected]
        prediction = self.predictor(states)
        target = self.target(features, umask)[1:][selected]
        return F.smooth_l1_loss(
            F.layer_norm(prediction, (self.latent_dim,)),
            F.layer_norm(target, (self.latent_dim,))), count

    @torch.no_grad()
    def update(self, encoder, local_path, tau):
        if not 0 <= tau <= 1:
            raise ValueError('EMA tau must be in [0,1]')
        for teacher, online in ((self.teacher_encoder, encoder),
                                (self.teacher_local, local_path)):
            source = dict(online.named_parameters())
            for name, parameter in teacher.named_parameters():
                parameter.mul_(tau).add_(source[name], alpha=1-tau)
            source_buffers = dict(online.named_buffers())
            for name, buffer in teacher.named_buffers():
                buffer.copy_(source_buffers[name])
        self.ema_updates.add_(1)
