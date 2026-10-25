"""Small, isolated correction blocks for the four-block experiment."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class NeuMissBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([nn.Linear(256, 16) for _ in range(3)])
        self.context = nn.Sequential(nn.LayerNorm(2304), nn.Linear(2304, 32))
        self.mu = nn.Parameter(torch.zeros(80))
        self.steps = nn.ParameterList([nn.Parameter(torch.empty(80, 80)) for _ in range(3)])
        self.finish = nn.Parameter(torch.empty(80, 80))
        for weight in [*self.steps, self.finish]:
            nn.init.xavier_normal_(weight)
        self.out = nn.Linear(80, 1600)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, frame):
        available = frame.availability & frame.valid.unsqueeze(-1)
        slots = [layer(frame.latents[..., m, :]) * available[..., m, None]
                 for m, layer in enumerate(self.slots)]
        x = torch.cat([*slots, self.context(frame.context)], dim=-1)
        observed = torch.cat([available.repeat_interleave(16, dim=-1),
                              frame.valid.unsqueeze(-1).expand(*frame.valid.shape, 32)], -1).to(x)
        h = x - observed * self.mu
        for weight in self.steps:
            h = (h @ weight) * observed
        q_minus_x = (1 - observed) * (self.mu + h @ self.finish)
        active = frame.valid & available.any(-1) & ~available.all(-1)
        return self.out(q_minus_x) * active.unsqueeze(-1)


def degree_two_signature(path):
    """Signature of a piecewise linear path with a fixed zero basepoint."""
    width = path.shape[-1]
    first = path.new_zeros(width)
    second = path.new_zeros(width, width)
    previous = path.new_zeros(width)
    for point in path:
        dx = point - previous
        second = second + torch.outer(first, dx) + .5 * torch.outer(dx, dx)
        first = first + dx
        previous = point
    return torch.cat([first, second.flatten()])


class SignatureBlock(nn.Module):
    def __init__(self, window=8):
        super().__init__()
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self.slots = nn.ModuleList([nn.Sequential(nn.Linear(256, 8), nn.GELU()) for _ in range(3)])
        self.out = nn.Sequential(nn.LayerNorm(532), nn.Linear(532, 128), nn.GELU(), nn.Linear(128, 1600))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def descriptor(self, frame):
        projected = torch.stack([layer(frame.latents[..., m, :]) for m, layer in enumerate(self.slots)], -2)
        t, b = frame.valid.shape
        if t == 0:
            return frame.local.new_empty((0, b, 532))
        valid = frame.valid.T
        positions = torch.arange(t, device=valid.device).expand(b, t)
        ordered = positions.masked_fill(~valid, t).sort(-1).values
        prior_count = valid.long().cumsum(-1) - valid.long()
        ranks = prior_count[..., None] + torch.arange(-self.window, 0, device=valid.device)
        indices = ordered.gather(1, ranks.clamp(0, t - 1).reshape(b, -1)).reshape(b, t, self.window)
        indices = indices.clamp_max(t - 1)
        batches = torch.arange(b, device=valid.device)[:, None, None]
        points = projected.transpose(0, 1)[batches, indices]
        observed = frame.availability.transpose(0, 1)[batches, indices] & (ranks >= 0)[..., None]
        # Fixed window coordinates retain missing utterance positions.
        times = torch.arange(1, self.window + 1, device=valid.device, dtype=points.dtype) / self.window
        times = times.view(1, 1, self.window, 1, 1).expand(b, t, self.window, 3, 1)
        points = torch.cat([points, times], -1).transpose(2, 3)  # B,T,modality,window,9
        observed = observed.transpose(2, 3)
        slots = torch.arange(1, self.window + 1, device=valid.device)
        latest = torch.where(observed, slots, 0).cummax(-1).values
        previous = F.pad(latest[..., :-1], (1, 0))
        padded = F.pad(points, (0, 0, 1, 0))
        previous_points = padded.gather(-2, previous[..., None].expand(*previous.shape, 9))
        dx = (points - previous_points) * observed[..., None]
        prefix = dx.cumsum(-2) - dx
        second = ((prefix[..., :, :, None] * dx[..., :, None, :])
                  + .5 * dx[..., :, :, None] * dx[..., :, None, :]).sum(-3)
        signatures = torch.cat([dx.sum(-2), second.flatten(-2)], -1).flatten(-2)
        return torch.cat([signatures.transpose(0, 1), frame.local,
                          frame.availability.to(frame.local.dtype), observed.sum(-1).transpose(0, 1).to(frame.local.dtype)], -1)

    def forward(self, frame):
        has_history = (frame.valid.long().cumsum(0) - frame.valid.long()) > 0
        return self.out(self.descriptor(frame)) * (frame.valid & has_history).unsqueeze(-1)


@dataclass
class SupportBank:
    hidden: torch.Tensor
    labels: torch.Tensor
    base_prediction: torch.Tensor
    conversation_ids: list[str]
    rate_ids: torch.Tensor
    pattern_ids: torch.Tensor
    parent_id: str
    split: str = "train"


class RidgeResidualBlock(nn.Module):
    def __init__(self, neighbors=32):
        super().__init__()
        self.embedding = nn.Sequential(nn.LayerNorm(1600), nn.Linear(1600, 64))
        self.raw_ridge = nn.Parameter(torch.tensor(0.0))
        self.neighbors = neighbors

    def forward(self, query_hidden, bank, query_conversation_ids, rate_index, pattern_ids, parent_id):
        if bank.split != "train":
            raise ValueError("support bank must contain training data only")
        if bank.parent_id != parent_id:
            raise ValueError("support bank parent hash does not match query parent")
        if len(query_conversation_ids) != len(query_hidden):
            raise ValueError("one conversation ID is required per query")
        queries = self.embedding(query_hidden)
        if len(query_hidden) == 0:
            return query_hidden.new_zeros((0, 1))
        selected = torch.where(bank.rate_ids == rate_index)[0]
        if len(selected) == 0:
            return queries.sum(-1, keepdim=True) * 0
        support = self.embedding(bank.hidden[selected])
        conversation_codes = {name: code for code, name in enumerate(set(bank.conversation_ids))}
        support_codes = torch.tensor([conversation_codes[name] for name in bank.conversation_ids],
                                     device=bank.hidden.device, dtype=torch.long)[selected]
        query_codes = torch.tensor([conversation_codes.get(name, -1) for name in query_conversation_ids],
                                   device=bank.hidden.device, dtype=torch.long)
        legal = (query_codes[:, None] != support_codes[None]) & (pattern_ids[:, None] == bank.pattern_ids[selected][None])
        distances = (queries.square().sum(-1, keepdim=True) + support.square().sum(-1)[None]
                     - 2 * queries @ support.T).masked_fill(~legal, float("inf"))
        k = min(self.neighbors, len(selected))
        values, indices = distances.topk(k, dim=-1, largest=False)
        active = torch.isfinite(values)
        z = F.pad(support[indices].double(), (0, 1), value=1) * active[..., None]
        q = F.pad(queries.double(), (0, 1), value=1)
        residual = (bank.labels[selected][indices] - bank.base_prediction[selected][indices]).double() * active[..., None]
        ridge = F.softplus(self.raw_ridge.double()) + 1e-4
        u = torch.linalg.solve(z @ z.transpose(-1, -2) + ridge * torch.eye(k, device=z.device, dtype=z.dtype), residual)
        return (q[:, None] @ z.transpose(-1, -2) @ u).reshape(-1, 1).to(query_hidden.dtype)
