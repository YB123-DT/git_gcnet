"""Whole-role OSRAM adapters for PO and exact RepSet, with task loss only.

PO: Zhang et al., ICLR 2019, arxiv.org/abs/1812.03928, Eq.7.
Author code: github.com/Cyanogenoid/perm-optim/permutation.py and
ban-vqa/permnet.py. The inner alternate gradient and LSTM cell readout remain.
RepSet: Skianis et al., AISTATS 2020, proceedings.mlr.press/v108/skianis20a.
Author code: github.com/giannisnik/repset/repset/models.py. Exact matching
objective values, not transport-pooled features, form the representation.
"""
import itertools

import torch
from torch import nn
from torch.nn import functional as F


class _RoleAdapter(nn.Module):
    """Five physical slots; no splitting into heads or persistent sample state."""

    def __init__(self, latent_dim, num_heads, value_dim, summary_dim):
        super().__init__()
        if min(latent_dim, num_heads, value_dim) < 1:
            raise ValueError("dimensions must be positive")
        self.width = num_heads * value_dim
        self.local_in = nn.Linear(latent_dim, self.width)
        self.role = nn.Parameter(torch.randn(5, self.width) * 0.02)
        self.local_out = nn.Linear(self.width + summary_dim, latent_dim)
        self.evidence_out = nn.ModuleList(
            nn.Linear(self.width + summary_dim, self.width) for _ in range(4)
        )
        for head in [self.local_out, *self.evidence_out]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, local, evidence, active, availability):
        if evidence.shape != (local.shape[0], 4, self.width) or active.shape != evidence.shape[:2]:
            raise ValueError("expected local plus four whole evidence roles")
        active = active.bool()
        clean = torch.where(active[..., None], evidence, torch.zeros_like(evidence))
        mask = torch.cat((torch.ones_like(active[:, :1]), active), dim=1)
        x = torch.cat((self.local_in(local)[:, None], clean), dim=1) + self.role
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        # Group only current batch rows by presence; absent roles never enter
        # assignment matrices, the sequence, or empirical matching cardinality.
        summary = x.new_zeros(x.shape[0], self.local_out.in_features - self.width)
        for pattern in mask.unique(dim=0):
            rows = (mask == pattern).all(dim=1).nonzero(as_tuple=True)[0]
            packed = x[rows][:, pattern]
            summary = summary.index_copy(0, rows, self.summarize(packed))
        decoded = torch.cat((x, summary[:, None].expand(-1, 5, -1)), dim=-1)
        delta = torch.stack([head(decoded[:, i + 1]) for i, head in enumerate(self.evidence_out)], dim=1)
        changed = torch.where(active[..., None], clean + delta, torch.zeros_like(clean))
        return local + self.local_out(decoded[:, 0]), changed


class PermutationOptimization(_RoleAdapter):
    """Antisymmetric ordering cost -> alternate inner updates -> LSTM cell."""

    def __init__(self, latent_dim, num_heads, value_dim):
        hidden = min(128, num_heads * value_dim)
        super().__init__(latent_dim, num_heads, value_dim, hidden)
        self.comparator = nn.Sequential(nn.Linear(2 * self.width, 64), nn.ReLU(), nn.Linear(64, 1))
        self.step_size = nn.Parameter(torch.ones(()))
        self.steps = 3
        self.lstm = nn.LSTM(self.width, hidden, batch_first=True)

    @staticmethod
    def sinkhorn(logits):
        assignment = logits.softmax(dim=-1)
        for _ in range(4):
            assignment = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-12)
            assignment = assignment / assignment.sum(-2, keepdim=True).clamp_min(1e-12)
        return assignment

    def ordering(self, x):
        count = x.shape[1]
        left = x[:, :, None].expand(-1, -1, count, -1)
        right = x[:, None, :].expand(-1, count, -1, -1)
        costs = self.comparator(torch.cat((left, right), dim=-1)).squeeze(-1)
        costs = costs - costs.transpose(-1, -2)
        costs = costs / costs.flatten(1).norm(dim=-1)[:, None, None].clamp_min(1e-10)
        logits = torch.zeros_like(costs)
        for _ in range(self.steps):
            assignment = self.sinkhorn(logits)
            before = assignment.cumsum(-1) - assignment
            after = assignment.flip([-1]).cumsum(-1).flip([-1]) - assignment
            # Source Eq.7: update logits with d cost / d P, omitting the
            # Sinkhorn Jacobian in this inner step. No autograd.grad needed;
            # ordinary outer backprop and inference_mode both remain valid.
            gradient = 2 * costs @ (after - before)
            logits = logits - self.step_size.abs() * gradient
        return self.sinkhorn(logits)

    def summarize(self, x):
        ordered = self.ordering(x).transpose(-1, -2) @ x
        _, (_, cell) = self.lstm(ordered)
        return cell[-1]


class ExactRepSet(_RoleAdapter):
    """One score per learned hidden set with exact two-sided capacity constraints."""

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim, summary_dim=16)
        self.templates = nn.Parameter(torch.randn(16, 5, self.width) / self.width**0.5)
        for count in range(1, 6):
            # At most 5! injections, pure combinatorial constants, not caches.
            assignments = torch.tensor(list(itertools.permutations(range(5), count)), dtype=torch.long)
            self.register_buffer(f"assignments_{count}", assignments, persistent=False)

    def matching_scores(self, x):
        count = x.shape[1]
        if count < 1 or count > 5:
            raise ValueError("RepSet expects one to five packed roles")
        affinities = F.relu(torch.einsum("bnd,ksd->bkns", x, self.templates))
        assignments = getattr(self, f"assignments_{count}")
        rows = torch.arange(count, device=x.device)[None, :]
        scores = affinities[:, :, rows, assignments].sum(-1)
        # Nonnegative affinities + >=N supports mean a maximal injection
        # attains the source partial matching optimum, including zero edges.
        # torch.max chooses one valid subgradient at assignment ties.
        return scores.max(-1).values / count

    def summarize(self, x):
        return self.matching_scores(x)


METHODS = {
    "geometry_po_canonical_sequence": PermutationOptimization,
    "geometry_repset_exact_template_matching": ExactRepSet,
}


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    if method not in METHODS:
        raise ValueError(f"Unknown geometry method: {method}")
    return METHODS[method](latent_dim, num_heads, value_dim)
