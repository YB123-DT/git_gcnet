"""Frozen-candidate Text-source routing; this module never changes CHM state.

Candidate order is full, omit Gap-T<-T (Base retained), omit Base<-T
(Gap retained). Uniform initialization is intentionally not baseline identity.
"""

import torch
from torch import nn
from torch.nn import functional as F


class TextPathRouter(nn.Module):
    """Mix three probabilities; frozen inputs remain the backward-compatible default."""

    def __init__(self, local_dim=256, context_dim=512, projection_dim=64, hidden_dim=64,
                 detach_inputs=True):
        super().__init__()
        if min(local_dim, context_dim, projection_dim, hidden_dim) < 1:
            raise ValueError('Router dimensions must be positive')
        self.local_dim = local_dim
        self.context_dim = context_dim
        self.detach_inputs = bool(detach_inputs)
        # Creating this optional module must not change the baseline RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.local_projection = nn.Linear(local_dim, projection_dim)
            self.base_projection = nn.Linear(context_dim, projection_dim)
            self.gap_projection = nn.Linear(context_dim, projection_dim)
            self.network = nn.Sequential(
                nn.LayerNorm(3 * projection_dim + 6),
                nn.Linear(3 * projection_dim + 6, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, local, base_text, gap_text, text_trajectory, candidates, eligible):
        """Return log probabilities and [full, Base-retained, Gap-retained] weights.

        ``text_trajectory`` holds strictly-past count, coverage, age and ever-seen.
        Its ever-seen flag defines eligibility upstream, not a label-dependent rule.
        By default all inputs are detached, preserving frozen-router experiments.
        ``detach_inputs=False`` allows one-stage gradients through both the
        candidate logits and observed features. It adds no parameters or RNG use.
        """
        n = local.shape[0]
        if (local.shape != (n, self.local_dim)
                or base_text.shape != (n, self.context_dim)
                or gap_text.shape != (n, self.context_dim)
                or text_trajectory.shape != (n, 4)
                or candidates.ndim != 3 or candidates.shape[:2] != (n, 3)
                or eligible.shape != (n,) or eligible.dtype != torch.bool):
            raise ValueError('Expected aligned flat rows and three ordered candidate logits')
        if self.detach_inputs:
            local, base_text, gap_text, text_trajectory, candidates = (
                tensor.detach() for tensor in (local, base_text, gap_text, text_trajectory, candidates)
            )
        base_norm = torch.linalg.vector_norm(base_text, dim=-1)
        gap_norm = torch.linalg.vector_norm(gap_text, dim=-1)
        scalars = torch.stack((
            text_trajectory[:, 0].clamp_min(0).log1p(),
            text_trajectory[:, 1],
            text_trajectory[:, 2].clamp_min(0).log1p(),
            base_norm.log1p(),
            gap_norm.log1p(),
            F.cosine_similarity(base_text, gap_text, dim=-1, eps=1e-8),
        ), dim=-1)
        features = torch.cat((
            self.local_projection(local),
            self.base_projection(base_text),
            self.gap_projection(gap_text),
            scalars,
        ), dim=-1)
        scores = self.network(features)
        log_weights = F.log_softmax(scores, dim=-1)
        candidate_log_probs = F.log_softmax(candidates, dim=-1)
        mixed = torch.logsumexp(candidate_log_probs + log_weights.unsqueeze(-1), dim=1)
        # Preserve exact original probabilities outside the predeclared condition.
        log_probs = torch.where(eligible.unsqueeze(-1), mixed, candidate_log_probs[:, 0])
        full_only = torch.zeros_like(log_weights)
        full_only[:, 0] = 1
        weights = torch.where(eligible.unsqueeze(-1), F.softmax(scores, dim=-1), full_only)
        return log_probs, weights
