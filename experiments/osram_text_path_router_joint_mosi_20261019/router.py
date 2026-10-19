"""Three-candidate Text-source routing for the unchanged MOSI scalar task.

Only the route weights are normalized. Candidate sentiment scores are never
passed through Softmax or LogSoftmax: their weighted sum retains scalar MSE
supervision and the existing regression head.
"""

import torch
from torch.nn import functional as F

from experiments.osram_text_path_router_20261019.router import TextPathRouter


class RegressionTextPathRouter(TextPathRouter):
    """Reuse the classification Router's features and parameters, not its loss."""

    def __init__(self, local_dim=256, context_dim=512, projection_dim=64,
                 hidden_dim=64, detach_inputs=False):
        super().__init__(local_dim, context_dim, projection_dim, hidden_dim,
                         detach_inputs=detach_inputs)

    def forward(self, local, base_text, gap_text, text_trajectory, candidates, eligible):
        """Return scalar predictions and [full, Base-retained, Gap-retained] weights.

        Eligibility is the upstream, label-free condition: current Text is
        absent and strictly-past Text was observed. Other rows exactly preserve
        the full candidate. Uniform initialization is not baseline identity.
        """
        n = local.shape[0] if local.ndim == 2 else -1
        if (local.shape != (n, self.local_dim)
                or base_text.shape != (n, self.context_dim)
                or gap_text.shape != (n, self.context_dim)
                or text_trajectory.shape != (n, 4)
                or candidates.shape != (n, 3, 1)
                or eligible.shape != (n,) or eligible.dtype != torch.bool):
            raise ValueError('Expected aligned flat rows and three ordered scalar candidates')
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
        route_weights = F.softmax(self.network(features), dim=-1)
        full_only = torch.zeros_like(route_weights)
        full_only[:, 0] = 1
        weights = torch.where(eligible[:, None], route_weights, full_only)
        mixed = (candidates * weights[..., None]).sum(1)
        prediction = torch.where(eligible[:, None], mixed, candidates[:, 0])
        return prediction, weights
