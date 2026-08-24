"""Complete-preserving missing-pattern FiLM relational propagation."""

import torch
from torch import Tensor, nn
from torch_geometric.nn import RGCNConv
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter

from missing_patterns import encode_missing_patterns


class MissingPatternFiLMRGCNConv(RGCNConv):
    """RGCN mean propagation with residual missing-pattern FiLM messages."""

    VARIANTS = (
        "full",
        "faithful_edgewise",
        "pattern_only",
        "content_film_control",
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        variant: str = "full",
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown MPFiLM variant: {variant!r}")
        super().__init__(in_channels, out_channels, num_relations)
        self.variant = variant
        self.pattern_weight = nn.Parameter(
            torch.empty(num_relations, 6, out_channels)
        )
        film_input_dim = 6 if variant == "pattern_only" else in_channels + 6
        self.film_weight = nn.Parameter(
            torch.empty(num_relations, film_input_dim, 2 * out_channels)
        )
        zeros(self.pattern_weight)
        zeros(self.film_weight)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "pattern_weight"):
            zeros(self.pattern_weight)
        if hasattr(self, "film_weight"):
            zeros(self.film_weight)

    def _film_input(self, x: Tensor, pattern: Tensor) -> Tensor:
        if self.variant == "pattern_only":
            return pattern
        if self.variant == "content_film_control":
            return torch.cat((x, torch.zeros_like(pattern)), dim=-1)
        return torch.cat((x, pattern), dim=-1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        if x.dim() != 2 or x.size(-1) != self.in_channels:
            raise ValueError(
                f"x must have shape [N,{self.in_channels}]"
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_type.dim() != 1 or edge_type.numel() != edge_index.size(1):
            raise ValueError("edge_type must have shape [E]")
        if node_mask.device != x.device:
            node_mask = node_mask.to(x.device)
        node_mask = node_mask.to(dtype=x.dtype)
        pattern, complete = encode_missing_patterns(node_mask)
        if pattern.size(0) != x.size(0):
            raise ValueError("node_mask and x must contain the same nodes")
        if self.variant != "content_film_control" and bool(complete.all()):
            return super().forward(x, edge_index, edge_type)

        output = x.new_zeros((x.size(0), self.out_channels))
        film_input = self._film_input(x, pattern)
        for relation in range(self.num_relations):
            relation_edges = edge_type == relation
            if not bool(relation_edges.any()):
                continue
            source = edge_index[0, relation_edges]
            target = edge_index[1, relation_edges]
            base_message = x[source] @ self.weight[relation]
            if self.variant == "content_film_control":
                source_message = base_message
            else:
                source_message = (
                    base_message + pattern[source] @ self.pattern_weight[relation]
                )
            film = film_input @ self.film_weight[relation]
            delta_gamma, delta_beta = film.chunk(2, dim=-1)
            if self.variant == "content_film_control":
                active = x.new_ones((source.numel(), 1))
            else:
                active = (
                    1.0
                    - (complete[source] & complete[target]).to(x.dtype)
                ).unsqueeze(-1)
            message = (
                (1.0 + active * delta_gamma[target]) * source_message
                + active * delta_beta[target]
            )
            if self.variant == "faithful_edgewise":
                message = torch.where(
                    active.bool(), torch.relu(message), message
                )
            output = output + scatter(
                message,
                target,
                dim=0,
                dim_size=x.size(0),
                reduce="mean",
            )
        return output + x @ self.root + self.bias
