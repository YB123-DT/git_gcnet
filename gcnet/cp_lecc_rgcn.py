"""Complete-preserving low-rank edge-conditioned RGCN propagation."""

import torch
from torch import Tensor, nn
from torch_geometric.nn import RGCNConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter

from missing_patterns import encode_missing_patterns


class CompletePreservingLowRankECCConv(RGCNConv):
    """Add low-rank, missingness-conditioned corrections to an RGCN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        content_dim: int = 16,
        relation_dim: int = 8,
        generator_hidden: int = 32,
        num_bases: int = 4,
        basis_rank: int = 8,
    ) -> None:
        super().__init__(in_channels, out_channels, num_relations)
        self.content_dim = content_dim
        self.relation_dim = relation_dim
        self.generator_hidden = generator_hidden
        self.num_dynamic_bases = num_bases
        self.basis_rank = basis_rank

        descriptor_dim = 18 + relation_dim + content_dim
        self.target_content = nn.Parameter(torch.empty(in_channels, content_dim))
        self.source_content = nn.Parameter(torch.empty(in_channels, content_dim))
        self.relation_embedding = nn.Parameter(
            torch.empty(num_relations, relation_dim)
        )
        self.generator_hidden_weight = nn.Parameter(
            torch.empty(descriptor_dim, generator_hidden)
        )
        self.generator_hidden_bias = nn.Parameter(torch.empty(generator_hidden))
        self.generator_output_weight = nn.Parameter(
            torch.empty(generator_hidden, num_bases)
        )
        self.generator_output_bias = nn.Parameter(torch.empty(num_bases))
        self.basis_left = nn.Parameter(
            torch.empty(num_bases, in_channels, basis_rank)
        )
        self.basis_right = nn.Parameter(
            torch.empty(num_bases, basis_rank, out_channels)
        )
        self._reset_dynamic_parameters_preserving_rng()

    def _reset_dynamic_parameters_preserving_rng(self) -> None:
        rng_state = torch.random.get_rng_state()
        try:
            glorot(self.target_content)
            glorot(self.source_content)
            glorot(self.relation_embedding)
            glorot(self.generator_hidden_weight)
            zeros(self.generator_hidden_bias)
            zeros(self.generator_output_weight)
            zeros(self.generator_output_bias)
            glorot(self.basis_left)
            glorot(self.basis_right)
        finally:
            torch.random.set_rng_state(rng_state)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "target_content"):
            self._reset_dynamic_parameters_preserving_rng()

    def _validate_inputs(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_mask: Tensor,
    ) -> None:
        if x.dim() != 2 or x.size(1) != self.in_channels:
            raise ValueError(f"x must have shape [N,{self.in_channels}]")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must contain integer node indices")
        if edge_type.dim() != 1 or edge_type.numel() != edge_index.size(1):
            raise ValueError("edge_type must have shape [E]")
        if edge_type.dtype != torch.long:
            raise ValueError("edge_type must contain integer relation indices")
        if node_mask.dim() != 2 or node_mask.size(1) != 3:
            raise ValueError("node_mask must have shape [N,3]")
        if node_mask.size(0) != x.size(0):
            raise ValueError("node_mask and x must contain the same nodes")
        if edge_index.numel() > 0:
            if bool((edge_index < 0).any()) or bool((edge_index >= x.size(0)).any()):
                raise ValueError("edge_index contains an out-of-range node index")
            if bool((edge_type < 0).any()) or bool(
                (edge_type >= self.num_relations).any()
            ):
                raise ValueError("edge_type contains an out-of-range relation")

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        self._validate_inputs(x, edge_index, edge_type, node_mask)
        if node_mask.device != x.device:
            node_mask = node_mask.to(x.device)
        pattern, complete = encode_missing_patterns(node_mask.to(dtype=x.dtype))
        if bool(complete.all()):
            return super().forward(x, edge_index, edge_type)

        base_output = super().forward(x, edge_index, edge_type)
        source = edge_index[0]
        target = edge_index[1]
        content_pair = (x[target] @ self.target_content) * (
            x[source] @ self.source_content
        )
        descriptor = torch.cat(
            (
                pattern[target],
                pattern[source],
                pattern[target] * pattern[source],
                self.relation_embedding[edge_type],
                content_pair,
            ),
            dim=-1,
        )
        hidden = torch.relu(
            descriptor @ self.generator_hidden_weight
            + self.generator_hidden_bias
        )
        coefficient = torch.tanh(
            hidden @ self.generator_output_weight
            + self.generator_output_bias
        )

        projected = torch.einsum("ei,kir->ekr", x[source], self.basis_left)
        basis_messages = torch.einsum(
            "ekr,kro->eko", projected, self.basis_right
        )
        correction_message = (basis_messages * coefficient.unsqueeze(-1)).sum(dim=1)
        active = (~(complete[source] & complete[target])).to(x.dtype).unsqueeze(-1)
        correction_message = correction_message * active

        correction = x.new_zeros((x.size(0), self.out_channels))
        for relation in range(self.num_relations):
            relation_edges = edge_type == relation
            if bool(relation_edges.any()):
                correction = correction + scatter(
                    correction_message[relation_edges],
                    target[relation_edges],
                    dim=0,
                    dim_size=x.size(0),
                    reduce="mean",
                )
        return base_output + correction
