"""Complete-preserving missing-pattern FiLM relational propagation."""

from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter


_PATTERN_TO_COLUMN = {
    (1, 0, 0): 0,
    (0, 1, 0): 1,
    (0, 0, 1): 2,
    (1, 1, 0): 3,
    (1, 0, 1): 4,
    (0, 1, 1): 5,
}
_COMPLETE_PATTERN = (1, 1, 1)


def _validate_node_mask(node_mask: Tensor) -> Tensor:
    if node_mask.dim() != 2 or node_mask.size(-1) != 3:
        raise ValueError("node_mask must have shape [N,3]")
    binary = node_mask == node_mask.bool().to(node_mask.dtype)
    if not bool(binary.all()):
        raise ValueError("node_mask must contain only binary values")
    if bool((node_mask.sum(dim=-1) == 0).any()):
        raise ValueError("every node must retain at least one modality")
    return node_mask


def encode_missing_patterns(node_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Encode six incomplete patterns relative to the all-complete origin."""

    node_mask = _validate_node_mask(node_mask)
    pattern = node_mask.new_zeros((node_mask.size(0), 6))
    integer_rows = node_mask.detach().to(device="cpu", dtype=torch.long).tolist()
    for row_index, row in enumerate(integer_rows):
        key = tuple(row)
        if key == _COMPLETE_PATTERN:
            continue
        try:
            column = _PATTERN_TO_COLUMN[key]
        except KeyError as exc:
            raise ValueError(f"unsupported modality pattern: {key}") from exc
        pattern[row_index, column] = 1
    complete = node_mask.bool().all(dim=-1)
    return pattern, complete


def flatten_valid_node_masks(
    modality_mask: Tensor, lengths: Sequence[int]
) -> Tensor:
    """Match the conversation-major node order produced by ``batch_graphify``."""

    if modality_mask.dim() != 3 or modality_mask.size(-1) != 3:
        raise ValueError("modality_mask must have shape [T,B,3]")
    if len(lengths) != modality_mask.size(1):
        raise ValueError("length count must equal the batch dimension")
    pieces = []
    for batch_index, raw_length in enumerate(lengths):
        length = int(raw_length)
        if length < 0 or length > modality_mask.size(0):
            raise ValueError(f"invalid sequence length: {length}")
        pieces.append(modality_mask[:length, batch_index])
    flattened = torch.cat(pieces, dim=0) if pieces else modality_mask.new_empty((0, 3))
    _validate_node_mask(flattened)
    return flattened


class MissingPatternFiLMRGCNConv(nn.Module):
    """RGCN mean propagation with residual missing-pattern FiLM messages."""

    VARIANTS = ("full", "pattern_only", "content_film_control")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        variant: str = "full",
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown MPFiLM variant: {variant!r}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_relations = int(num_relations)
        self.variant = variant

        self.weight = nn.Parameter(
            torch.empty(num_relations, in_channels, out_channels)
        )
        self.root = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.pattern_weight = nn.Parameter(
            torch.empty(num_relations, 6, out_channels)
        )
        film_input_dim = 6 if variant == "pattern_only" else in_channels + 6
        self.film_weight = nn.Parameter(
            torch.empty(num_relations, film_input_dim, 2 * out_channels)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)
        zeros(self.pattern_weight)
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
            output = output + scatter(
                message,
                target,
                dim=0,
                dim_size=x.size(0),
                reduce="mean",
            )
        return output + x @ self.root + self.bias
