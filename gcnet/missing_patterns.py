"""Shared missing-pattern encoding and node-mask ordering utilities."""

from typing import Sequence, Tuple

import torch
from torch import Tensor


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
