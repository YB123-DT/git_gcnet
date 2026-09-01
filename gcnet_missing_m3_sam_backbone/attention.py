"""Availability-safe attention primitives for the SAM-style backbone."""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
from torch import Tensor, nn


class SafeDirectedAttention(nn.Module):
    """Directed attention that never evaluates an all-masked key sequence."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.query_norm = nn.LayerNorm(width)
        self.key_value_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(width)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        query_valid: Tensor,
        key_valid: Tensor,
        return_weights: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        if query.ndim != 3 or key_value.ndim != 3:
            raise ValueError("query and key_value must have shape [L,B,D]")
        if query.shape != key_value.shape:
            raise ValueError("query and key_value must have identical shapes")
        expected_mask_shape = query.shape[:2]
        if query_valid.shape != expected_mask_shape:
            raise ValueError("query_valid must have shape [L,B]")
        if key_valid.shape != expected_mask_shape:
            raise ValueError("key_valid must have shape [L,B]")

        query_valid = query_valid.bool()
        key_valid = key_valid.bool()
        length, batch_size, _ = query.shape
        output = torch.zeros_like(query)
        weights = query.new_zeros((batch_size, length, length))
        active = query_valid & key_valid.any(dim=0).unsqueeze(0)
        normalized_query = self.query_norm(query)
        normalized_key_value = self.key_value_norm(key_value)

        for batch_index in range(batch_size):
            if not bool(key_valid[:, batch_index].any()):
                continue
            attended, attention_weights = self.attention(
                normalized_query[:, batch_index : batch_index + 1],
                normalized_key_value[:, batch_index : batch_index + 1],
                normalized_key_value[:, batch_index : batch_index + 1],
                key_padding_mask=(~key_valid[:, batch_index]).unsqueeze(0),
                need_weights=True,
            )
            residual = self.output_norm(
                query[:, batch_index : batch_index + 1]
                + self.dropout(attended)
            )
            batch_active = active[:, batch_index].to(query.dtype).view(
                length, 1, 1
            )
            output[:, batch_index : batch_index + 1] = residual * batch_active
            weights[batch_index] = (
                attention_weights[0]
                * active[:, batch_index].to(query.dtype).unsqueeze(-1)
            )

        if return_weights:
            return output, active, weights
        return output, active


class MaskedTrackPooling(nn.Module):
    """Pool a variable set of modality-interaction tracks per utterance."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width
        self.track_norm = nn.LayerNorm(width)
        self.query = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.query, mean=0.0, std=width ** -0.5)

    def forward(
        self,
        tracks: Tensor,
        track_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if tracks.ndim != 4:
            raise ValueError("tracks must have shape [L,B,K,D]")
        if tracks.shape[-1] != self.width:
            raise ValueError("track width does not match the module width")
        if track_valid.shape != tracks.shape[:3]:
            raise ValueError("track_valid must have shape [L,B,K]")

        track_valid = track_valid.bool()
        scores = torch.einsum(
            "lbkd,d->lbk",
            self.track_norm(tracks),
            self.query,
        ) / math.sqrt(self.width)
        flat_scores = scores.reshape(-1, scores.shape[-1])
        flat_valid = track_valid.reshape(-1, track_valid.shape[-1])
        flat_weights = torch.zeros_like(flat_scores)
        active_rows = flat_valid.any(dim=-1)
        if bool(active_rows.any()):
            active_scores = flat_scores[active_rows].masked_fill(
                ~flat_valid[active_rows],
                float("-inf"),
            )
            flat_weights[active_rows] = torch.softmax(active_scores, dim=-1)
        weights = flat_weights.reshape_as(scores)
        pooled = torch.sum(tracks * weights.unsqueeze(-1), dim=2)
        return pooled, weights


__all__ = ["MaskedTrackPooling", "SafeDirectedAttention"]
