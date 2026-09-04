"""Text-only temporal regression model for frozen utterance features."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextOnlyTemporalModel(nn.Module):
    """Project frozen text features, model their sequence, and regress sentiment."""

    def __init__(self, text_dim: int, hidden_dim: int = 200, dropout: float = 0.5) -> None:
        super().__init__()
        if text_dim <= 0 or hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("text_dim and an even hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text: Tensor, umask: Tensor):
        if text.ndim != 3:
            raise ValueError("text must have shape [L,B,D]")
        length, batch, _ = text.shape
        if umask.shape != (batch, length):
            raise ValueError("umask must have shape [B,L]")
        lengths = umask.gt(0).sum(dim=1).to(dtype=torch.long, device="cpu")
        if bool((lengths <= 0).any()):
            raise ValueError("every sequence must contain a valid utterance")
        projected = self.text_projection(text)
        packed = pack_padded_sequence(projected, lengths, enforce_sorted=False)
        packed_hidden, _ = self.temporal(packed)
        hidden, _ = pad_packed_sequence(packed_hidden, total_length=length)
        valid = umask.transpose(0, 1).to(hidden.dtype).unsqueeze(-1)
        hidden = hidden * valid
        prediction = self.regressor(hidden) * valid
        return prediction, hidden


__all__ = ["TextOnlyTemporalModel"]
