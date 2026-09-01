"""Text-anchored residual fusion for frozen utterance features."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn

from gcnet_missing_m3_sam_backbone.attention import (
    MaskedTrackPooling,
    SafeDirectedAttention,
)
from gcnet_missing_m3_sam_backbone.model import ModalityTemporalEncoder


class TextAnchoredResidualModel(nn.Module):
    """Use Text as the semantic base and Audio/Visual as bounded residuals."""

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        visual_dim: int,
        *,
        width: int = 120,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.width = int(width)
        self.input_dims = (int(audio_dim), int(text_dim), int(visual_dim))
        self.encoders = nn.ModuleList(
            ModalityTemporalEncoder(dim, width, heads, dropout)
            for dim in self.input_dims
        )
        self.text_to_audio = SafeDirectedAttention(width, heads, dropout)
        self.text_to_visual = SafeDirectedAttention(width, heads, dropout)
        self.audio_gate = nn.Linear(2 * width, width)
        self.visual_gate = nn.Linear(2 * width, width)
        nn.init.constant_(self.audio_gate.bias, -2.0)
        nn.init.constant_(self.visual_gate.bias, -2.0)
        self.fallback_pool = MaskedTrackPooling(width)
        self.output_block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout),
            nn.LayerNorm(width),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def _validate_inputs(
        self,
        features: Sequence[Tensor],
        availability: Tensor,
        umask: Tensor,
    ) -> Tensor:
        if len(features) != 3:
            raise ValueError("features must contain audio, text, and visual")
        leading = features[0].shape[:2]
        for value, expected_dim in zip(features, self.input_dims):
            if value.ndim != 3 or value.shape[:2] != leading:
                raise ValueError("all modality features must share [L,B]")
            if value.shape[-1] != expected_dim:
                raise ValueError("modality feature width is incorrect")
        if availability.shape != leading + (3,):
            raise ValueError("availability must have shape [L,B,3]")
        if umask.shape != (leading[1], leading[0]):
            raise ValueError("umask must have shape [B,L]")
        valid = umask.transpose(0, 1).bool()
        observed_count = availability.gt(0).sum(dim=-1)
        if bool((valid & observed_count.eq(0)).any()):
            raise ValueError("each valid utterance needs at least one observed modality")
        return valid

    def forward(
        self,
        features: Sequence[Tensor],
        availability: Tensor,
        umask: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, object]]:
        valid = self._validate_inputs(features, availability, umask)
        modality_valid = availability.gt(0) & valid.unsqueeze(-1)
        encoded: List[Tensor] = [
            encoder(value, modality_valid[..., index])
            for index, (encoder, value) in enumerate(zip(self.encoders, features))
        ]
        audio, text, visual = encoded
        text_valid = modality_valid[..., 1]
        audio_context, audio_active, audio_weights = self.text_to_audio(
            text,
            audio,
            text_valid,
            modality_valid[..., 0],
            return_weights=True,
        )
        visual_context, visual_active, visual_weights = self.text_to_visual(
            text,
            visual,
            text_valid,
            modality_valid[..., 2],
            return_weights=True,
        )
        audio_gate = torch.sigmoid(
            self.audio_gate(torch.cat([text, audio_context], dim=-1))
        ) * audio_active.to(text.dtype).unsqueeze(-1)
        visual_gate = torch.sigmoid(
            self.visual_gate(torch.cat([text, visual_context], dim=-1))
        ) * visual_active.to(text.dtype).unsqueeze(-1)
        anchored = text + audio_gate * audio_context + visual_gate * visual_context

        fallback, fallback_weights = self.fallback_pool(
            torch.stack([audio, visual], dim=2),
            torch.stack(
                [modality_valid[..., 0], modality_valid[..., 2]],
                dim=2,
            ),
        )
        use_text = text_valid.unsqueeze(-1)
        fused = torch.where(use_text, anchored, fallback)
        valid_float = valid.to(fused.dtype).unsqueeze(-1)
        hidden = self.output_block(fused) * valid_float
        prediction = self.regressor(hidden) * valid_float
        return prediction, hidden, {
            "cross_attention": {
                "text-audio": audio_weights,
                "text-visual": visual_weights,
            },
            "audio_gate": audio_gate,
            "visual_gate": visual_gate,
            "fallback_pooling": fallback_weights,
            "used_text_anchor": text_valid,
        }


__all__ = ["TextAnchoredResidualModel"]
