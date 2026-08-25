"""Mask-conditioned attentional fusion for conversation sequences."""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MaskConditionedSequenceAFF(nn.Module):
    """Fuse two ``[T,B,D]`` branches while preserving complete utterances."""

    def __init__(
        self, channels: int, reduction: int = 4, pattern_dim: int = 6
    ) -> None:
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if not isinstance(reduction, int) or reduction <= 0:
            raise ValueError("reduction must be a positive integer")
        if pattern_dim != 6:
            raise ValueError("pattern_dim must be 6")

        self.channels = channels
        self.pattern_dim = pattern_dim
        self.register_buffer(
            "_pattern_lookup",
            torch.tensor([6, 2, 1, 5, 0, 4, 3, 6], dtype=torch.long),
            persistent=False,
        )
        bottleneck = max(channels // reduction, 2)
        self.local_context = self._make_context(bottleneck)
        self.global_context = self._make_context(bottleneck)
        self._zero_initialize_outputs()

    def _make_context(self, bottleneck: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.channels + self.pattern_dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, self.channels),
        )

    def _zero_initialize_outputs(self) -> None:
        for context in (self.local_context, self.global_context):
            nn.init.zeros_(context[-1].weight)
            nn.init.zeros_(context[-1].bias)

    @staticmethod
    def _is_binary(tensor: Tensor) -> Tensor:
        return (tensor == tensor.bool().to(tensor.dtype)).all()

    def _validate_inputs(
        self, x: Tensor, y: Tensor, modality_mask: Tensor, umask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if x.dim() != 3 or x.size(-1) != self.channels:
            raise ValueError(f"x must have shape [T,B,{self.channels}]")
        if y.shape != x.shape:
            raise ValueError("y must have the same shape as x")
        if not x.is_floating_point() or not y.is_floating_point():
            raise ValueError("x and y must have floating-point dtypes")
        if x.dtype != y.dtype:
            raise ValueError("x and y must have the same dtype")
        if x.device != y.device:
            raise ValueError("x and y must be on the same device")
        if modality_mask.dim() != 3 or modality_mask.shape != x.shape[:2] + (3,):
            raise ValueError("modality_mask must have shape [T,B,3]")
        if umask.dim() != 2 or umask.shape != (x.size(1), x.size(0)):
            raise ValueError("umask must have shape [B,T]")
        if modality_mask.is_complex():
            raise ValueError("modality_mask must contain only binary values")
        if umask.is_complex():
            raise ValueError("umask must contain only binary values")

        mask = modality_mask.to(device=x.device, dtype=x.dtype)
        valid_values = umask.to(device=x.device, dtype=x.dtype)
        binary_mask = self._is_binary(mask)
        binary_valid = self._is_binary(valid_values)
        valid = valid_values.t().bool()
        conversations_nonempty = valid.any(dim=0).all()
        valid_modalities_nonempty = ((mask.sum(dim=-1) > 0) | ~valid).all()
        checks = torch.stack(
            (
                binary_mask,
                binary_valid,
                conversations_nonempty,
                valid_modalities_nonempty,
            )
        )
        if not bool(checks.all()):
            if not bool(binary_mask):
                raise ValueError("modality_mask must contain only binary values")
            if not bool(binary_valid):
                raise ValueError("umask must contain only binary values")
            if not bool(conversations_nonempty):
                raise ValueError("every conversation must contain a valid utterance")
            raise ValueError("every valid utterance must retain a modality")
        return mask, valid

    def _encode_patterns(
        self, modality_mask: Tensor, valid: Tensor
    ) -> Tuple[Tensor, Tensor]:
        safe_mask = torch.where(
            valid.unsqueeze(-1), modality_mask, torch.ones_like(modality_mask)
        )
        bits = safe_mask.to(dtype=torch.long)
        codes = bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]
        columns = self._pattern_lookup[codes]
        pattern = F.one_hot(columns, num_classes=7)[..., : self.pattern_dim]
        pattern = pattern.to(dtype=modality_mask.dtype)
        incomplete = (codes != 7) & valid
        return pattern, incomplete

    def forward(
        self, x: Tensor, y: Tensor, modality_mask: Tensor, umask: Tensor
    ) -> Tensor:
        mask, valid = self._validate_inputs(x, y, modality_mask, umask)
        pattern, incomplete = self._encode_patterns(mask, valid)

        base = x + y
        local_input = torch.cat((base, pattern), dim=-1)
        local = self.local_context(local_input)

        valid_weight = valid.to(x.dtype).unsqueeze(-1)
        count = valid_weight.sum(dim=0)
        valid_base = base.masked_fill(~valid.unsqueeze(-1), 0)
        base_mean = valid_base.sum(dim=0) / count
        pattern_mean = (pattern * valid_weight).sum(dim=0) / count
        global_input = torch.cat((base_mean, pattern_mean), dim=-1)
        global_context = self.global_context(global_input).unsqueeze(0)

        gate = torch.sigmoid(local + global_context)
        aff = 2 * (gate * x + (1 - gate) * y)
        return torch.where(incomplete.unsqueeze(-1), aff, base)
