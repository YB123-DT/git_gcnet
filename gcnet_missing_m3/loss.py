"""Missing-target latent objective for the single-view model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional as F

from .model import MODALITIES, MissingM3Predictions


@dataclass(frozen=True)
class MissingM3Loss:
    total: torch.Tensor
    regression: torch.Tensor
    contrastive: torch.Tensor
    target_count: int


def _symmetric_info_nce(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(prediction.shape[0], device=prediction.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def missing_m3_loss(
    predictions: MissingM3Predictions,
    teacher_targets: Mapping[str, torch.Tensor],
    temperature: float = 0.03,
    regression_aggregation: str = "target",
    contrastive_prediction_source: str = "contrastive",
) -> MissingM3Loss:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if regression_aggregation not in {"target", "utterance"}:
        raise ValueError(
            "regression_aggregation must be 'target' or 'utterance'"
        )
    if contrastive_prediction_source not in {"contrastive", "regression"}:
        raise ValueError(
            "contrastive_prediction_source must be 'contrastive' or "
            "'regression'"
        )
    if set(teacher_targets) != set(MODALITIES):
        raise ValueError("teacher_targets must contain audio, text, and visual")
    target_count = int(predictions.target_mask.sum().item())
    zero = predictions.reg_predictions.sum() * 0.0
    if target_count == 0:
        return MissingM3Loss(zero, zero, zero, 0)

    regression_values = []
    contrastive_values = []
    regression_sum = predictions.reg_predictions.new_zeros(
        predictions.target_mask.shape[:2]
    )
    for target_index, name in enumerate(MODALITIES):
        selected = predictions.target_mask[..., target_index]
        if not bool(selected.any()):
            continue
        target = teacher_targets[name][selected].detach()
        reg_prediction = predictions.reg_predictions[..., target_index, :][selected]
        cl_prediction = predictions.cl_predictions[..., target_index, :][selected]
        target_regression = F.smooth_l1_loss(
            reg_prediction, target, reduction="none"
        ).mean(-1)
        regression_values.append(target_regression)
        if regression_aggregation == "utterance":
            regression_sum = regression_sum.index_put(
                selected.nonzero(as_tuple=True),
                target_regression,
                accumulate=True,
            )
        if target.shape[0] >= 2:
            contrastive_prediction = (
                reg_prediction
                if contrastive_prediction_source == "regression"
                else cl_prediction
            )
            contrastive_values.append(
                _symmetric_info_nce(
                    contrastive_prediction, target, temperature
                )
            )

    if regression_aggregation == "target":
        regression = torch.cat(regression_values).mean()
    else:
        target_counts = predictions.target_mask.sum(dim=-1)
        predicted_utterance = target_counts > 0
        regression = (
            regression_sum[predicted_utterance]
            / target_counts[predicted_utterance].to(regression_sum.dtype)
        ).mean()
    if contrastive_values:
        contrastive = torch.stack(contrastive_values).mean()
        total = 0.5 * regression + 0.5 * contrastive
    else:
        contrastive = zero
        total = regression
    return MissingM3Loss(total, regression, contrastive, target_count)
