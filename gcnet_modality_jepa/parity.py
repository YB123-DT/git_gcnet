from __future__ import annotations

import torch
from torch import nn


def load_shared_backbone(baseline: nn.Module, jepa: nn.Module) -> None:
    """Load one baseline checkpoint into the shared JEPA backbone parameters."""
    incompatible = jepa.load_state_dict(baseline.state_dict(), strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    if unexpected:
        raise ValueError(f"unexpected baseline keys: {unexpected}")
    if any(not key.startswith("modality_predictor.") for key in missing):
        raise ValueError(f"non-predictor JEPA keys were not initialized: {missing}")


def compare_shared_tensors(baseline: nn.Module, jepa: nn.Module) -> float:
    """Return max absolute difference over all baseline state tensors."""
    baseline_state = baseline.state_dict()
    jepa_state = jepa.state_dict()
    differences = []
    for name, baseline_tensor in baseline_state.items():
        if name not in jepa_state:
            raise KeyError(f"JEPA model is missing shared tensor {name}")
        differences.append(
            torch.max(torch.abs(baseline_tensor - jepa_state[name])).item()
        )
    return max(differences, default=0.0)


def compare_shared_gradients(baseline: nn.Module, jepa: nn.Module) -> float:
    """Return max absolute gradient difference over shared parameters."""
    jepa_parameters = dict(jepa.named_parameters())
    differences = []
    for name, baseline_parameter in baseline.named_parameters():
        if name not in jepa_parameters:
            raise KeyError(f"JEPA model is missing shared parameter {name}")
        baseline_gradient = baseline_parameter.grad
        jepa_gradient = jepa_parameters[name].grad
        if baseline_gradient is None and jepa_gradient is None:
            continue
        if baseline_gradient is None or jepa_gradient is None:
            raise ValueError(f"gradient presence differs for shared parameter {name}")
        differences.append(
            torch.max(torch.abs(baseline_gradient - jepa_gradient)).item()
        )
    return max(differences, default=0.0)


def miss0_jepa_loss(model: nn.Module) -> tuple[torch.Tensor, float]:
    """Return a detached zero; no JEPA autograd path exists at miss=0."""
    parameter = next(model.parameters())
    return parameter.new_zeros(()), 0.0
