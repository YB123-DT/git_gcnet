from __future__ import annotations

import torch
import torch.nn.functional as F


def _effective_rank(matrix: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(matrix.float())
    total = singular_values.sum()
    if singular_values.numel() == 0 or total <= 1e-12:
        return 1.0
    probabilities = singular_values / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


def compute_modality_diagnostics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    shuffle_seed: int = 66,
) -> dict[str, float | int | None]:
    """Compute deterministic anti-shortcut diagnostics for one modality."""
    if predictions.ndim != 2 or targets.shape != predictions.shape:
        raise ValueError("predictions and targets must share shape [N, D]")
    count = predictions.shape[0]
    if count < 2:
        return {
            "count": count,
            "real_cosine": None,
            "shuffled_cosine": None,
            "real_shuffle_gap": None,
            "prediction_std": None,
            "target_std": None,
            "prediction_effective_rank": None,
            "target_effective_rank": None,
        }
    predictions = predictions.detach().float().cpu()
    targets = targets.detach().float().cpu()
    generator = torch.Generator().manual_seed(shuffle_seed)
    permutation = torch.randperm(count, generator=generator)
    if torch.equal(permutation, torch.arange(count)):
        permutation = torch.roll(permutation, shifts=1)
    real = F.cosine_similarity(predictions, targets, dim=-1).mean().item()
    shuffled = F.cosine_similarity(
        predictions, targets[permutation], dim=-1
    ).mean().item()
    return {
        "count": count,
        "real_cosine": real,
        "shuffled_cosine": shuffled,
        "real_shuffle_gap": real - shuffled,
        "prediction_std": predictions.std(dim=0, unbiased=False).mean().item(),
        "target_std": targets.std(dim=0, unbiased=False).mean().item(),
        "prediction_effective_rank": _effective_rank(predictions),
        "target_effective_rank": _effective_rank(targets),
    }
