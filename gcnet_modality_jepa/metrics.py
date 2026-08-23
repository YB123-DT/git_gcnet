from __future__ import annotations

import torch
import torch.nn.functional as F


def _effective_rank(matrix: torch.Tensor) -> float:
    if hasattr(torch.linalg, "svdvals"):
        singular_values = torch.linalg.svdvals(matrix.float())
    else:
        singular_values = torch.svd(matrix.float(), some=False).S
    total = singular_values.sum()
    if singular_values.numel() == 0 or total <= 1e-12:
        return 1.0
    probabilities = singular_values / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


def compute_epoch_collapse_diagnostics(
    temporal_pre: torch.Tensor,
    temporal_hidden: torch.Tensor,
    speaker_pre: torch.Tensor,
    speaker_hidden: torch.Tensor,
    final_hidden: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    regression_head: torch.nn.Linear,
) -> dict[str, float]:
    """Summarize one training epoch for MOSI/MOSEI collapse diagnosis.

    All representation tensors are expected to contain only real utterances
    (padding removed) and have shape [N, D]. Predictions are [N, 1].
    """
    tensors = (
        temporal_pre,
        temporal_hidden,
        speaker_pre,
        speaker_hidden,
        final_hidden,
        predictions,
    )
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("diagnostic representations must have shape [N, D]")
    count = final_hidden.shape[0]
    if count == 0 or any(tensor.shape[0] != count for tensor in tensors):
        raise ValueError("diagnostic tensors must contain the same nonzero N")
    if labels.numel() != count:
        raise ValueError("labels must contain one value per real utterance")

    temporal_pre = temporal_pre.detach().float().cpu()
    temporal_hidden = temporal_hidden.detach().float().cpu()
    speaker_pre = speaker_pre.detach().float().cpu()
    speaker_hidden = speaker_hidden.detach().float().cpu()
    final_hidden = final_hidden.detach().float().cpu()
    predictions = predictions.detach().float().cpu().reshape(count, -1)
    labels = labels.detach().float().cpu().reshape(-1)
    centered_hidden = final_hidden - final_hidden.mean(dim=0, keepdim=True)
    head_weight = regression_head.weight.detach().float().cpu()
    head_bias = regression_head.bias.detach().float().cpu()
    label_mean = labels.mean()
    bias_mean = head_bias.mean()

    def distribution(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
        flattened = tensor.reshape(-1)
        probabilities = torch.tensor(
            [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99], dtype=torch.float32
        )
        quantiles = torch.quantile(flattened, probabilities)
        names = ("q01", "q10", "q25", "q50", "q75", "q90", "q99")
        values = {
            f"{prefix}_{name}": float(value.item())
            for name, value in zip(names, quantiles)
        }
        values[f"{prefix}_mean"] = float(flattened.mean().item())
        values[f"{prefix}_std"] = float(flattened.std(unbiased=False).item())
        return values

    result = {
        "utterance_count": float(count),
        "temporal_pre_nonpositive_ratio": float(
            (temporal_pre <= 0).float().mean().item()
        ),
        "speaker_pre_nonpositive_ratio": float(
            (speaker_pre <= 0).float().mean().item()
        ),
        "temporal_zero_ratio": float(
            (temporal_hidden == 0).float().mean().item()
        ),
        "speaker_zero_ratio": float(
            (speaker_hidden == 0).float().mean().item()
        ),
        "final_hidden_std": float(
            final_hidden.std(dim=0, unbiased=False).mean().item()
        ),
        "final_hidden_effective_rank": _effective_rank(centered_hidden),
        "regression_head_weight_norm": float(head_weight.norm().item()),
        "regression_bias": float(bias_mean.item()),
        "training_label_mean": float(label_mean.item()),
        "regression_bias_label_mean_gap": float(
            (bias_mean - label_mean).abs().item()
        ),
        "prediction_mean": float(predictions.mean().item()),
        "prediction_std": float(predictions.std(unbiased=False).item()),
    }
    result.update(distribution("temporal_pre", temporal_pre))
    result.update(distribution("speaker_pre", speaker_pre))
    return result


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
