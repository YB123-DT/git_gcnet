"""The hierarchical cosine objective used by PLCI-JEPA."""

from typing import Dict, List, Tuple

import torch

from .modules import MODALITIES, PLCIPredictions, normalize_latent


def _empty_counts() -> Dict[str, int]:
    counts = {"utterances": 0, "targets": 0, "paths": 0}
    for name in MODALITIES:
        counts[name + "_targets"] = 0
        counts[name + "_paths"] = 0
    return counts


def plci_jepa_loss(
    predictions: PLCIPredictions,
    teacher_targets: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Average paths within targets, targets within utterances, then utterances."""
    counts = _empty_counts()
    if not predictions.targets:
        reference = getattr(predictions, "_reference", None)
        if reference is not None:
            return reference.sum() * 0.0, counts
        return torch.tensor(0.0, requires_grad=True), counts
    if set(teacher_targets) != set(MODALITIES):
        raise ValueError("teacher_targets must contain audio, text, and visual")

    first = next(iter(teacher_targets.values()))
    if first.ndim != 3:
        raise ValueError("teacher targets must have shape [L, B, latent_dim]")
    length, batch = first.shape[:2]
    for name in MODALITIES:
        value = teacher_targets[name]
        if value.ndim != 3 or value.shape[:2] != (length, batch):
            raise ValueError("teacher target leading dimensions differ")

    utterance_losses = {}  # type: Dict[int, List[torch.Tensor]]
    for record in predictions.targets:
        if record.target_modality < 0 or record.target_modality >= len(MODALITIES):
            raise ValueError("target_modality is invalid")
        if record.utterance_index < 0 or record.utterance_index >= length * batch:
            raise ValueError("utterance_index is invalid")
        name = MODALITIES[record.target_modality]
        target = teacher_targets[name].reshape(length * batch, -1)[
            record.utterance_index
        ].detach()
        target = normalize_latent(target)
        paths = normalize_latent(record.paths)
        target_loss = (1.0 - torch.sum(paths * target, dim=-1)).mean()
        utterance_losses.setdefault(record.utterance_index, []).append(target_loss)
        counts["targets"] += 1
        counts["paths"] += int(record.paths.shape[0])
        counts[name + "_targets"] += 1
        counts[name + "_paths"] += int(record.paths.shape[0])

    per_utterance = [torch.stack(losses).mean() for losses in utterance_losses.values()]
    counts["utterances"] = len(per_utterance)
    return torch.stack(per_utterance).mean(), counts
