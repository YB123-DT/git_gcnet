from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ModalityMeans:
    audio: torch.Tensor
    text: torch.Tensor
    visual: torch.Tensor

    def to(self, device: torch.device | str) -> "ModalityMeans":
        return ModalityMeans(
            self.audio.to(device), self.text.to(device), self.visual.to(device)
        )


def select_speaker_features(
    host: torch.Tensor, guest: torch.Tensor, qmask: torch.Tensor
) -> torch.Tensor:
    """Select the active speaker's [seq,batch,dim] feature tensor."""
    speaker_is_guest = qmask.transpose(0, 1).unsqueeze(-1).bool()
    return torch.where(speaker_is_guest, guest, host)


def compute_modality_means(dataloader: Iterable) -> ModalityMeans:
    """Compute means from real utterances in the supplied training loader only."""
    sums = [None, None, None]
    count = 0
    for data in dataloader:
        qmask, umask = data[6], data[7]
        valid = umask.transpose(0, 1).bool()
        selected = (
            select_speaker_features(data[0], data[3], qmask),
            select_speaker_features(data[1], data[4], qmask),
            select_speaker_features(data[2], data[5], qmask),
        )
        batch_count = int(valid.sum().item())
        if batch_count == 0:
            continue
        for index, features in enumerate(selected):
            values = features[valid].to(dtype=torch.float64, device="cpu")
            value_sum = values.sum(dim=0)
            sums[index] = value_sum if sums[index] is None else sums[index] + value_sum
        count += batch_count
    if count == 0 or any(value is None for value in sums):
        raise ValueError("training loader contains no real utterances")
    means = [value.div(count).float() for value in sums]
    return ModalityMeans(means[0], means[1], means[2])
