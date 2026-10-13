"""The batch-level missing-view sampler used by the original GCNet code."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch


ALGORITHM = "gcnet-random-mask-v1"


def random_mask(
    view_num: int,
    input_len: int,
    missing_rate: float,
    rng=np.random,
) -> np.ndarray:
    """Reproduce GCNet's ``random_mask`` sampling algorithm.

    The implementation preserves its batch-level count tolerance, minimum-one
    modality rule, 32-row sampling floor, and high-missing one-hot branch.
    """
    if view_num < 1 or input_len < 1:
        raise ValueError("view_num and input_len must be positive")
    one_rate = 1.0 - float(missing_rate)
    if not math.isfinite(one_rate) or one_rate < 0.0 or one_rate > 1.0:
        raise ValueError("missing_rate must be between zero and one")

    if one_rate <= 1.0 / view_num:
        selected = rng.randint(0, view_num, size=input_len)
        return np.eye(view_num, dtype=np.int64)[selected]
    if one_rate == 1.0:
        return rng.randint(1, 2, size=(input_len, view_num))

    sample_len = max(input_len, 32)
    error = 1.0
    while error >= 0.005:
        selected = rng.randint(0, view_num, size=sample_len)
        preserved = np.eye(view_num, dtype=np.int64)[selected]
        one_num = view_num * sample_len * one_rate - sample_len
        ratio = one_num / (view_num * sample_len)
        proposal = (
            rng.randint(0, 100, size=(sample_len, view_num)) < int(ratio * 100)
        ).astype(np.int64)
        overlap = np.sum(((proposal + preserved) > 1).astype(np.int64))
        corrected = one_num / (1.0 - overlap / one_num)
        ratio = corrected / (view_num * sample_len)
        proposal = (
            rng.randint(0, 100, size=(sample_len, view_num)) < int(ratio * 100)
        ).astype(np.int64)
        matrix = ((proposal + preserved) > 0).astype(np.int64)
        error = abs(one_rate - np.sum(matrix) / (view_num * sample_len))
    return matrix[:input_len]


class GCNetLegacyRandomMaskSchedule:
    """Adapter exposing original GCNet batch sampling to the OSRAM trainer."""

    def __init__(
        self,
        dataset: str,
        split: str,
        fold: int | str,
        requested_missing_rate: float,
        mask_seed: int,
        freeze_evaluation: bool = True,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.fold = fold
        self.requested_missing_rate = float(requested_missing_rate)
        self.mask_seed = int(mask_seed)
        self.freeze_evaluation = bool(freeze_evaluation)
        self.config_hash = hashlib.sha256(
            json.dumps(
                {
                    "algorithm": ALGORITHM,
                    "dataset": dataset,
                    "split": split,
                    "fold": fold,
                    "rate": format(self.requested_missing_rate, ".17g"),
                    "seed": mask_seed,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def generate_batch(
        self,
        conversation_ids,
        umask: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del epoch
        if umask.ndim != 2:
            raise ValueError("umask must have shape [batch, sequence]")
        batch_size, sequence_length = umask.shape
        if len(conversation_ids) != batch_size:
            raise ValueError("conversation IDs must match the umask batch size")
        count = sequence_length * batch_size
        sides = []
        for _ in ("host", "guest"):
            sampled = random_mask(3, count, self.requested_missing_rate)
            shaped = sampled.reshape(batch_size, sequence_length, 3).transpose(1, 0, 2)
            sides.append(torch.as_tensor(shaped, device=umask.device))
        return sides[0], sides[1]
