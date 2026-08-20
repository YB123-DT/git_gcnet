"""Conversation-keyed deterministic modality availability schedules."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np


_ALGORITHM_VERSION = "conversation-mask-v2"
_EVALUATION_SPLITS = frozenset(("validation", "test"))
_SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def _stable_payload(values: Dict[str, Any]) -> bytes:
    return json.dumps(
        values,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _rate_key(rate: float) -> str:
    return format(rate, ".17g")


@dataclass(frozen=True)
class ConversationMask:
    """One generated availability matrix and its audit metadata."""

    availability: np.ndarray
    valid_utterance_mask: np.ndarray
    requested_missing_rate: float
    realized_missing_rate: float
    epoch: int
    schedule_hash: str


class ConversationMaskSchedule:
    """Generate deterministic masks without consuming any global RNG stream."""

    def __init__(
        self,
        dataset: str,
        split: str,
        fold: Union[int, str],
        requested_missing_rate: float,
        mask_seed: int,
        freeze_evaluation: bool = True,
    ) -> None:
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("dataset must be a non-empty string")
        if not isinstance(split, str) or not split:
            raise ValueError("split must be a non-empty string")
        try:
            canonical_split = _SPLIT_ALIASES[split.strip().lower()]
        except KeyError:
            raise ValueError(
                "split must be train, validation (or val), or test"
            )
        if isinstance(mask_seed, bool) or not isinstance(mask_seed, int):
            raise ValueError("mask_seed must be an integer")
        if not isinstance(freeze_evaluation, bool):
            raise TypeError("freeze_evaluation must be Boolean")

        rate = float(requested_missing_rate)
        if not math.isfinite(rate) or rate < 0.0 or rate > 0.7:
            raise ValueError("requested_missing_rate must be between 0 and 0.7")

        self.dataset = dataset
        self.split = canonical_split
        self.fold = fold
        self.requested_missing_rate = rate
        self.mask_seed = mask_seed
        self.freeze_evaluation = freeze_evaluation
        self.config_hash = hashlib.sha256(
            _stable_payload(
                {
                    "algorithm": _ALGORITHM_VERSION,
                    "dataset": dataset,
                    "fold": fold,
                    "freeze_evaluation": freeze_evaluation,
                    "mask_seed": mask_seed,
                    "requested_missing_rate": _rate_key(rate),
                    "split": canonical_split,
                }
            )
        ).hexdigest()

    def generate(
        self,
        conversation_id: str,
        length: int,
        side: str,
        epoch: int = 0,
        valid_length: Optional[int] = None,
    ) -> ConversationMask:
        """Generate a ``[length, 3]`` availability matrix for a conversation.

        ``valid_length`` can be shorter than ``length`` for padded storage. Such
        rows are zeroed and identified separately by ``valid_utterance_mask``.
        """
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be a non-empty string")
        if not isinstance(side, str) or not side:
            raise ValueError("side must be a non-empty string")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError("length must be positive")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be non-negative")
        if valid_length is None:
            valid_length = length
        if (
            isinstance(valid_length, bool)
            or not isinstance(valid_length, int)
            or valid_length < 1
            or valid_length > length
        ):
            raise ValueError("valid_length must be between one and length")

        effective_epoch = (
            0
            if self.freeze_evaluation and self.split in _EVALUATION_SPLITS
            else epoch
        )
        key = {
            "dataset": self.dataset,
            "split": self.split,
            "fold": self.fold,
            "epoch": effective_epoch,
            "conversation_id": conversation_id,
            "side": side,
            "requested_missing_rate": _rate_key(self.requested_missing_rate),
            "mask_seed": self.mask_seed,
        }
        key_payload = _stable_payload(key)
        seed = int.from_bytes(
            hashlib.sha256(key_payload).digest()[:8], byteorder="big"
        )
        rng = np.random.default_rng(seed)

        availability = np.zeros((length, 3), dtype=np.uint8)
        if valid_length:
            real_availability = (
                rng.random((valid_length, 3)) >= self.requested_missing_rate
            )
            empty_rows = np.flatnonzero(~real_availability.any(axis=1))
            if empty_rows.size:
                retained_modalities = rng.integers(0, 3, size=empty_rows.size)
                real_availability[empty_rows, retained_modalities] = True
            availability[:valid_length] = real_availability.astype(
                np.uint8, copy=False
            )

        valid_utterance_mask = np.zeros(length, dtype=bool)
        valid_utterance_mask[:valid_length] = True
        if valid_length:
            realized_missing_rate = 1.0 - float(
                availability[:valid_length].mean()
            )
        else:
            realized_missing_rate = 0.0

        digest = hashlib.sha256()
        digest.update(key_payload)
        digest.update(b"\0")
        digest.update(str(length).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(valid_length).encode("ascii"))
        digest.update(b"\0")
        digest.update(availability.tobytes(order="C"))

        return ConversationMask(
            availability=availability,
            valid_utterance_mask=valid_utterance_mask,
            requested_missing_rate=self.requested_missing_rate,
            realized_missing_rate=realized_missing_rate,
            epoch=effective_epoch,
            schedule_hash=digest.hexdigest(),
        )
