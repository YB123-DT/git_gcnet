"""Frozen copy of the mask schedule used by the retained MELD checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from gcnet_modality_jepa.mask_schedule import ConversationMask


ALGORITHM = "gcnet-legacy-fixed-v1"


def _payload(**values: object) -> bytes:
    return json.dumps(
        values, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _observed_target(length: int, missing_rate: float) -> int:
    if missing_rate == 0.0:
        return 3 * length
    if 1.0 - missing_rate <= 1.0 / 3.0:
        return length
    return min(3 * length, max(length, round(3 * length * (1.0 - missing_rate))))


@dataclass(frozen=True)
class LegacyFixedMaskSchedule:
    dataset: str
    split: str
    fold: int | str
    requested_missing_rate: float
    mask_seed: int
    freeze_evaluation: bool = True

    def __post_init__(self) -> None:
        if not self.dataset or not self.split:
            raise ValueError("dataset and split must be non-empty")
        rate = float(self.requested_missing_rate)
        if not math.isfinite(rate) or rate < 0.0 or rate > 0.7:
            raise ValueError("requested_missing_rate must be between 0 and 0.7")
        if isinstance(self.mask_seed, bool) or not isinstance(self.mask_seed, int):
            raise TypeError("mask_seed must be an integer")
        object.__setattr__(self, "requested_missing_rate", rate)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            _payload(
                algorithm=ALGORITHM,
                dataset=self.dataset,
                split=self.split,
                fold=self.fold,
                rate=format(self.requested_missing_rate, ".17g"),
                seed=self.mask_seed,
                freeze_evaluation=self.freeze_evaluation,
            )
        ).hexdigest()

    def generate(
        self,
        conversation_id: str,
        length: int,
        side: str,
        epoch: int = 0,
        valid_length: int | None = None,
    ) -> ConversationMask:
        if not conversation_id or not side:
            raise ValueError("conversation_id and side must be non-empty")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError("length must be positive")
        if valid_length is None:
            valid_length = length
        if not isinstance(valid_length, int) or not 1 <= valid_length <= length:
            raise ValueError("valid_length must be between one and length")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be non-negative")

        effective_epoch = (
            0
            if self.freeze_evaluation
            and self.split.lower() in {"validation", "val", "test"}
            else epoch
        )
        key = _payload(
            algorithm=ALGORITHM,
            dataset=self.dataset,
            split=self.split.lower(),
            fold=self.fold,
            conversation_id=conversation_id,
            epoch=effective_epoch,
            rate=format(self.requested_missing_rate, ".17g"),
            seed=self.mask_seed,
        )
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big")
        )

        availability = np.zeros((length, 3), dtype=np.uint8)
        if self.requested_missing_rate == 0.0:
            availability[:valid_length] = 1
        else:
            mandatory = rng.integers(0, 3, size=valid_length)
            availability[np.arange(valid_length), mandatory] = 1
            target = _observed_target(valid_length, self.requested_missing_rate)
            extra = target - valid_length
            if extra:
                zero_rows, zero_modalities = np.nonzero(
                    availability[:valid_length] == 0
                )
                chosen = rng.choice(len(zero_rows), size=extra, replace=False)
                availability[zero_rows[chosen], zero_modalities[chosen]] = 1

        valid = np.zeros(length, dtype=bool)
        valid[:valid_length] = True
        realized = 1.0 - float(availability[:valid_length].mean())
        digest = hashlib.sha256(key + b"\0" + availability.tobytes()).hexdigest()
        return ConversationMask(
            availability=availability,
            valid_utterance_mask=valid,
            requested_missing_rate=self.requested_missing_rate,
            realized_missing_rate=realized,
            epoch=effective_epoch,
            schedule_hash=digest,
        )
