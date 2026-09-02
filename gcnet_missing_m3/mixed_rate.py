"""Deterministic balanced scheduling for one-checkpoint mixed-rate training."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Mapping, Sequence


MISSING_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
STRATIFIED_RATE_ALGORITHM = "conversation-rate-stratified-v1"


@dataclass(frozen=True)
class StratifiedRateAssignment:
    rates: tuple[float, ...]
    assignment_hash: str
    algorithm: str = STRATIFIED_RATE_ALGORITHM


def stratified_rates_for_batch(
    rates: Sequence[float],
    *,
    master_seed: int,
    dataset: str,
    fold: int | str,
    epoch: int,
    batch_index: int,
    epoch_size: int,
    conversations_seen: int,
    conversation_ids: Sequence[str],
) -> StratifiedRateAssignment:
    normalized_rates = tuple(float(value) for value in rates)
    if not normalized_rates or len(set(normalized_rates)) != len(
        normalized_rates
    ):
        raise ValueError("rates must be a non-empty unique sequence")

    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset must be a non-empty string")
    if (
        isinstance(fold, bool)
        or not isinstance(fold, (int, str))
        or (isinstance(fold, str) and not fold)
    ):
        raise ValueError("fold must be a non-empty integer or string")

    normalized_ids = tuple(conversation_ids)
    if not normalized_ids or any(
        not isinstance(value, str) or not value for value in normalized_ids
    ):
        raise ValueError("conversation_ids must contain non-empty strings")

    for name, value in (
        ("epoch", epoch),
        ("batch_index", batch_index),
        ("conversations_seen", conversations_seen),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(epoch_size, bool) or not isinstance(epoch_size, int):
        raise ValueError("epoch_size must be an integer")

    batch_size = len(normalized_ids)
    if epoch_size < batch_size:
        raise ValueError("epoch_size must be at least the batch size")
    if conversations_seen + batch_size > epoch_size:
        raise ValueError("batch extends beyond epoch_size")

    quotient, remainder = divmod(batch_size, len(normalized_rates))
    stream_offset = epoch * epoch_size + conversations_seen
    assigned_rates = list(normalized_rates) * quotient
    assigned_rates.extend(
        normalized_rates[(stream_offset + index) % len(normalized_rates)]
        for index in range(remainder)
    )

    payload = json.dumps(
        {
            "algorithm": STRATIFIED_RATE_ALGORITHM,
            "batch": batch_index,
            "dataset": dataset,
            "epoch": epoch,
            "epoch_size": epoch_size,
            "fold": fold,
            "ids": normalized_ids,
            "rates": [format(value, ".17g") for value in normalized_rates],
            "seed": master_seed,
            "seen": conversations_seen,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    local_seed = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="big"
    )
    random.Random(local_seed).shuffle(assigned_rates)

    assigned_payload = json.dumps(
        [format(value, ".17g") for value in assigned_rates],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    assignment_hash = hashlib.sha256(
        payload + b"\0" + assigned_payload
    ).hexdigest()
    return StratifiedRateAssignment(tuple(assigned_rates), assignment_hash)


@dataclass(frozen=True)
class BalancedBatchRateSchedule:
    rates: Sequence[float] = MISSING_RATES

    def __post_init__(self) -> None:
        normalized = tuple(float(value) for value in self.rates)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("rates must be a non-empty unique sequence")
        if any(value < 0.0 or value > 0.7 for value in normalized):
            raise ValueError("rates must be between 0.0 and 0.7")
        object.__setattr__(self, "rates", normalized)

    def rate_for(self, epoch: int, batch_index: int) -> float:
        if epoch < 0 or batch_index < 0:
            raise ValueError("epoch and batch_index must be non-negative")
        return self.rates[(int(epoch) + int(batch_index)) % len(self.rates)]


def mean_validation_weighted_f1(
    metrics_by_rate: Mapping[float, Mapping[str, float]],
) -> float:
    if set(metrics_by_rate) != set(MISSING_RATES):
        raise ValueError("validation metrics must contain all eight missing rates")
    return sum(
        float(metrics_by_rate[rate]["weighted_f1"]) for rate in MISSING_RATES
    ) / len(MISSING_RATES)


def select_best_epoch(history: Sequence[Mapping[str, object]]) -> int:
    if not history:
        raise ValueError("history must not be empty")
    best = max(
        history,
        key=lambda record: mean_validation_weighted_f1(record["validation"]),
    )
    return int(best["epoch"])
