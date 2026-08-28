"""Deterministic balanced scheduling for one-checkpoint mixed-rate training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


MISSING_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


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
