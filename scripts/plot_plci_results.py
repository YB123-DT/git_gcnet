#!/usr/bin/env python3
"""Collect and plot PLCI-JEPA versus Original GCNet experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class ScoreRecord:
    dataset: str
    method: str
    missing_rate: float
    seed: int
    value: float
    source: str


def weighted_f1_score(
    labels: np.ndarray,
    predictions: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute multiclass weighted F1 without requiring scikit-learn."""
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    if sample_weight is None:
        weights = np.ones(labels.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weights.shape != labels.shape:
            raise ValueError("sample_weight must match labels")
    total_support = float(weights.sum())
    if total_support <= 0:
        raise ValueError("weighted F1 requires positive total sample weight")

    weighted_sum = 0.0
    for label in np.unique(np.concatenate([labels, predictions])):
        true = labels == label
        predicted = predictions == label
        support = float(weights[true].sum())
        true_positive = float(weights[true & predicted].sum())
        false_positive = float(weights[~true & predicted].sum())
        false_negative = float(weights[true & ~predicted].sum())
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
        weighted_sum += support * f1
    return weighted_sum / total_support


def _accuracy_score(
    labels: np.ndarray,
    predictions: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    weights = (
        np.ones(labels.shape[0], dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    )
    if weights.shape != labels.shape or weights.sum() <= 0:
        raise ValueError("invalid sample weights")
    return float(np.sum(weights * (labels == predictions)) / weights.sum())


def _as_percentage(value: float) -> float:
    return 100.0 * value if abs(value) <= 1.5 else value


def _parse_rate(path: Path) -> float:
    for part in path.parts:
        match = re.fullmatch(r"miss_([0-9]+(?:[p.][0-9]+)?)", part)
        if match:
            return float(match.group(1).replace("p", "."))
    raise ValueError(f"cannot parse missing rate from {path}")


def _parse_seed(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"cannot parse seed from {path}")


def load_plci_records(
    root: Path,
    dataset: str,
    metric: str = "weighted_f1",
) -> list[ScoreRecord]:
    field = "weighted_f1" if metric == "weighted_f1" else "accuracy"
    records: list[ScoreRecord] = []
    for path in sorted(root.glob("miss_*/seed_*/fold_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if field not in entry:
                raise ValueError(f"{path} does not contain {field}")
            records.append(ScoreRecord(
                dataset=dataset,
                method="PLCI-JEPA",
                missing_rate=float(entry.get("missing_rate", _parse_rate(path))),
                seed=int(entry.get("seed", _parse_seed(path))),
                value=_as_percentage(float(entry[field])),
                source=str(path),
            ))
    return records


def _walk_objects(value: object) -> Iterator[dict]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for child in value.flat:
            yield from _walk_objects(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_objects(child)


def _saved_prediction_dict(npz: np.lib.npyio.NpzFile, path: Path) -> dict:
    if "folder_savewhole" not in npz:
        raise ValueError(f"{path} does not contain folder_savewhole")
    candidates = [
        value for value in _walk_objects(npz["folder_savewhole"])
        if {"test_labels", "test_preds"}.issubset(value)
    ]
    if not candidates:
        raise ValueError(f"{path} contains no saved test prediction dictionary")
    return candidates[-1]


def _score_saved_predictions(payload: dict, dataset: str, metric: str) -> float:
    labels = np.asarray(payload["test_labels"]).reshape(-1)
    predictions = np.asarray(payload["test_preds"])
    weights = np.asarray(
        payload.get("test_fmask", np.ones(labels.shape[0])), dtype=np.float64
    ).reshape(-1)
    if weights.shape[0] != labels.shape[0]:
        raise ValueError("test_fmask length does not match test_labels")

    if dataset in {"CMUMOSI", "CMUMOSEI"}:
        predictions = predictions.reshape(-1)
        valid = (weights > 0) & (labels != 0)
        labels = labels[valid] > 0
        predictions = predictions[valid] > 0
        weights = np.ones(labels.shape[0], dtype=np.float64)
    else:
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=-1)
        predictions = predictions.reshape(-1)
        valid = weights > 0
        labels = labels[valid]
        predictions = predictions[valid]
        weights = weights[valid]

    scorer = weighted_f1_score if metric == "weighted_f1" else _accuracy_score
    return 100.0 * scorer(labels, predictions, weights)


def load_original_records(
    root: Path,
    dataset: str,
    metric: str = "weighted_f1",
) -> list[ScoreRecord]:
    grouped: dict[tuple[float, int], list[Path]] = {}
    for path in sorted(root.glob("miss_*/seed_*/**/saved/*.npz")):
        key = (_parse_rate(path), _parse_seed(path))
        grouped.setdefault(key, []).append(path)

    records: list[ScoreRecord] = []
    for (rate, seed), paths in sorted(grouped.items()):
        if len(paths) != 1:
            rendered = "\n".join(str(path) for path in paths)
            raise ValueError(
                f"expected one Original NPZ for rate={rate}, seed={seed}; "
                f"found {len(paths)}:\n{rendered}"
            )
        path = paths[0]
        with np.load(path, allow_pickle=True) as archive:
            payload = _saved_prediction_dict(archive, path)
            value = _score_saved_predictions(payload, dataset, metric)
        records.append(ScoreRecord(
            dataset=dataset,
            method="Original GCNet",
            missing_rate=rate,
            seed=seed,
            value=value,
            source=str(path),
        ))
    return records


def build_matrix(
    records: Sequence[ScoreRecord],
    seeds: Sequence[int],
    missing_rates: Sequence[float],
) -> np.ndarray:
    values = {(record.seed, record.missing_rate): record.value for record in records}
    return np.asarray([
        [values.get((seed, rate), np.nan) for rate in missing_rates]
        for seed in seeds
    ], dtype=np.float64)


def validate_grid(
    records: Sequence[ScoreRecord],
    seeds: Sequence[int],
    missing_rates: Sequence[float],
    method: str,
) -> None:
    counts: dict[tuple[int, float], int] = {}
    for record in records:
        key = (record.seed, record.missing_rate)
        counts[key] = counts.get(key, 0) + 1
    problems = []
    for seed in seeds:
        for rate in missing_rates:
            count = counts.get((seed, rate), 0)
            if count != 1:
                problems.append(f"seed={seed}, rate={rate:.1f}, count={count}")
    if problems:
        raise ValueError(f"incomplete {method} result grid: " + "; ".join(problems))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--plci-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(66, 71)))
    parser.add_argument(
        "--missing-rates", nargs="+", type=float,
        default=[index / 10 for index in range(8)],
    )
    parser.add_argument("--metric", choices=["weighted_f1", "accuracy"], default="weighted_f1")
    parser.add_argument("--no-strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    plci = load_plci_records(args.plci_root, args.dataset, args.metric)
    original = load_original_records(args.original_root, args.dataset, args.metric)
    if not args.no_strict:
        validate_grid(plci, args.seeds, args.missing_rates, "PLCI-JEPA")
        validate_grid(original, args.seeds, args.missing_rates, "Original GCNet")
    raise NotImplementedError("plot rendering is implemented in task 2")


if __name__ == "__main__":
    raise SystemExit(main())
