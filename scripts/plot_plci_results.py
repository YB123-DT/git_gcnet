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


def _write_scores(path: Path, records: Sequence[ScoreRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "method", "missing_rate", "seed", "value", "source"
        ])
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda item: (item.dataset, item.method, item.missing_rate, item.seed),
        ):
            writer.writerow({
                "dataset": record.dataset,
                "method": record.method,
                "missing_rate": f"{record.missing_rate:.1f}",
                "seed": record.seed,
                "value": f"{record.value:.8f}",
                "source": record.source,
            })


def _mean_and_error(matrix: np.ndarray, error: str) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(matrix, axis=0)
    counts = np.sum(np.isfinite(matrix), axis=0)
    sample_sd = np.asarray([
        np.nanstd(matrix[:, index], ddof=1) if counts[index] >= 2 else 0.0
        for index in range(matrix.shape[1])
    ])
    if error == "std":
        spread = sample_sd
    elif error == "sem":
        spread = sample_sd / np.sqrt(np.maximum(counts, 1))
    elif error == "ci95":
        spread = 1.96 * sample_sd / np.sqrt(np.maximum(counts, 1))
    elif error == "none":
        spread = np.zeros_like(sample_sd)
    else:
        raise ValueError(f"unsupported error mode: {error}")
    return mean, spread


def _metric_label(metric: str) -> str:
    return "Weighted F1 (%)" if metric == "weighted_f1" else "Accuracy (%)"


def _save_figure(figure, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> None:
    for extension in formats:
        kwargs = {"bbox_inches": "tight"}
        if extension.lower() == "png":
            kwargs["dpi"] = dpi
        figure.savefig(out_dir / f"{stem}.{extension}", **kwargs)


def _annotate_heatmap(axis, matrix: np.ndarray, signed: bool = False) -> None:
    finite = matrix[np.isfinite(matrix)]
    threshold = float(np.nanmean(finite)) if finite.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                label = "NA"
                color = "black"
            else:
                label = f"{value:+.2f}" if signed else f"{value:.2f}"
                color = "white" if (not signed and value < threshold) else "black"
            axis.text(column, row, label, ha="center", va="center", fontsize=7, color=color)


def render_figures(
    dataset: str,
    plci_matrix: np.ndarray,
    original_matrix: np.ndarray,
    seeds: Sequence[int],
    missing_rates: Sequence[float],
    metric: str,
    error: str,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    import matplotlib

    if not matplotlib.get_backend().lower().startswith("agg"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(missing_rates, dtype=np.float64)
    ylabel = _metric_label(metric)
    colors = {"Original GCNet": "#4C78A8", "PLCI-JEPA": "#E45756"}
    matrices = {
        "Original GCNet": original_matrix,
        "PLCI-JEPA": plci_matrix,
    }

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for method, matrix in matrices.items():
        mean, spread = _mean_and_error(matrix, error)
        axis.plot(x, mean, marker="o", linewidth=2.2, label=method, color=colors[method])
        if error != "none":
            axis.fill_between(x, mean - spread, mean + spread, color=colors[method], alpha=0.16)
    axis.set(xlabel="Missing rate", ylabel=ylabel, title=f"{dataset}: performance under missing modalities")
    axis.set_xticks(x)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, out_dir, "mean_curve", formats, dpi)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)
    for axis, (method, matrix) in zip(axes, matrices.items()):
        for row, seed in enumerate(seeds):
            axis.plot(x, matrix[row], marker="o", linewidth=1.25, alpha=0.82, label=f"seed {seed}")
        axis.set_title(method)
        axis.set_xlabel("Missing rate")
        axis.set_xticks(x)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False, fontsize=8, ncol=1)
    figure.suptitle(f"{dataset}: individual seed trajectories")
    figure.tight_layout()
    _save_figure(figure, out_dir, "seed_curves", formats, dpi)
    plt.close(figure)

    combined = np.concatenate([original_matrix.ravel(), plci_matrix.ravel()])
    finite = combined[np.isfinite(combined)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharey=True)
    image = None
    for axis, (method, matrix) in zip(axes, matrices.items()):
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(method)
        axis.set_xticks(range(len(x)), [f"{rate:.1f}" for rate in x])
        axis.set_yticks(range(len(seeds)), [str(seed) for seed in seeds])
        axis.set_xlabel("Missing rate")
        _annotate_heatmap(axis, matrix)
    axes[0].set_ylabel("Seed")
    figure.colorbar(image, ax=axes, label=ylabel, shrink=0.86)
    figure.suptitle(f"{dataset}: score matrix")
    figure.subplots_adjust(left=0.08, right=0.91, bottom=0.12, top=0.82, wspace=0.16)
    _save_figure(figure, out_dir, "score_heatmaps", formats, dpi)
    plt.close(figure)

    delta = plci_matrix - original_matrix
    finite_delta = np.abs(delta[np.isfinite(delta)])
    bound = float(finite_delta.max()) if finite_delta.size else 1.0
    bound = max(bound, 1e-9)
    figure, axis = plt.subplots(figsize=(7.5, 4.3))
    image = axis.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    axis.set_xticks(range(len(x)), [f"{rate:.1f}" for rate in x])
    axis.set_yticks(range(len(seeds)), [str(seed) for seed in seeds])
    axis.set(xlabel="Missing rate", ylabel="Seed", title=f"{dataset}: PLCI-JEPA − Original GCNet")
    _annotate_heatmap(axis, delta, signed=True)
    figure.colorbar(image, ax=axis, label="Difference (percentage points)")
    figure.tight_layout()
    _save_figure(figure, out_dir, "delta_heatmap", formats, dpi)
    plt.close(figure)

    delta_mean, delta_spread = _mean_and_error(delta, error)
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.axhline(0.0, color="#444444", linewidth=1.0)
    axis.plot(x, delta_mean, marker="o", linewidth=2.2, color="#7A5195")
    if error != "none":
        axis.fill_between(x, delta_mean - delta_spread, delta_mean + delta_spread, color="#7A5195", alpha=0.18)
    axis.set(xlabel="Missing rate", ylabel="PLCI − Original (percentage points)", title=f"{dataset}: seed-aligned performance difference")
    axis.set_xticks(x)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    _save_figure(figure, out_dir, "delta_curve", formats, dpi)
    plt.close(figure)


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
    parser.add_argument("--error", choices=["std", "sem", "ci95", "none"], default="std")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf"], default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        record for record in original + plci
        if record.seed in args.seeds and record.missing_rate in args.missing_rates
    ]
    _write_scores(args.out_dir / "scores.csv", selected)
    render_figures(
        dataset=args.dataset,
        plci_matrix=build_matrix(plci, args.seeds, args.missing_rates),
        original_matrix=build_matrix(original, args.seeds, args.missing_rates),
        seeds=args.seeds,
        missing_rates=args.missing_rates,
        metric=args.metric,
        error=args.error,
        out_dir=args.out_dir,
        formats=args.formats,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
