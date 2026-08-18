from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics_from_saved_fold(saved_fold: dict) -> dict[str, float | int]:
    labels = np.concatenate(saved_fold["test_labels"])
    logits = np.concatenate(saved_fold["test_preds"])
    predictions = np.argmax(logits, axis=-1)
    return {
        "sample_count": int(labels.size),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def summarize_npz(path: Path) -> dict:
    archive = np.load(path, allow_pickle=True)
    fold_savewhole = archive["folder_savewhole"]
    fold_metrics = []
    for fold_index, fold_snapshots in enumerate(fold_savewhole):
        metrics = classification_metrics_from_saved_fold(fold_snapshots[-1])
        metrics["fold"] = fold_index + 1
        metrics["best_epoch"] = int(fold_snapshots[0]) + 1
        fold_metrics.append(metrics)
    summary = {}
    for metric_name in ("weighted_f1", "macro_f1", "accuracy"):
        values = np.asarray([fold[metric_name] for fold in fold_metrics], dtype=float)
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return {"result_file": str(path), "folds": fold_metrics, "summary": summary}


def _single_result_file(rate_directory: Path) -> Path:
    candidates = sorted(rate_directory.rglob("*.npz"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one result archive in {rate_directory}, found {len(candidates)}"
        )
    return candidates[0]


def summarize_sweep(root: Path) -> dict:
    experiment_roots = {
        "original": root / "experiments" / "original_missing_sweep_seed66_20260818",
        "jepa": root / "experiments" / "modality_jepa_seed66_20260818",
    }
    result = {"seed": 66, "methods": {}}
    for method, experiment_root in experiment_roots.items():
        method_results = {}
        for rate_index in range(8):
            rate = rate_index / 10
            rate_name = f"miss_{rate:.1f}".replace(".", "p")
            rate_directory = experiment_root / rate_name
            status_path = rate_directory / "status.json"
            if not status_path.exists():
                method_results[f"{rate:.1f}"] = {"status": "missing"}
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("returncode") != 0:
                method_results[f"{rate:.1f}"] = {"status": "failed", **status}
                continue
            metrics = summarize_npz(_single_result_file(rate_directory))
            metrics["status"] = "complete"
            metrics["runtime"] = status
            if method == "jepa":
                diagnostics_path = rate_directory / "saved" / "fold_metrics.json"
                if diagnostics_path.exists():
                    metrics["jepa_folds"] = json.loads(
                        diagnostics_path.read_text(encoding="utf-8")
                    )
            method_results[f"{rate:.1f}"] = metrics
        result["methods"][method] = method_results
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "experiments" / "missing_sweep_seed66_summary.json"
    summary = summarize_sweep(root)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
