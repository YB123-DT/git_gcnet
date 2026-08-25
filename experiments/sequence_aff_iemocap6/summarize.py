"""Validate and summarize the locked IEMOCAP-6 Sequence AFF experiment.

The training archives contain pickled ``argparse.Namespace`` objects.  This
module therefore accepts only complete, locally generated job directories and
validates their immutable runner artifacts before opening the single NPZ.
"""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats

from experiments.mpfilm_iemocap6.run_locked_ab import (
    ARM_TO_BRANCH_FUSION,
    Job,
    LOCKED_TRAINING,
    _completed_log,
    _job_payload,
)


ARMS = ("original", "sequence_aff")
RATES = tuple(index / 10 for index in range(8))
SEEDS = (66, 67, 68, 69, 70)
STORED_PARAMETER_COUNT = 36_419_816
SELECTED_PARAMETER_COUNTS = {
    "original": 34_140_166,
    "sequence_aff": 34_393_416,
}


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.size == 1 else value


def _concatenate(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value
    parts = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
    return np.concatenate([np.asarray(part) for part in parts], axis=0)


def _selected_snapshot(data: Mapping[str, Any]) -> Tuple[int, Mapping[str, Any]]:
    saves = data["folder_savewhole"]
    if len(saves) != 1 or len(saves[0]) < 2:
        raise ValueError("folder_savewhole must contain one fold and a selected snapshot")
    epoch = int(_scalar(saves[0][0]))
    snapshot = saves[0][-1]
    if isinstance(snapshot, np.ndarray) and snapshot.shape == ():
        snapshot = snapshot.item()
    if not isinstance(snapshot, Mapping):
        raise ValueError("selected folder_savewhole snapshot is not a mapping")
    return epoch, snapshot


def _classification_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> Tuple[float, float]:
    accuracy = float(np.mean(labels == predictions))
    scores = []
    supports = []
    for label in np.unique(labels):
        true_positive = float(np.sum((labels == label) & (predictions == label)))
        false_positive = float(np.sum((labels != label) & (predictions == label)))
        false_negative = float(np.sum((labels == label) & (predictions != label)))
        denominator = 2.0 * true_positive + false_positive + false_negative
        score = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
        scores.append(score)
        supports.append(float(np.sum(labels == label)))
    weighted_f1 = float(np.average(np.asarray(scores), weights=np.asarray(supports)))
    return weighted_f1, accuracy


def _locked_paths(command: Sequence[str]) -> Tuple[Path, Path, Path, Path]:
    if not isinstance(command, list) or len(command) < 3:
        raise ValueError("command payload is not a complete command list")
    try:
        python = Path(command[0])
        train_script = Path(command[2])
        repository = train_script.parents[1]
        data_root = Path(command[command.index("--data-root") + 1])
        mask_root = Path(command[command.index("--mask-bank-root") + 1])
    except (ValueError, IndexError):
        raise ValueError("command list is missing locked paths") from None
    if train_script != repository / "gcnet" / "train_gcnet.py":
        raise ValueError("command training script mismatch: {}".format(train_script))
    return python, repository, data_root, mask_root


def _validated_archive(
    fold_directory: Path, arm: str, rate: float, seed: int
) -> Path:
    fold_directory = Path(fold_directory)
    if arm not in ARMS:
        raise ValueError("unknown arm: {!r}".format(arm))
    if not fold_directory.is_dir() or fold_directory.is_symlink():
        raise ValueError("job directory is missing or unsafe: {}".format(fold_directory))
    lock = fold_directory / ".active.lock"
    if lock.exists():
        raise ValueError("active or stale lock present: {}".format(lock))
    required = ("command.json", "status.json", "train.log", "saved")
    missing = [name for name in required if not (fold_directory / name).exists()]
    if missing:
        raise ValueError("missing job artifacts in {}: {}".format(fold_directory, missing))
    saved = fold_directory / "saved"
    if not saved.is_dir() or saved.is_symlink():
        raise ValueError("saved directory is missing or unsafe: {}".format(saved))
    archives = sorted(saved.glob("*.npz"))
    if len(archives) != 1:
        raise ValueError(
            "expected exactly one NPZ archive in {}, found {}".format(saved, len(archives))
        )
    archive = archives[0]
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("archive is missing or unsafe: {}".format(archive))

    payload = json.loads((fold_directory / "command.json").read_text(encoding="utf-8"))
    command = payload.get("command")
    python, repository, data_root, mask_root = _locked_paths(command)
    job = Job("formal", arm, rate, seed, fold_directory)
    expected_payload = _job_payload(
        job, str(payload.get("gpu")), python, repository, data_root, mask_root
    )
    if payload != expected_payload:
        raise ValueError("command.json mismatch for {}".format(fold_directory))
    status = json.loads((fold_directory / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "success" or status.get("return_code") != 0:
        raise ValueError("status return_code is not successful for {}".format(fold_directory))
    if not _completed_log(fold_directory / "train.log"):
        raise ValueError("train.log does not contain exactly 100 epoch records and completion markers")
    return archive


def _archive_metrics(archive: Path) -> Dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(archive), flags)
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            with np.load(handle, allow_pickle=True) as data:
                epoch, snapshot = _selected_snapshot(data)
                labels = _concatenate(snapshot["test_labels"]).reshape(-1)
                scores = _concatenate(snapshot["test_preds"])
                predictions = np.argmax(scores, axis=-1).reshape(-1)
                if labels.shape != predictions.shape or labels.size == 0:
                    raise ValueError(
                        "label/prediction shape mismatch: {} != {}".format(
                            labels.shape, predictions.shape
                        )
                    )
                counts = np.unique(predictions, return_counts=True)[1]
                weighted_f1, accuracy = _classification_metrics(labels, predictions)
                args = _scalar(data["args"])
                manifest = _scalar(data["mask_bank_manifest"])
                if not isinstance(manifest, Mapping):
                    raise ValueError("mask_bank_manifest is not a mapping")
                return {
                    "weighted_f1": weighted_f1,
                    "accuracy": accuracy,
                    "class_coverage": int(len(np.unique(predictions))),
                    "dominant_ratio": float(counts.max() / predictions.size),
                    "epoch": epoch,
                    "manifest_hash": str(manifest["sha256"]),
                    "requested_missing_rate": float(manifest["requested_missing_rate"]),
                    "manifest_seed": int(manifest["seed"]),
                    "parameter_count": int(_scalar(data["parameter_count"])),
                    "selected_path_parameter_count": int(
                        _scalar(data["selected_path_parameter_count"])
                    ),
                    "fold_numbers": [int(value) for value in np.asarray(data["fold_numbers"]).flat],
                    "smoke_only": bool(_scalar(data["smoke_only"])),
                    "args": args,
                }
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    return actual == expected


def collect_job(fold_directory: Path, arm: str, rate: float, seed: int) -> Dict[str, Any]:
    """Validate one completed job, then load metrics from its trusted NPZ."""
    archive = _validated_archive(fold_directory, arm, rate, seed)
    row = _archive_metrics(archive)
    args = row.pop("args")
    expected_args = dict(LOCKED_TRAINING)
    expected_args.pop("fold")
    expected_args.update(
        graph_conv_variant="original",
        branch_fusion=ARM_TO_BRANCH_FUSION[arm],
        seed=seed,
        mask_seed=seed,
        mask_type="constant-{:.1f}".format(rate),
        fold_index=5,
    )
    for field, expected in expected_args.items():
        actual = getattr(args, field, None)
        if not _matches(actual, expected):
            raise ValueError(
                "archive provenance {} mismatch: {!r} != {!r}".format(
                    field, actual, expected
                )
            )
    expected_fields = {
        "fold_numbers": [5],
        "smoke_only": False,
        "requested_missing_rate": rate,
        "manifest_seed": seed,
        "parameter_count": STORED_PARAMETER_COUNT,
        "selected_path_parameter_count": SELECTED_PARAMETER_COUNTS[arm],
    }
    for field, expected in expected_fields.items():
        if not _matches(row[field], expected):
            raise ValueError(
                "archive provenance {} mismatch: {!r} != {!r}".format(
                    field, row[field], expected
                )
            )
    row.update(
        arm=arm,
        rate=float(rate),
        seed=int(seed),
        graph_conv_variant="original",
        branch_fusion=ARM_TO_BRANCH_FUSION[arm],
    )
    return row


def collect_grid(
    root: Path, rates: Sequence[float] = RATES, seeds: Sequence[int] = SEEDS
) -> Dict[str, list]:
    rows = {arm: [] for arm in ARMS}
    for rate in rates:
        for seed in seeds:
            pair = []
            for arm in ARMS:
                fold = (
                    Path(root)
                    / arm
                    / "miss_{}".format("{:.1f}".format(rate).replace(".", "p"))
                    / "seed_{}".format(seed)
                    / "fold_5"
                )
                row = collect_job(fold, arm, float(rate), int(seed))
                rows[arm].append(row)
                pair.append(row)
            if pair[0]["manifest_hash"] != pair[1]["manifest_hash"]:
                raise ValueError(
                    "paired mask manifest hash mismatch at rate={}, seed={}".format(rate, seed)
                )
    return rows


def _index(
    rows: Iterable[Mapping[str, Any]], rates: Sequence[float], seeds: Sequence[int]
) -> Dict[Tuple[float, int], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = (float(row["rate"]), int(row["seed"]))
        if key in result:
            raise ValueError("duplicate result key: rate={}, seed={}".format(*key))
        result[key] = row
    expected = {(float(rate), int(seed)) for rate in rates for seed in seeds}
    if set(result) != expected:
        raise ValueError("result grid mismatch: expected {}, found {}".format(sorted(expected), sorted(result)))
    return result


def _finite_number(value: Any) -> Any:
    number = float(value)
    return number if math.isfinite(number) else None


def _paired_statistics(deltas: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0:
        raise ValueError("paired statistics require at least one delta")
    result = {
        "mean_delta": float(np.mean(values)),
        "sd_delta": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "wins": int(np.sum(values > 0)),
        "ties": int(np.sum(values == 0)),
        "losses": int(np.sum(values < 0)),
        "n": int(values.size),
    }
    if values.size < 2:
        t_statistic = t_pvalue = None
    elif np.all(values == values[0]):
        if values[0] == 0:
            t_statistic, t_pvalue = 0.0, 1.0
        else:
            t_statistic = math.copysign(math.inf, values[0])
            t_pvalue = 0.0
    else:
        test = stats.ttest_1samp(values, popmean=0.0)
        t_statistic, t_pvalue = test.statistic, test.pvalue
    if np.all(values == 0):
        w_statistic, w_pvalue = 0.0, 1.0
    else:
        test = stats.wilcoxon(values)
        w_statistic, w_pvalue = test.statistic, test.pvalue
    result["paired_t_test"] = {
        "statistic": _finite_number(t_statistic) if t_statistic is not None else None,
        "pvalue": _finite_number(t_pvalue) if t_pvalue is not None else None,
    }
    result["wilcoxon"] = {
        "statistic": _finite_number(w_statistic),
        "pvalue": _finite_number(w_pvalue),
    }
    return result


def paired_summary(
    original_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    rates: Sequence[float] = RATES,
    seeds: Sequence[int] = SEEDS,
) -> Dict[str, Any]:
    original = _index(original_rows, rates, seeds)
    candidate = _index(candidate_rows, rates, seeds)
    tasks = []
    rate_summaries = {}
    seed_deltas = {int(seed): [] for seed in seeds}
    for rate in rates:
        deltas = []
        original_scores = []
        candidate_scores = []
        for seed in seeds:
            key = (float(rate), int(seed))
            left, right = original[key], candidate[key]
            if left["manifest_hash"] != right["manifest_hash"]:
                raise ValueError("paired mask manifest hash mismatch at rate={}, seed={}".format(rate, seed))
            delta = float(right["weighted_f1"]) - float(left["weighted_f1"])
            deltas.append(delta)
            seed_deltas[int(seed)].append(delta)
            original_scores.append(float(left["weighted_f1"]))
            candidate_scores.append(float(right["weighted_f1"]))
            tasks.append(
                {
                    "rate": float(rate),
                    "seed": int(seed),
                    "original": dict(left),
                    "sequence_aff": dict(right),
                    "delta_weighted_f1": delta,
                }
            )
        stats_row = _paired_statistics(deltas)
        stats_row.update(
            original_mean=float(np.mean(original_scores)),
            sequence_aff_mean=float(np.mean(candidate_scores)),
        )
        rate_summaries[str(float(rate))] = stats_row
    seed_macro_deltas = [float(np.mean(seed_deltas[int(seed)])) for seed in seeds]
    macro = _paired_statistics(seed_macro_deltas)
    macro.update(
        original_mean=float(
            np.mean([float(row["weighted_f1"]) for row in original_rows])
        ),
        sequence_aff_mean=float(
            np.mean([float(row["weighted_f1"]) for row in candidate_rows])
        ),
        unit="seed mean across rates",
    )
    return {
        "experiment": "Sequence AFF vs addition, locked IEMOCAPSix fold 5",
        "rates": rate_summaries,
        "macro": macro,
        "seed_macro_deltas": {
            str(seed): seed_macro_deltas[index] for index, seed in enumerate(seeds)
        },
        "tasks": tasks,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Sequence AFF 配对实验汇总",
        "",
        "本报告比较锁定的 IEMOCAPSix fold 5 Original/addition 与 Sequence AFF；每个差值均按相同缺失掩码和随机种子配对。",
        "",
        "| 缺失率 | Original F1 | Sequence AFF F1 | 配对差值均值 | 差值 SD | 胜/平/负 | t 检验 p | Wilcoxon p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rate, row in summary["rates"].items():
        lines.append(
            "| {rate} | {original_mean:.6f} | {sequence_aff_mean:.6f} | {mean_delta:+.6f} | {sd_delta:.6f} | {wins}/{ties}/{losses} | {tp} | {wp} |".format(
                rate=rate,
                tp="NA" if row["paired_t_test"]["pvalue"] is None else "{:.6g}".format(row["paired_t_test"]["pvalue"]),
                wp="NA" if row["wilcoxon"]["pvalue"] is None else "{:.6g}".format(row["wilcoxon"]["pvalue"]),
                **row
            )
        )
    macro = summary["macro"]
    lines.extend(
        [
            "",
            "## 八档宏平均",
            "",
            "Original 加权 F1 为 {original_mean:.6f}，Sequence AFF 为 {sequence_aff_mean:.6f}；按种子先跨缺失率平均后的配对差值为 {mean_delta:+.6f} ± {sd_delta:.6f}（胜/平/负：{wins}/{ties}/{losses}）。".format(**macro),
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent), delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize trusted, locally generated Sequence AFF experiment jobs."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    rows = collect_grid(args.root)
    summary = paired_summary(rows["original"], rows["sequence_aff"])
    _write_text_atomic(
        args.output_json, json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_text_atomic(args.output_markdown, render_markdown(summary))


if __name__ == "__main__":
    main()
