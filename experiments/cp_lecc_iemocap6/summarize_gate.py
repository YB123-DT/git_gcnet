"""Summarize and gate the locked CP-LECC IEMOCAP-6 experiment."""

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


RATES = (0.5, 0.7)
SEEDS = (66, 67, 68, 69, 70)
EXPECTED_KEYS = {(rate, seed) for rate in RATES for seed in SEEDS}


def _archive_path(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"archive path is not an NPZ file: {path}")
        return path
    if not path.is_dir():
        raise ValueError(f"archive path does not exist: {path}")
    archives = sorted(path.glob("*.npz"))
    if len(archives) != 1:
        raise ValueError(
            f"expected exactly one NPZ archive in {path}, found {len(archives)}"
        )
    return archives[0]


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.size == 1 else value


def _manifest(data: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = _scalar(data["mask_bank_manifest"])
    if not isinstance(manifest, Mapping):
        raise ValueError("mask_bank_manifest is not a mapping")
    return manifest


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


def _concatenate(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value
    parts = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
    return np.concatenate([np.asarray(part) for part in parts], axis=0)


def archive_metrics(path: Path) -> Dict[str, Any]:
    """Load the selected snapshot metrics from one saved experiment archive."""
    archive = _archive_path(path)
    with np.load(archive, allow_pickle=True) as data:
        epoch, snapshot = _selected_snapshot(data)
        labels = _concatenate(snapshot["test_labels"]).reshape(-1)
        scores = _concatenate(snapshot["test_preds"])
        predictions = np.argmax(scores, axis=-1).reshape(-1)
        if labels.shape != predictions.shape:
            raise ValueError(
                f"label/prediction shape mismatch: {labels.shape} != {predictions.shape}"
            )
        counts = np.unique(predictions, return_counts=True)[1]
        manifest_hash = str(_manifest(data)["sha256"])
        parameter_count = int(_scalar(data["parameter_count"]))
    return {
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "accuracy": float(accuracy_score(labels, predictions)),
        "class_coverage": int(len(np.unique(predictions))),
        "dominant_ratio": float(counts.max() / len(predictions)),
        "manifest_hash": manifest_hash,
        "mask_sha256": manifest_hash,
        "parameter_count": parameter_count,
        "epoch": epoch,
    }


def _assert_exact(candidate: Any, original: Any, field: str) -> None:
    if isinstance(candidate, np.ndarray) or isinstance(original, np.ndarray):
        left, right = np.asarray(candidate), np.asarray(original)
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(
                f"{field} mismatch: shape/dtype {left.shape}/{left.dtype} != "
                f"{right.shape}/{right.dtype}"
            )
        if left.dtype == object:
            for index, (left_item, right_item) in enumerate(
                zip(left.flat, right.flat)
            ):
                _assert_exact(left_item, right_item, f"{field}[{index}]")
        elif not np.array_equal(left, right, equal_nan=True):
            raise AssertionError(f"{field} mismatch: arrays differ")
        return
    if isinstance(candidate, Mapping) or isinstance(original, Mapping):
        if not isinstance(candidate, Mapping) or not isinstance(original, Mapping):
            raise AssertionError(f"{field} mismatch: different container types")
        if set(candidate) != set(original):
            raise AssertionError(f"{field} mismatch: keys differ")
        for key in candidate:
            _assert_exact(candidate[key], original[key], f"{field}.{key}")
        return
    if isinstance(candidate, (list, tuple)) or isinstance(original, (list, tuple)):
        if type(candidate) is not type(original) or len(candidate) != len(original):
            raise AssertionError(f"{field} mismatch: sequences differ")
        for index, (left_item, right_item) in enumerate(zip(candidate, original)):
            _assert_exact(left_item, right_item, f"{field}[{index}]")
        return
    try:
        equal = bool(candidate == original)
    except (TypeError, ValueError):
        equal = False
    if not equal:
        raise AssertionError(f"{field} mismatch: {candidate!r} != {original!r}")


def assert_complete_archive_equal(candidate: Path, original: Path) -> None:
    """Assert locked complete-data fields are recursively and exactly equal."""
    candidate_path = _archive_path(candidate)
    original_path = _archive_path(original)
    with np.load(candidate_path, allow_pickle=True) as left, np.load(
        original_path, allow_pickle=True
    ) as right:
        left_epoch, left_snapshot = _selected_snapshot(left)
        right_epoch, right_snapshot = _selected_snapshot(right)
        _assert_exact(left_epoch, right_epoch, "best epoch")
        _assert_exact(
            left["folder_losswhole"], right["folder_losswhole"], "folder_losswhole"
        )
        for field in ("test_labels", "test_preds", "test_hiddens", "test_fmask"):
            _assert_exact(left_snapshot[field], right_snapshot[field], field)
        _assert_exact(
            _manifest(left)["sha256"], _manifest(right)["sha256"], "mask sha"
        )


def _indexed(rows: Iterable[Mapping[str, Any]], name: str) -> Dict[Tuple[float, int], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = (float(row["rate"]), int(row["seed"]))
        if key in result:
            raise ValueError(f"duplicate {name} key: rate={key[0]}, seed={key[1]}")
        result[key] = row
    missing = EXPECTED_KEYS - set(result)
    extra = set(result) - EXPECTED_KEYS
    if missing:
        raise ValueError(f"missing {name} keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"unexpected {name} keys: {sorted(extra)}")
    return result


def paired_gate(
    candidate_rows: Sequence[Mapping[str, Any]],
    original_rows: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate the preregistered paired promotion gate at full precision."""
    candidate = _indexed(candidate_rows, "candidate")
    original = _indexed(original_rows, "original")
    full = _indexed(full_rows, "full")
    task_rows = []
    for rate, seed in sorted(EXPECTED_KEYS):
        candidate_row, original_row, full_row = (
            candidate[(rate, seed)], original[(rate, seed)], full[(rate, seed)]
        )
        task_rows.append(
            {
                "rate": rate,
                "seed": seed,
                "candidate_weighted_f1": float(candidate_row["weighted_f1"]),
                "original_weighted_f1": float(original_row["weighted_f1"]),
                "full_weighted_f1": float(full_row["weighted_f1"]),
                "delta_original": float(candidate_row["weighted_f1"])
                - float(original_row["weighted_f1"]),
                "delta_full": float(candidate_row["weighted_f1"])
                - float(full_row["weighted_f1"]),
            }
        )
    rate_means = {
        str(rate): float(
            np.mean([row["delta_original"] for row in task_rows if row["rate"] == rate])
        )
        for rate in RATES
    }
    seed_deltas = {
        str(seed): {
            "original": float(
                np.mean([row["delta_original"] for row in task_rows if row["seed"] == seed])
            ),
            "full": float(
                np.mean([row["delta_full"] for row in task_rows if row["seed"] == seed])
            ),
        }
        for seed in SEEDS
    }
    wins = sum(delta["original"] > 0 for delta in seed_deltas.values())
    coverage_dominant = [
        {
            "rate": rate,
            "seed": seed,
            "class_coverage": int(candidate[(rate, seed)]["class_coverage"]),
            "dominant_ratio": float(candidate[(rate, seed)]["dominant_ratio"]),
        }
        for rate, seed in sorted(EXPECTED_KEYS)
    ]
    hash_matches = all(
        candidate[key]["manifest_hash"]
        == original[key]["manifest_hash"]
        == full[key]["manifest_hash"]
        for key in EXPECTED_KEYS
    )
    conditions = {
        "rate_0.5_nonnegative_vs_original": rate_means["0.5"] >= 0,
        "rate_0.7_nonnegative_vs_original": rate_means["0.7"] >= 0,
        "seed_mean_delta_original_at_least_0.005": float(
            np.mean([value["original"] for value in seed_deltas.values()])
        )
        >= 0.005,
        "at_least_four_positive_seed_deltas": wins >= 4,
        "candidate_seed_mean_strictly_greater_full": all(
            value["full"] > 0 for value in seed_deltas.values()
        ),
        "all_candidate_coverage_six": all(
            row["class_coverage"] == 6 for row in coverage_dominant
        ),
        "all_pair_mask_hashes_match": hash_matches,
    }
    return {
        "promote": all(conditions.values()),
        "task_rows": task_rows,
        "rate_means": rate_means,
        "seed_deltas": seed_deltas,
        "wins": wins,
        "coverage_dominant": coverage_dominant,
        "conditions": conditions,
    }


def _rate_tag(rate: float) -> str:
    return f"{rate:.1f}".replace(".", "p")


def _collect(root: Path) -> list:
    rows = []
    for rate, seed in sorted(EXPECTED_KEYS):
        saved = root / f"miss_{_rate_tag(rate)}" / f"seed_{seed}" / "fold_5" / "saved"
        row = archive_metrics(saved)
        row.update(rate=rate, seed=seed)
        rows.append(row)
    return rows


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--complete-candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    complete_original = (
        args.original_root / "miss_0p0" / "seed_66" / "fold_5" / "saved"
    )
    assert_complete_archive_equal(args.complete_candidate, complete_original)
    evidence = paired_gate(
        _collect(args.candidate_root),
        _collect(args.original_root),
        _collect(args.full_root),
    )
    _write_json_atomic(args.output_json, evidence)
    print("PROMOTE" if evidence["promote"] else "REJECT")


if __name__ == "__main__":
    main()
