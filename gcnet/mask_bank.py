"""Deterministic utterance-indexed banks using GCNet's masking rule."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch


GENERATOR_VERSION = "gcnet-random-mask-v1-fixed-bank"
STAGE_GENERATOR_VERSION = "gcnet-random-mask-v2-stage-aware"


def _legacy_random_mask(
    view_num: int, input_len: int, missing_rate: float, seed: int
) -> np.ndarray:
    """Run the released GCNet rule with an isolated legacy NumPy RNG."""

    if view_num < 1:
        raise ValueError("view_num must be positive")
    if input_len < 0:
        raise ValueError("input_len must be non-negative")
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be in [0,1]")
    if input_len == 0:
        return np.empty((0, view_num), dtype=np.uint8)

    rng = np.random.RandomState(int(seed))
    one_rate = 1.0 - float(missing_rate)
    if one_rate <= 1.0 / view_num:
        selected = rng.randint(0, view_num, size=input_len)
        return np.eye(view_num, dtype=np.uint8)[selected]
    if one_rate == 1.0:
        return np.ones((input_len, view_num), dtype=np.uint8)

    all_data_len = max(input_len, 32)
    error = 1.0
    matrix = None
    for _ in range(10000):
        selected = rng.randint(0, view_num, size=all_data_len)
        view_preserve = np.eye(view_num, dtype=np.uint8)[selected]
        one_num = view_num * all_data_len * one_rate - all_data_len
        ratio = one_num / (view_num * all_data_len)
        matrix_iter = (
            rng.randint(0, 100, size=(all_data_len, view_num))
            < int(ratio * 100)
        ).astype(np.uint8)
        overlap = np.sum((matrix_iter + view_preserve > 1).astype(np.uint8))
        one_num_iter = one_num / (1.0 - overlap / one_num)
        ratio = one_num_iter / (view_num * all_data_len)
        matrix_iter = (
            rng.randint(0, 100, size=(all_data_len, view_num))
            < int(ratio * 100)
        ).astype(np.uint8)
        matrix = (matrix_iter + view_preserve > 0).astype(np.uint8)
        realized_one_rate = np.sum(matrix) / (view_num * all_data_len)
        error = abs(one_rate - realized_one_rate)
        if error < 0.005:
            break
    if matrix is None or error >= 0.005:
        raise RuntimeError("GCNet random-mask rule did not converge")
    return matrix[:input_len]


def build_mask_bank(
    video_ids: Mapping[str, Sequence[str]], missing_rate: float, seed: int
) -> Dict[str, np.ndarray]:
    """Build one immutable pattern array per conversation."""

    ordered_vids = sorted(str(vid) for vid in video_ids)
    lengths = [len(video_ids[vid]) for vid in ordered_vids]
    matrix = _legacy_random_mask(3, sum(lengths), missing_rate, seed)
    bank = {}
    offset = 0
    for vid, length in zip(ordered_vids, lengths):
        bank[vid] = matrix[offset : offset + length].copy()
        offset += length
    return bank


def mask_bank_sha256(bank: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for vid in sorted(bank):
        array = np.asarray(bank[vid], dtype=np.uint8)
        digest.update(vid.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _derived_stage_seed(seed: int, stage: str, epoch: int = -1) -> int:
    payload = f"{int(seed)}\0{stage}\0{int(epoch)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def build_stage_mask_bundle(
    video_ids: Mapping[str, Sequence[str]],
    missing_rate: float,
    seed: int,
    epochs: int,
) -> dict:
    """Build epoch-varying train masks and fixed evaluation masks."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    train = tuple(
        build_mask_bank(
            video_ids,
            missing_rate,
            _derived_stage_seed(seed, "train", epoch),
        )
        for epoch in range(epochs)
    )
    return {
        "train": train,
        "validation": build_mask_bank(
            video_ids,
            missing_rate,
            _derived_stage_seed(seed, "validation"),
        ),
        "test": build_mask_bank(
            video_ids,
            missing_rate,
            _derived_stage_seed(seed, "test"),
        ),
    }


def select_stage_mask(bundle: Mapping[str, object], stage: str, epoch=None):
    """Select one bank without relying on mutable call order."""

    if stage == "train":
        if epoch is None:
            raise ValueError("train stage requires an epoch")
        train = bundle["train"]
        if not 0 <= int(epoch) < len(train):
            raise ValueError("train epoch is outside the mask bundle")
        return train[int(epoch)]
    if stage not in ("validation", "test"):
        raise ValueError(f"unknown mask stage: {stage}")
    if epoch is not None:
        raise ValueError(f"{stage} stage does not accept an epoch")
    return bundle[stage]


def stage_mask_bundle_sha256(bundle: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for epoch, bank in enumerate(bundle["train"]):
        digest.update(f"train\0{epoch}\0".encode("utf-8"))
        digest.update(mask_bank_sha256(bank).encode("ascii"))
    for stage in ("validation", "test"):
        digest.update(f"{stage}\0".encode("utf-8"))
        digest.update(mask_bank_sha256(bundle[stage]).encode("ascii"))
    return digest.hexdigest()


def _bank_paths(root: Path, missing_rate: float, seed: int) -> Tuple[Path, Path]:
    rate_tag = f"{missing_rate:.1f}".replace(".", "p")
    stem = f"mask_rate_{rate_tag}_seed_{int(seed)}"
    return root / f"{stem}.npz", root / f"{stem}.json"


def _stage_bundle_paths(
    root: Path, missing_rate: float, seed: int, epochs: int
) -> Tuple[Path, Path]:
    rate_tag = f"{missing_rate:.1f}".replace(".", "p")
    stem = f"mask_stage_v2_rate_{rate_tag}_seed_{int(seed)}_epochs_{int(epochs)}"
    return root / f"{stem}.npz", root / f"{stem}.json"


def _flatten_stage_bundle(bundle: Mapping[str, object]) -> Dict[str, np.ndarray]:
    arrays = {}
    for epoch, bank in enumerate(bundle["train"]):
        for vid, array in bank.items():
            arrays[f"train_{epoch:03d}__{vid}"] = np.asarray(array, dtype=np.uint8)
    for stage in ("validation", "test"):
        for vid, array in bundle[stage].items():
            arrays[f"{stage}__{vid}"] = np.asarray(array, dtype=np.uint8)
    return arrays


def _unflatten_stage_bundle(arrays: Mapping[str, np.ndarray], epochs: int) -> dict:
    train = [dict() for _ in range(epochs)]
    validation, test = {}, {}
    for name, array in arrays.items():
        if name.startswith("train_"):
            prefix, vid = name.split("__", 1)
            epoch = int(prefix[len("train_") :])
            if not 0 <= epoch < epochs:
                raise ValueError("saved train mask epoch is outside the manifest")
            train[epoch][vid] = np.asarray(array, dtype=np.uint8)
        elif name.startswith("validation__"):
            validation[name.split("__", 1)[1]] = np.asarray(array, dtype=np.uint8)
        elif name.startswith("test__"):
            test[name.split("__", 1)[1]] = np.asarray(array, dtype=np.uint8)
        else:
            raise ValueError(f"unknown saved stage mask key: {name}")
    return {"train": tuple(train), "validation": validation, "test": test}


def load_or_create_stage_mask_bundle(
    root: Path,
    video_ids: Mapping[str, Sequence[str]],
    missing_rate: float,
    seed: int,
    epochs: int,
) -> Tuple[dict, dict]:
    """Persist and verify a stage-aware bundle shared by model arms."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    bank_path, manifest_path = _stage_bundle_paths(
        root, missing_rate, seed, epochs
    )
    expected_lengths = {
        str(vid): len(video_ids[vid]) for vid in sorted(video_ids)
    }
    if bank_path.exists() != manifest_path.exists():
        raise RuntimeError(
            "stage mask bundle and manifest must either both exist or both be absent"
        )
    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].astype(np.uint8) for name in archive.files}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generator") != STAGE_GENERATOR_VERSION:
            raise ValueError("saved stage mask bundle uses another generator")
        if manifest.get("epochs") != int(epochs):
            raise ValueError("saved stage mask bundle has another epoch count")
        if manifest.get("video_lengths") != expected_lengths:
            raise ValueError("saved stage mask bundle does not match the dataset")
        bundle = _unflatten_stage_bundle(arrays, epochs)
        if manifest.get("sha256") != stage_mask_bundle_sha256(bundle):
            raise ValueError("saved stage mask bundle hash mismatch")
        return bundle, manifest

    bundle = build_stage_mask_bundle(video_ids, missing_rate, seed, epochs)
    all_banks = list(bundle["train"]) + [bundle["validation"], bundle["test"]]
    rows = np.concatenate(
        [np.concatenate(list(bank.values()), axis=0) for bank in all_banks],
        axis=0,
    )
    manifest = {
        "generator": STAGE_GENERATOR_VERSION,
        "requested_missing_rate": float(missing_rate),
        "realized_missing_rate": 1.0 - float(rows.mean()) if rows.size else 0.0,
        "seed": int(seed),
        "epochs": int(epochs),
        "video_lengths": expected_lengths,
        "train_sha256": [mask_bank_sha256(bank) for bank in bundle["train"]],
        "validation_sha256": mask_bank_sha256(bundle["validation"]),
        "test_sha256": mask_bank_sha256(bundle["test"]),
        "sha256": stage_mask_bundle_sha256(bundle),
    }
    with bank_path.open("wb") as handle:
        np.savez_compressed(handle, **_flatten_stage_bundle(bundle))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle, manifest


def load_or_create_mask_bank(
    root: Path,
    video_ids: Mapping[str, Sequence[str]],
    missing_rate: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """Persist and verify a bank shared by every model arm."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    bank_path, manifest_path = _bank_paths(root, missing_rate, seed)
    expected_lengths = {
        str(vid): len(video_ids[vid]) for vid in sorted(video_ids)
    }
    if bank_path.exists() != manifest_path.exists():
        raise RuntimeError("mask bank and manifest must either both exist or both be absent")
    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as archive:
            bank = {name: archive[name].astype(np.uint8) for name in archive.files}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("video_lengths") != expected_lengths:
            raise ValueError("saved mask bank does not match the dataset")
        if manifest.get("sha256") != mask_bank_sha256(bank):
            raise ValueError("saved mask bank hash mismatch")
        return bank, manifest

    bank = build_mask_bank(video_ids, missing_rate, seed)
    rows = np.concatenate(list(bank.values()), axis=0)
    realized_missing_rate = 1.0 - float(rows.mean()) if rows.size else 0.0
    manifest = {
        "generator": GENERATOR_VERSION,
        "requested_missing_rate": float(missing_rate),
        "realized_missing_rate": realized_missing_rate,
        "seed": int(seed),
        "video_lengths": expected_lengths,
        "sha256": mask_bank_sha256(bank),
    }
    with bank_path.open("wb") as handle:
        np.savez_compressed(handle, **bank)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bank, manifest


def batch_mask_from_bank(
    bank: Mapping[str, np.ndarray],
    vidnames: Sequence[str],
    max_length: int,
) -> torch.Tensor:
    """Recreate a padded `[T,B,3]` mask for one loader batch."""

    output = torch.ones(max_length, len(vidnames), 3, dtype=torch.uint8)
    for batch_index, vid in enumerate(vidnames):
        if vid not in bank:
            raise KeyError(f"conversation is absent from mask bank: {vid}")
        rows = torch.as_tensor(bank[vid], dtype=torch.uint8)
        if rows.dim() != 2 or rows.size(1) != 3 or rows.size(0) > max_length:
            raise ValueError(f"invalid saved mask shape for {vid}: {tuple(rows.shape)}")
        output[: rows.size(0), batch_index] = rows
    return output
