"""Deterministic utterance-indexed banks using GCNet's masking rule."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch


GENERATOR_VERSION = "gcnet-random-mask-v1-fixed-bank"


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


def _bank_paths(root: Path, missing_rate: float, seed: int) -> Tuple[Path, Path]:
    rate_tag = f"{missing_rate:.1f}".replace(".", "p")
    stem = f"mask_rate_{rate_tag}_seed_{int(seed)}"
    return root / f"{stem}.npz", root / f"{stem}.json"


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
