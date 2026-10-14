"""Causal feature preparation and label-free donor matching for MELD probes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DonorMatch:
    indices: np.ndarray
    included: np.ndarray
    total: int
    matched: int


def _check_rows(data: dict, keys: tuple[str, ...]) -> int:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    n = len(np.asarray(data[keys[0]]))
    if any(len(np.asarray(data[key])) != n for key in keys):
        raise ValueError("row fields have inconsistent lengths")
    return n


def strict_past_summaries(data: dict) -> dict[str, np.ndarray]:
    """Compute all/same-speaker Local means using only earlier utterances."""
    n = _check_rows(data, ("conversation_ids", "utterance_indices", "speaker", "local"))
    local = np.asarray(data["local"], dtype=np.float32)
    if local.ndim != 2 or local.shape[0] != n or not np.isfinite(local).all():
        raise ValueError("local must be finite [row,feature]")
    conversations = np.asarray(data["conversation_ids"]).astype(str)
    positions = np.asarray(data["utterance_indices"])
    speakers = np.asarray(data["speaker"])
    all_mean = np.zeros_like(local)
    same_mean = np.zeros_like(local)
    all_count = np.zeros(n, dtype=np.int64)
    same_count = np.zeros(n, dtype=np.int64)
    start = 0
    while start < n:
        conversation = conversations[start]
        stop = start
        while stop < n and conversations[stop] == conversation:
            stop += 1
        expected = np.arange(stop - start)
        if not np.array_equal(positions[start:stop], expected):
            raise ValueError("rows must follow contiguous temporal order within each conversation")
        total = np.zeros(local.shape[1], dtype=np.float64)
        speaker_total: dict[object, np.ndarray] = {}
        speaker_counts: dict[object, int] = {}
        for row in range(start, stop):
            count = row - start
            all_count[row] = count
            if count:
                all_mean[row] = total / count
            speaker = speakers[row].item() if hasattr(speakers[row], "item") else speakers[row]
            count_same = speaker_counts.get(speaker, 0)
            same_count[row] = count_same
            if count_same:
                same_mean[row] = speaker_total[speaker] / count_same
            total += local[row]
            speaker_total.setdefault(speaker, np.zeros(local.shape[1], dtype=np.float64))
            speaker_total[speaker] += local[row]
            speaker_counts[speaker] = count_same + 1
        start = stop
    return {
        "all_mean": all_mean,
        "same_speaker_mean": same_mean,
        "all_count": all_count,
        "same_speaker_count": same_count,
        "all_present": (all_count > 0).astype(np.int64),
        "same_speaker_present": (same_count > 0).astype(np.int64),
    }


def mask_forward_memory(
    base: np.ndarray, gaps: np.ndarray, availability: np.ndarray, *, forward_dim: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base, dtype=np.float32)
    gaps = np.asarray(gaps, dtype=np.float32)
    availability = np.asarray(availability)
    n = len(availability)
    if base.shape[0] != n or gaps.shape[:2] != (n, 3) or availability.shape != (n, 3):
        raise ValueError("invalid Base/Gap/availability shapes")
    if base.shape[-1] < forward_dim or gaps.shape[-1] < forward_dim:
        raise ValueError("context tensors are narrower than forward_dim")
    if not np.isin(availability, (0, 1)).all():
        raise ValueError("availability must be binary")
    forward_base = base[:, :forward_dim].copy()
    active = (1 - availability).astype(bool)
    forward_gaps = np.where(active[..., None], gaps[:, :, :forward_dim], 0).astype(np.float32)
    return forward_base, forward_gaps


def match_donors(data: dict, seed: int) -> DonorMatch:
    """Match within a split without labels; never donate from the same conversation."""
    n = _check_rows(
        data,
        ("conversation_ids", "availability", "all_present", "same_speaker_present"),
    )
    conversation = np.asarray(data["conversation_ids"]).astype(str)
    availability = np.asarray(data["availability"])
    all_present = np.asarray(data["all_present"])
    same_present = np.asarray(data["same_speaker_present"])
    if availability.shape != (n, 3):
        raise ValueError("availability must have shape [row,3]")
    groups: dict[tuple, list[int]] = {}
    for row in range(n):
        key = (*availability[row].astype(int).tolist(), int(all_present[row]), int(same_present[row]))
        groups.setdefault(key, []).append(row)
    rng = np.random.default_rng(seed)
    indices = np.full(n, -1, dtype=np.int64)
    for row in range(n):
        key = (*availability[row].astype(int).tolist(), int(all_present[row]), int(same_present[row]))
        candidates = np.asarray(
            [candidate for candidate in groups[key] if conversation[candidate] != conversation[row]],
            dtype=np.int64,
        )
        if len(candidates):
            indices[row] = int(candidates[rng.integers(len(candidates))])
    included = indices >= 0
    return DonorMatch(indices=indices, included=included, total=n, matched=int(included.sum()))


def validate_cache(data: dict) -> int:
    keys = (
        "conversation_ids", "utterance_indices", "speaker", "labels", "availability",
        "local", "base", "gap", "prediction", "all_mean", "same_speaker_mean",
        "all_count", "same_speaker_count", "all_present", "same_speaker_present",
    )
    n = _check_rows(data, keys)
    expected = {
        "availability": (n, 3), "local": (n, 256), "base": (n, 512),
        "gap": (n, 3, 512), "prediction": (n, 7), "all_mean": (n, 256),
        "same_speaker_mean": (n, 256),
    }
    for key, shape in expected.items():
        values = np.asarray(data[key])
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"invalid cache field {key}: {values.shape}")
    if not np.isin(data["availability"], (0, 1)).all():
        raise ValueError("invalid availability")
    if np.count_nonzero(np.asarray(data["gap"])[np.asarray(data["availability"], dtype=bool)]):
        raise ValueError("inactive Gap leakage")
    identities = list(zip(np.asarray(data["conversation_ids"]).astype(str), data["utterance_indices"]))
    if len(set(identities)) != n:
        raise ValueError("duplicate utterance identity")
    return n
