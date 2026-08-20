"""Deterministic, leakage-free conversation split construction."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple


_IEMOCAP_SESSION = re.compile(
    r"Ses0([1-5])[FM]_(?:impro[0-9]+|script[0-9]+_[0-9]+)"
)


@dataclass(frozen=True)
class SplitIndices:
    """Immutable train/validation/test partition with a stable content hash."""

    train: Tuple[int, ...]
    validation: Tuple[int, ...]
    test: Tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            indices = tuple(getattr(self, name))
            if not indices:
                raise ValueError("{} split must be nonempty".format(name))
            if any(not isinstance(index, numbers.Integral) or isinstance(index, bool)
                   for index in indices):
                raise TypeError("{} indices must be integers".format(name))
            object.__setattr__(self, name, tuple(int(index) for index in indices))

        groups = (set(self.train), set(self.validation), set(self.test))
        if len(groups[0]) != len(self.train):
            raise ValueError("duplicate train indices")
        if len(groups[1]) != len(self.validation):
            raise ValueError("duplicate validation indices")
        if len(groups[2]) != len(self.test):
            raise ValueError("duplicate test indices")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("train/validation/test indices overlap")

        all_indices = groups[0] | groups[1] | groups[2]
        if min(all_indices) < 0 or all_indices != set(range(len(all_indices))):
            raise ValueError("split indices must cover every index exactly once")

    @property
    def split_hash(self) -> str:
        """Return SHA-256 over a canonical JSON representation of the partition.

        Index order is intentionally significant because it records the exact
        dataset-to-split mapping consumed by samplers, not only set membership.
        """
        payload = {
            "test": list(self.test),
            "train": list(self.train),
            "validation": list(self.validation),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _unique_vids(vids: Sequence[Any]) -> Tuple[Any, ...]:
    normalized = tuple(vids)
    if not normalized:
        raise ValueError("vids must be nonempty")
    try:
        unique = set(normalized)
    except TypeError as error:
        raise TypeError("video IDs must be hashable") from error
    if len(unique) != len(normalized):
        raise ValueError("vids contains duplicate conversation IDs")
    return normalized


def _labels_for_conversation(labels: Any, vid: str) -> Tuple[Any, ...]:
    if labels is None:
        raise ValueError("missing labels for {!r}".format(vid))
    if isinstance(labels, (str, bytes)):
        normalized = (labels,)
    else:
        try:
            normalized = tuple(labels)
        except TypeError:
            normalized = (labels,)
    if not normalized:
        raise ValueError("missing labels for {!r}".format(vid))
    try:
        for label in normalized:
            hash(label)
    except TypeError as error:
        raise TypeError("labels for {!r} must be hashable".format(vid)) from error
    return normalized


def _parse_iemocap_session(vid: Any) -> int:
    if not isinstance(vid, str):
        raise ValueError("malformed IEMOCAP conversation ID: {!r}".format(vid))
    match = _IEMOCAP_SESSION.fullmatch(vid)
    if match is None:
        raise ValueError("malformed IEMOCAP conversation ID: {!r}".format(vid))
    return int(match.group(1))


def build_iemocap_loso_split(
    vids: Sequence[str],
    labels_by_vid: Mapping[str, Any],
    test_session: int,
    validation_fraction: float,
    seed: int,
) -> SplitIndices:
    """Build one IEMOCAP leave-one-session-out split.

    The input must contain all five sessions. The requested session is assigned
    wholly to test. From the other sessions, ``round(conversation_count *
    validation_fraction)`` conversations are chosen for validation. Selection is
    greedy: at each step it adds the conversation whose accumulated validation-
    label proportions have the smallest L1 distance from the complete non-test
    label distribution. Seeded ordering resolves equal scores, so the algorithm
    is reproducible without consulting global RNG state. The remaining non-test
    conversations form the training split.
    """
    normalized_vids = _unique_vids(vids)
    if (not isinstance(test_session, numbers.Integral)
            or isinstance(test_session, bool)
            or not 1 <= int(test_session) <= 5):
        raise ValueError("test_session must be an integer from 1 through 5")
    test_session = int(test_session)
    if (not isinstance(validation_fraction, numbers.Real)
            or isinstance(validation_fraction, bool)
            or not math.isfinite(float(validation_fraction))
            or not 0.0 < float(validation_fraction) < 1.0):
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if not isinstance(seed, numbers.Integral) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    sessions = tuple(_parse_iemocap_session(vid) for vid in normalized_vids)
    if set(sessions) != set(range(1, 6)):
        raise ValueError("IEMOCAP vids must contain exactly sessions 1 through 5")
    if test_session not in sessions:
        raise ValueError("unknown test_session {} for supplied vids".format(test_session))

    labels = {}
    for vid in normalized_vids:
        if vid not in labels_by_vid:
            raise ValueError("missing labels for {!r}".format(vid))
        labels[vid] = _labels_for_conversation(labels_by_vid[vid], vid)

    test_indices = tuple(
        index for index, session in enumerate(sessions) if session == test_session
    )
    candidate_indices = tuple(
        index for index, session in enumerate(sessions) if session != test_session
    )
    validation_count = int(round(len(candidate_indices) * float(validation_fraction)))
    if validation_count < 1 or validation_count >= len(candidate_indices):
        raise ValueError(
            "not enough non-test conversations for nonempty train and validation splits"
        )

    pool_counts = Counter()
    for index in candidate_indices:
        pool_counts.update(labels[normalized_vids[index]])
    pool_total = sum(pool_counts.values())
    pool_proportions = {
        label: count / float(pool_total) for label, count in pool_counts.items()
    }

    tie_order = sorted(candidate_indices, key=lambda index: normalized_vids[index])
    random.Random(int(seed)).shuffle(tie_order)
    tie_rank = {index: rank for rank, index in enumerate(tie_order)}
    selected = []
    selected_counts = Counter()
    remaining = set(candidate_indices)

    def score(index: int) -> Tuple[float, int]:
        proposed = selected_counts + Counter(labels[normalized_vids[index]])
        proposed_total = float(sum(proposed.values()))
        distribution_error = sum(
            abs(proposed[label] / proposed_total - pool_proportions[label])
            for label in pool_counts
        )
        return distribution_error, tie_rank[index]

    while len(selected) < validation_count:
        chosen = min(remaining, key=score)
        selected.append(chosen)
        selected_counts.update(labels[normalized_vids[chosen]])
        remaining.remove(chosen)

    validation_set = set(selected)
    train_indices = tuple(
        index for index in candidate_indices if index not in validation_set
    )
    validation_indices = tuple(
        index for index in candidate_indices if index in validation_set
    )
    return SplitIndices(train_indices, validation_indices, test_indices)


def _official_ids(name: str, values: Iterable[Any]) -> Tuple[Any, ...]:
    normalized = tuple(values)
    try:
        unique = set(normalized)
    except TypeError as error:
        raise TypeError("{} video IDs must be hashable".format(name)) from error
    if len(unique) != len(normalized):
        raise ValueError("{} contains duplicate video IDs".format(name))
    return normalized


def build_official_split(
    vids: Sequence[Any],
    train_vids: Iterable[Any],
    validation_vids: Iterable[Any],
    test_vids: Iterable[Any],
) -> SplitIndices:
    """Map exact official MOSI/MOSEI video memberships to dataset indices."""
    normalized_vids = _unique_vids(vids)
    groups = {
        "train": set(_official_ids("train_vids", train_vids)),
        "validation": set(_official_ids("validation_vids", validation_vids)),
        "test": set(_official_ids("test_vids", test_vids)),
    }
    if (groups["train"] & groups["validation"]
            or groups["train"] & groups["test"]
            or groups["validation"] & groups["test"]):
        raise ValueError("official train/validation/test video IDs overlap")

    known = set(normalized_vids)
    assigned = groups["train"] | groups["validation"] | groups["test"]
    unknown = assigned - known
    if unknown:
        raise ValueError("official splits contain unknown video IDs: {!r}".format(unknown))
    missing = known - assigned
    if missing:
        raise ValueError("official splits are missing video IDs: {!r}".format(missing))

    return SplitIndices(
        train=tuple(index for index, vid in enumerate(normalized_vids)
                    if vid in groups["train"]),
        validation=tuple(index for index, vid in enumerate(normalized_vids)
                         if vid in groups["validation"]),
        test=tuple(index for index, vid in enumerate(normalized_vids)
                   if vid in groups["test"]),
    )
