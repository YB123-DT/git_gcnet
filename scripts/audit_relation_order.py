#!/usr/bin/env python3
"""Audit Temporal/Speaker relation order before implementing a new backbone.

The audit is deliberately training-free.  It uses frozen utterance features
and the existing GCNet window topology to answer two questions:

1. Do typed Temporal and Speaker operators measurably fail to commute?
2. Does adding ordered first-order compositions improve a small ridge probe?

Only train and validation splits are read.  Test data is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from gcnet_modality_jepa.mask_schedule import ConversationMaskSchedule
from gcnet_modality_jepa.protocol import SeedBundle
from gcnet_modality_jepa.train_gcnet import (
    build_primary_mask_tensors,
    generate_inputs,
    get_loaders,
)


RATES = (0.0, 0.5, 0.7)
PROBE_SEEDS = (66, 67, 68)
DEFAULT_W_PAST = 2
DEFAULT_W_FUTURE = 2
RANDOM_PROJECTION_DIM = 128
RANDOM_PROJECTION_SEED = 20260905
SHUFFLE_REPEATS = 8


def _stable_int(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _edge_records(
    length: int,
    speakers: Sequence[int],
    window_past: int,
    window_future: int,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    """Return ``(source, target, temporal_type, speaker_type)`` records.

    The orientation follows ``batch_graphify``: ``edge_index[0]`` is the
    sender and ``edge_index[1]`` is the receiver.  Thus a source utterance at
    an earlier time has temporal type ``past`` when it sends to a later target.
    """

    if len(speakers) != length:
        raise ValueError("speaker sequence length mismatch")
    records: list[tuple[int, int, int, int]] = []
    for target in range(length):
        lo = 0 if window_past == -1 else max(0, target - window_past)
        hi = (
            length
            if window_future == -1
            else min(length, target + window_future + 1)
        )
        for source in range(lo, hi):
            if source < target:
                temporal_type = 0  # past: source precedes target
            elif source == target:
                temporal_type = 1  # self/now
            else:
                temporal_type = 2  # future: source follows target
            speaker_type = int(speakers[target]) * 2 + int(speakers[source])
            records.append((source, target, temporal_type, speaker_type))
    return records, (3, max(1, (max(speakers) + 1) ** 2))


def _adjacency(
    length: int,
    records: Iterable[tuple[int, int, int, int]],
    temporal_type: int | None = None,
    speaker_type: int | None = None,
) -> np.ndarray:
    """Build a receiver-row, sender-column row-normalized adjacency."""

    matrix = np.zeros((length, length), dtype=np.float64)
    for source, target, temporal, speaker in records:
        if temporal_type is not None and temporal != temporal_type:
            continue
        if speaker_type is not None and speaker != speaker_type:
            continue
        matrix[target, source] += 1.0
    degree = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, degree, out=np.zeros_like(matrix), where=degree > 0)


def _commutator_score(left: np.ndarray, right: np.ndarray) -> float | None:
    product_lr = left @ right
    product_rl = right @ left
    denominator = np.linalg.norm(product_lr, ord="fro") + np.linalg.norm(
        product_rl, ord="fro"
    )
    if denominator <= 1e-12:
        return None
    return float(np.linalg.norm(product_lr - product_rl, ord="fro") / denominator)


def _weighted_operator(
    length: int,
    records: Iterable[tuple[int, int, int, int]],
    family: str,
    num_speaker_relations: int,
) -> np.ndarray:
    """Build a deterministic typed relation operator for the ridge probe.

    A plain untyped adjacency would be identical for Temporal and Speaker in
    this repository because both families use the same window support.  The
    probe therefore keeps relation semantics through fixed centered weights:
    past/self/future receive -1/0/+1, while speaker relations receive evenly
    spaced weights.  Rows are normalized by absolute weight, so this is a
    diagnostic operator rather than a trainable model.
    """

    if family == "temporal":
        weights = np.asarray((-1.0, 0.0, 1.0), dtype=np.float64)
    elif family == "speaker":
        if num_speaker_relations == 1:
            weights = np.ones(1, dtype=np.float64)
        else:
            weights = np.linspace(-1.0, 1.0, num_speaker_relations)
    else:
        raise ValueError("family must be temporal or speaker")
    matrix = np.zeros((length, length), dtype=np.float64)
    for source, target, temporal, speaker in records:
        relation = temporal if family == "temporal" else speaker
        if relation >= len(weights):
            continue
        matrix[target, source] += weights[relation]
    degree = np.abs(matrix).sum(axis=1, keepdims=True)
    return np.divide(matrix, degree, out=np.zeros_like(matrix), where=degree > 0)


def _conversation_commutators(
    speakers: Sequence[int],
    window_past: int,
    window_future: int,
    shuffle_rng: np.random.Generator | None = None,
) -> dict[str, object]:
    length = len(speakers)
    records, (num_temporal, num_speaker) = _edge_records(
        length, speakers, window_past, window_future
    )
    temporal_labels = [record[2] for record in records]
    speaker_labels = [record[3] for record in records]
    if shuffle_rng is not None:
        temporal_labels = shuffle_rng.permutation(temporal_labels).tolist()
        speaker_labels = shuffle_rng.permutation(speaker_labels).tolist()
        records = [
            (source, target, temporal, speaker)
            for (source, target, _, _), temporal, speaker in zip(
                records, temporal_labels, speaker_labels
            )
        ]

    temporal = {
        relation: _adjacency(
            length, records, temporal_type=relation
        )
        for relation in range(num_temporal)
    }
    speaker = {
        relation: _adjacency(length, records, speaker_type=relation)
        for relation in range(num_speaker)
    }
    pair_scores: dict[str, float] = {}
    for temporal_relation, temporal_matrix in temporal.items():
        for speaker_relation, speaker_matrix in speaker.items():
            score = _commutator_score(temporal_matrix, speaker_matrix)
            if score is not None:
                pair_scores[f"T{temporal_relation}-S{speaker_relation}"] = score

    weighted_temporal = _weighted_operator(
        length, records, "temporal", num_speaker
    )
    weighted_speaker = _weighted_operator(
        length, records, "speaker", num_speaker
    )
    return {
        "length": length,
        "edge_count": len(records),
        "temporal_relation_counts": {
            str(relation): sum(1 for value in temporal_labels if value == relation)
            for relation in range(num_temporal)
        },
        "speaker_relation_counts": {
            str(relation): sum(1 for value in speaker_labels if value == relation)
            for relation in range(num_speaker)
        },
        "pair_scores": pair_scores,
        "pair_mean": (
            float(np.mean(list(pair_scores.values()))) if pair_scores else None
        ),
        "pair_median": (
            float(np.median(list(pair_scores.values()))) if pair_scores else None
        ),
        "aggregate_score": _commutator_score(weighted_temporal, weighted_speaker),
    }


def _flatten_conversation_features(
    features: np.ndarray,
    labels: np.ndarray,
    speakers: Sequence[int],
    window_past: int,
    window_future: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = features.shape[0]
    records, _ = _edge_records(length, speakers, window_past, window_future)
    speaker_relation_count = max(record[3] for record in records) + 1
    temporal = _weighted_operator(
        length, records, "temporal", speaker_relation_count
    )
    speaker = _weighted_operator(
        length, records, "speaker", speaker_relation_count
    )
    x = features.astype(np.float64, copy=False)
    first_order_t = temporal @ x
    first_order_s = speaker @ x
    ordered_st = speaker @ first_order_t
    ordered_ts = temporal @ first_order_s
    first = np.concatenate([x, first_order_t, first_order_s], axis=1)
    ordered = np.concatenate([first, ordered_st, ordered_ts], axis=1)
    return first, ordered, labels.astype(np.float64, copy=False), x


def _project_features(
    features: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    centered = centered / np.maximum(scale, 1e-6)
    return centered @ projection


def _prepare_dataset_view(
    data: Sequence[object],
    schedule: ConversationMaskSchedule,
    dimensions: tuple[int, int, int],
    structure_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], list[dict[str, object]]]:
    host_availability, guest_availability = build_primary_mask_tensors(
        schedule,
        conversation_ids=data[-1],
        umask=data[7],
        epoch=0,
    )
    full = generate_inputs(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6]
    )[0]
    availability = generate_inputs(
        host_availability[..., 0:1],
        host_availability[..., 1:2],
        host_availability[..., 2:3],
        guest_availability[..., 0:1],
        guest_availability[..., 1:2],
        guest_availability[..., 2:3],
        data[6],
    )[0].to(dtype=full.dtype)
    expanded = torch.repeat_interleave(
        availability,
        torch.tensor(dimensions, device=availability.device),
        dim=-1,
    )
    incomplete = full * expanded
    lengths = [int(value) for value in data[7].sum(dim=1).tolist()]
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    speakers: list[list[int]] = []
    structures: list[dict[str, object]] = []
    for batch_index, length in enumerate(lengths):
        node_features = incomplete[:length, batch_index].detach().cpu().numpy()
        node_labels = data[8][batch_index, :length].detach().cpu().numpy()
        node_speakers = [
            int(value) for value in data[6][batch_index, :length].detach().cpu().tolist()
        ]
        features.append(node_features)
        labels.append(node_labels)
        speakers.append(node_speakers)
        structure = _conversation_commutators(
            node_speakers,
            DEFAULT_W_PAST,
            DEFAULT_W_FUTURE,
        )
        shuffle_scores = []
        for repeat in range(SHUFFLE_REPEATS):
            shuffle_rng = np.random.default_rng(
                _stable_int(structure_seed, data[-1][batch_index], repeat)
            )
            shuffled = _conversation_commutators(
                node_speakers,
                DEFAULT_W_PAST,
                DEFAULT_W_FUTURE,
                shuffle_rng=shuffle_rng,
            )
            if shuffled["pair_mean"] is not None:
                shuffle_scores.append(float(shuffled["pair_mean"]))
        structure["shuffle_pair_mean"] = (
            float(np.mean(shuffle_scores)) if shuffle_scores else None
        )
        structure["shuffle_repeats"] = SHUFFLE_REPEATS
        structures.append(structure)
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        speakers,
        structures,
    )


def _collect_split(
    loader,
    dataset: str,
    split: str,
    fold: int,
    seed: int,
    rate: float,
    dimensions: tuple[int, int, int],
    projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    schedule = ConversationMaskSchedule(
        dataset=dataset,
        split=split,
        fold=fold,
        requested_missing_rate=rate,
        mask_seed=SeedBundle(seed).derive("missing_mask"),
        freeze_evaluation=True,
    )
    first_parts: list[np.ndarray] = []
    ordered_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    structures: list[dict[str, object]] = []
    for data in loader:
        raw_features, raw_labels, speakers, batch_structures = _prepare_dataset_view(
            data,
            schedule,
            dimensions,
            SeedBundle(seed).derive(f"relation_order:{split}:{rate}"),
        )
        # _prepare_dataset_view returns conversation structure, while probe
        # features must be transformed conversation by conversation so that
        # no edge crosses a batch boundary.
        offset = 0
        for structure, node_speakers in zip(batch_structures, speakers):
            length = int(structure["length"])
            node_features = raw_features[offset : offset + length]
            node_labels = raw_labels[offset : offset + length]
            projected = _project_features(node_features, projection)
            first, ordered, labels, _ = _flatten_conversation_features(
                projected,
                node_labels,
                node_speakers,
                DEFAULT_W_PAST,
                DEFAULT_W_FUTURE,
            )
            first_parts.append(first)
            ordered_parts.append(ordered)
            labels_parts.append(labels)
            structures.append(structure)
            offset += length
        if offset != len(raw_features):
            raise RuntimeError("batch feature offset mismatch")
    del dataset, fold, seed, rate, dimensions
    return (
        np.concatenate(first_parts, axis=0),
        np.concatenate(ordered_parts, axis=0),
        np.concatenate(labels_parts, axis=0),
        structures,
    )


def _probe(
    dataset: str,
    first_train: np.ndarray,
    ordered_train: np.ndarray,
    labels_train: np.ndarray,
    first_val: np.ndarray,
    ordered_val: np.ndarray,
    labels_val: np.ndarray,
) -> dict[str, object]:
    if dataset == "CMUMOSI":
        first_model = Ridge(alpha=10.0)
        ordered_model = Ridge(alpha=10.0)
        first_target = labels_train
        val_target = labels_val
    else:
        first_model = RidgeClassifier(alpha=10.0)
        ordered_model = RidgeClassifier(alpha=10.0)
        first_target = labels_train.astype(np.int64)
        val_target = labels_val.astype(np.int64)

    scaler_first = StandardScaler().fit(first_train)
    scaler_ordered = StandardScaler().fit(ordered_train)
    first_train_scaled = scaler_first.transform(first_train)
    first_val_scaled = scaler_first.transform(first_val)
    ordered_train_scaled = scaler_ordered.transform(ordered_train)
    ordered_val_scaled = scaler_ordered.transform(ordered_val)
    first_model.fit(first_train_scaled, first_target)
    ordered_model.fit(ordered_train_scaled, first_target)
    first_prediction = first_model.predict(first_val_scaled)
    ordered_prediction = ordered_model.predict(ordered_val_scaled)
    if dataset == "CMUMOSI":
        first_score = {
            "mae": float(mean_absolute_error(val_target, first_prediction)),
            "r2": float(r2_score(val_target, first_prediction)),
            "correlation": float(
                np.corrcoef(val_target.reshape(-1), first_prediction.reshape(-1))[0, 1]
            ),
        }
        ordered_score = {
            "mae": float(mean_absolute_error(val_target, ordered_prediction)),
            "r2": float(r2_score(val_target, ordered_prediction)),
            "correlation": float(
                np.corrcoef(val_target.reshape(-1), ordered_prediction.reshape(-1))[0, 1]
            ),
        }
        delta = {
            "mae": ordered_score["mae"] - first_score["mae"],
            "r2": ordered_score["r2"] - first_score["r2"],
            "correlation": ordered_score["correlation"] - first_score["correlation"],
        }
    else:
        first_score = {
            "accuracy": float(accuracy_score(val_target, first_prediction)),
            "macro_f1": float(f1_score(val_target, first_prediction, average="macro")),
        }
        ordered_score = {
            "accuracy": float(accuracy_score(val_target, ordered_prediction)),
            "macro_f1": float(f1_score(val_target, ordered_prediction, average="macro")),
        }
        delta = {
            key: ordered_score[key] - first_score[key] for key in first_score
        }
    return {"first_order": first_score, "ordered": ordered_score, "delta": delta}


def _aggregate_structures(
    structures: Sequence[dict[str, object]],
    seed: int,
    dataset: str,
    split: str,
    rate: float,
) -> dict[str, object]:
    observed = [float(s["pair_mean"]) for s in structures if s["pair_mean"] is not None]
    median = [float(s["pair_median"]) for s in structures if s["pair_median"] is not None]
    aggregate = [
        float(s["aggregate_score"])
        for s in structures
        if s["aggregate_score"] is not None
    ]
    shuffle_means = [
        float(s["shuffle_pair_mean"])
        for s in structures
        if s.get("shuffle_pair_mean") is not None
    ]
    rng = np.random.default_rng(
        SeedBundle(seed).derive(f"relation_order_shuffle:{split}:{rate}")
    )
    return {
        "dataset": dataset,
        "split": split,
        "rate": rate,
        "conversation_count": len(structures),
        "observed_pair_mean": float(np.mean(observed)) if observed else None,
        "observed_pair_median": float(np.mean(median)) if median else None,
        "observed_aggregate_mean": float(np.mean(aggregate)) if aggregate else None,
        "shuffle_pair_mean": float(np.mean(shuffle_means)) if shuffle_means else None,
        "shuffle_repeats": SHUFFLE_REPEATS,
        "shuffle_seed": int(rng.integers(0, 2**31 - 1)),
    }


def _run_self_test() -> None:
    records, relation_sizes = _edge_records(3, [0, 1, 0], 2, 2)
    assert relation_sizes == (3, 4)
    assert (0, 1, 0, 2) in records  # source 0 -> target 1 is past
    assert (2, 1, 2, 2) in records  # source 2 -> target 1 is future
    matrices = [_adjacency(3, records, temporal_type=i) for i in range(3)]
    assert all(matrix.shape == (3, 3) for matrix in matrices)
    score = _commutator_score(matrices[0], matrices[1])
    assert score is not None and math.isfinite(score)
    print("self-test: PASS")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("CMUMOSI", "IEMOCAPSix"))
    parser.add_argument("--feature-root")
    parser.add_argument("--output")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--probe-seeds", default="66,67,68")
    parser.add_argument("--rates", default="0.0,0.5,0.7")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _run_self_test()
        return
    if not args.dataset or not args.feature_root or not args.output:
        raise ValueError("dataset, feature-root, and output are required")
    probe_seeds = tuple(int(value) for value in args.probe_seeds.split(","))
    rates = tuple(float(value) for value in args.rates.split(","))
    if not probe_seeds or not rates:
        raise ValueError("probe-seeds and rates cannot be empty")
    if args.dataset == "CMUMOSI" and args.fold != 1:
        raise ValueError("CMUMOSI has one fold")
    if args.dataset == "IEMOCAPSix" and args.fold != 5:
        raise ValueError("this audit uses IEMOCAP fold 5")

    projection = np.random.default_rng(RANDOM_PROJECTION_SEED).normal(
        0.0,
        1.0 / math.sqrt(2560.0),
        size=(2560, RANDOM_PROJECTION_DIM),
    ).astype(np.float64)
    all_results: dict[str, object] = {
        "dataset": args.dataset,
        "fold": args.fold,
        "feature_root": args.feature_root,
        "window_past": DEFAULT_W_PAST,
        "window_future": DEFAULT_W_FUTURE,
        "probe_seeds": list(probe_seeds),
        "rates": list(rates),
        "test_loaded": False,
        "projection_dim": RANDOM_PROJECTION_DIM,
        "projection_seed": RANDOM_PROJECTION_SEED,
        "structural": [],
        "probes": [],
    }

    loader_args = [
        args.feature_root + "/wav2vec-large-c-UTT",
        args.feature_root + "/deberta-large-4-UTT",
        args.feature_root + "/manet_UTT",
        5 if args.dataset == "IEMOCAPSix" else 1,
        args.dataset,
        args.batch_size,
        0,
    ]
    for seed in probe_seeds:
        train_loaders, validation_loaders, _, adim, tdim, vdim = get_loaders(
            *loader_args,
            seed=seed,
            validation_fraction=0.1,
            evaluation_protocol="official",
        )
        loader_index = 4 if args.dataset == "IEMOCAPSix" else 0
        train_loader = train_loaders[loader_index]
        validation_loader = validation_loaders[loader_index]
        dimensions = (int(adim), int(tdim), int(vdim))
        for rate in rates:
            train_first, train_ordered, train_y, train_structures = _collect_split(
                train_loader,
                args.dataset,
                "train",
                args.fold,
                seed,
                rate,
                dimensions,
                projection,
            )
            val_first, val_ordered, val_y, val_structures = _collect_split(
                validation_loader,
                args.dataset,
                "validation",
                args.fold,
                seed,
                rate,
                dimensions,
                projection,
            )
            probe = _probe(
                args.dataset,
                train_first,
                train_ordered,
                train_y,
                val_first,
                val_ordered,
                val_y,
            )
            all_results["probes"].append(
                {
                    "seed": seed,
                    "rate": rate,
                    "train_nodes": int(len(train_y)),
                    "validation_nodes": int(len(val_y)),
                    **probe,
                }
            )
            all_results["structural"].append(
                {
                    "seed": seed,
                    "rate": rate,
                    "train": _aggregate_structures(
                        train_structures, seed, args.dataset, "train", rate
                    ),
                    "validation": _aggregate_structures(
                        val_structures, seed, args.dataset, "validation", rate
                    ),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(all_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
