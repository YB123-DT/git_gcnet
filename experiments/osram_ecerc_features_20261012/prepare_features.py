"""Convert ECERC's conversation pickles into GCNet's utterance feature layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def _load(path: Path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream, encoding="latin1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_ecerc_features(base_pickle: Path, roberta_pickle: Path, output: Path) -> dict:
    """Export ECERC audio/RoBERTa/visual features without changing sample order."""
    base_pickle, roberta_pickle, output = map(Path, (base_pickle, roberta_pickle, output))
    video_ids, _, labels, _, audio, visual, _, train_ids, test_ids = _load(base_pickle)
    roberta = _load(roberta_pickle)
    text = roberta[2]  # ECERC uses roberta1: last-layer CLS embedding.
    dialogue_ids = set(video_ids)
    for name, values in (("labels", labels), ("audio", audio), ("visual", visual), ("text", text)):
        if set(values) != dialogue_ids:
            raise ValueError(f"{name} dialogue ids do not match ECERC base pickle")
    if set(train_ids) | set(test_ids) != dialogue_ids or set(train_ids) & set(test_ids):
        raise ValueError("ECERC train/test dialogue ids are invalid")

    roots = {name: output / name for name in ("audio", "text", "visual")}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    dimensions = {}
    utterances = 0
    for dialogue in sorted(dialogue_ids):
        names = list(video_ids[dialogue])
        arrays = {
            "audio": np.asarray(audio[dialogue]),
            "text": np.asarray(text[dialogue]),
            "visual": np.asarray(visual[dialogue]),
        }
        expected = len(names)
        if len(labels[dialogue]) != expected or any(len(value) != expected for value in arrays.values()):
            raise ValueError(f"utterance count mismatch in {dialogue}")
        for modality, values in arrays.items():
            if values.ndim != 2:
                raise ValueError(f"{modality} must be [utterance, feature]")
            dimension = int(values.shape[1])
            if modality in dimensions and dimensions[modality] != dimension:
                raise ValueError(f"inconsistent {modality} dimension")
            dimensions[modality] = dimension
            for uid, value in zip(names, values):
                np.save(roots[modality] / f"{uid}.npy", np.asarray(value, dtype=np.float32))
        utterances += expected

    manifest = {
        "source": {
            "base_pickle": str(base_pickle.resolve()),
            "base_sha256": _sha256(base_pickle),
            "roberta_pickle": str(roberta_pickle.resolve()),
            "roberta_sha256": _sha256(roberta_pickle),
            "text_tensor": "roberta1 (tuple index 2)",
        },
        "dialogues": len(dialogue_ids),
        "train_dialogues": len(train_ids),
        "test_dialogues": len(test_ids),
        "utterances": utterances,
        "dimensions": dimensions,
        "feature_roots": {name: str(path.resolve()) for name, path in roots.items()},
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pickle", type=Path, required=True)
    parser.add_argument("--roberta-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_ecerc_features(args.base_pickle, args.roberta_pickle, args.output), indent=2))


if __name__ == "__main__":
    main()
