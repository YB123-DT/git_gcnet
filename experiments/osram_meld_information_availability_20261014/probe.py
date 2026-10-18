"""Uniform offline classification probes for frozen MELD representations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .features import DonorMatch


PROBE_PAIRS = {
    "Base": "BaseDonor",
    "Memory": "MemoryDonor",
    "LocalBase": "LocalBaseDonor",
    "LocalMemory": "LocalMemoryDonor",
    "RawHistory": "RawHistoryDonor",
    "LocalRawHistory": "LocalRawHistoryDonor",
}


class MeldProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, classes: int = 7):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _metadata(data: dict, rows: np.ndarray | None = None) -> np.ndarray:
    if rows is None:
        rows = np.arange(len(data["local"]))
    availability = np.asarray(data["availability"], dtype=np.float32)
    return np.column_stack(
        (
            availability[rows],
            np.asarray(data["all_present"], dtype=np.float32)[rows],
            np.asarray(data["same_speaker_present"], dtype=np.float32)[rows],
            np.log1p(np.asarray(data["all_count"], dtype=np.float32)[rows]),
            np.log1p(np.asarray(data["same_speaker_count"], dtype=np.float32)[rows]),
        )
    ).astype(np.float32)


def _donated(values: np.ndarray, donor: DonorMatch) -> np.ndarray:
    result = np.zeros_like(values)
    result[donor.included] = values[donor.indices[donor.included]]
    return result


def _donor_metadata(data: dict, donor: DonorMatch) -> np.ndarray:
    n = len(data["local"])
    result = np.zeros((n, 7), dtype=np.float32)
    result[:, :3] = np.asarray(data["availability"], dtype=np.float32)
    valid = donor.included
    source = donor.indices[valid]
    result[valid, 3] = np.asarray(data["all_present"], dtype=np.float32)[source]
    result[valid, 4] = np.asarray(data["same_speaker_present"], dtype=np.float32)[source]
    result[valid, 5] = np.log1p(np.asarray(data["all_count"], dtype=np.float32)[source])
    result[valid, 6] = np.log1p(np.asarray(data["same_speaker_count"], dtype=np.float32)[source])
    return result


def build_probe_inputs(data: dict, donor: DonorMatch) -> dict[str, np.ndarray]:
    """Build real and exact-width donor inputs; current Local is never donated."""
    local = np.asarray(data["local"], dtype=np.float32)
    base = np.asarray(data["base"], dtype=np.float32)
    memory = np.concatenate((base, np.asarray(data["gap"], dtype=np.float32).reshape(len(local), -1)), axis=1)
    history = np.concatenate(
        (np.asarray(data["all_mean"], dtype=np.float32), np.asarray(data["same_speaker_mean"], dtype=np.float32)),
        axis=1,
    )
    metadata = _metadata(data)
    donor_metadata = _donor_metadata(data, donor)
    donor_base = _donated(base, donor)
    donor_memory = _donated(memory, donor)
    donor_history = _donated(history, donor)
    availability = np.asarray(data["availability"], dtype=np.float32)
    result = {
        "Local": np.concatenate((local, availability), axis=1),
        "Base": np.concatenate((base, metadata), axis=1),
        "BaseDonor": np.concatenate((donor_base, donor_metadata), axis=1),
        "Memory": np.concatenate((memory, metadata), axis=1),
        "MemoryDonor": np.concatenate((donor_memory, donor_metadata), axis=1),
        "LocalBase": np.concatenate((local, base, metadata), axis=1),
        "LocalBaseDonor": np.concatenate((local, donor_base, donor_metadata), axis=1),
        "LocalMemory": np.concatenate((local, memory, metadata), axis=1),
        "LocalMemoryDonor": np.concatenate((local, donor_memory, donor_metadata), axis=1),
        "RawHistory": np.concatenate((history, metadata), axis=1),
        "RawHistoryDonor": np.concatenate((donor_history, donor_metadata), axis=1),
        "LocalRawHistory": np.concatenate((local, history, metadata), axis=1),
        "LocalRawHistoryDonor": np.concatenate((local, donor_history, donor_metadata), axis=1),
    }
    return {name: values.astype(np.float32, copy=False) for name, values in result.items()}


def fit_normalizer(train: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(train, dtype=np.float32)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def transform(values: np.ndarray, normalizer: dict[str, np.ndarray]) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float32) - normalizer["mean"]) / normalizer["scale"]).astype(np.float32)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("labels and predictions must be nonempty aligned vectors")
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    destination: Path,
    device: str = "cpu",
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    classes: int = 7,
) -> dict:
    """Fit one internal Test-oracle probe and save enough to recompute metrics."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    normalizer = fit_normalizer(train_x)
    x_train = torch.as_tensor(transform(train_x, normalizer), device=device)
    x_test = torch.as_tensor(transform(test_x, normalizer), device=device)
    y_train = torch.as_tensor(train_y, dtype=torch.long, device=device)
    y_test = torch.as_tensor(test_y, dtype=torch.long, device=device)
    model = MeldProbe(train_x.shape[1], classes=classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    best_key = None
    best_state = None
    best_epoch = None
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for indices_cpu in order.split(batch_size):
            indices = indices_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x_train[indices]), y_train[indices])
            if not torch.isfinite(loss):
                raise ValueError("non-finite probe loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("non-finite probe gradient")
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x_test)
            loss_test = float(nn.functional.cross_entropy(logits, y_test))
            predicted = logits.argmax(-1).cpu().numpy()
        metric = classification_metrics(test_y, predicted)
        curve.append({"epoch": epoch, "cross_entropy": loss_test, **metric})
        key = (metric["accuracy"], metric["weighted_f1"], -loss_test)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("probe never produced a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x_test).cpu().numpy()
    predicted = logits.argmax(axis=1)
    destination.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "state_dict": best_state,
            "normalizer": normalizer,
            "seed": seed,
            "epoch": best_epoch,
            "selection": "maximum test accuracy, then weighted F1, then minimum CE",
            "input_dim": train_x.shape[1],
        },
        destination / "best.pt",
    )
    np.savez_compressed(destination / "predictions.npz", labels=test_y, logits=logits, predictions=predicted)
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selection_split": "test",
        "selection_rule": "accuracy>weighted_f1>-cross_entropy",
        "metrics": classification_metrics(test_y, predicted),
        "curve": curve,
    }
    (destination / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
