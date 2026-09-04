"""Audit modality-wise Original GCNet reconstruction gradients.

The diagnostic deliberately measures gradients at shared representations and
shared backbone parameters.  Gradients on ``linear_rec.weight`` are not used
for conflict claims because the audio/text/visual losses touch disjoint output
row blocks there and are therefore orthogonal by construction.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from gcnet_missing_m3.train_gcnet import _move_batch, _prepare_view, get_loaders
from gcnet_modality_jepa.mask_schedule import ConversationMaskSchedule
from gcnet_modality_jepa.protocol import SeedBundle
from gcnet_original_stratified.model import OriginalGCNetControl


RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
MODALITIES = ("audio", "text", "visual")


@dataclass(frozen=True)
class DatasetContract:
    classes: int
    speakers: int
    folds: int


CONTRACTS = {
    "IEMOCAPFour": DatasetContract(4, 2, 5),
    "IEMOCAPSix": DatasetContract(6, 2, 5),
    "CMUMOSI": DatasetContract(1, 1, 1),
    "CMUMOSEI": DatasetContract(1, 1, 1),
}


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _flatten_gradients(
    values: Iterable[torch.Tensor | None],
    references: Iterable[torch.Tensor],
) -> torch.Tensor:
    chunks = [
        (
            torch.zeros(reference.numel(), dtype=reference.dtype)
            if value is None
            else value.detach().reshape(-1).cpu()
        )
        for value, reference in zip(values, references)
    ]
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def _compatibility_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map historical PyG GraphConv names to the installed PyG convention."""

    mapped: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        name = name.replace(".conv2.lin_l.", ".conv2.lin_rel.")
        name = name.replace(".conv2.lin_r.", ".conv2.lin_root.")
        mapped[name] = value
    return mapped


def _load_checkpoint(
    checkpoint_path: Path,
    dimensions: tuple[int, int, int],
    device: torch.device,
) -> tuple[OriginalGCNetControl, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    dataset = str(config["dataset"])
    contract = CONTRACTS[dataset]
    model = OriginalGCNetControl(
        base_model=str(config.get("base_model", "LSTM")),
        adim=dimensions[0],
        tdim=dimensions[1],
        vdim=dimensions[2],
        D_e=int(config.get("hidden", 200)),
        graph_hidden_size=int(config.get("hidden", 200)) // 2,
        n_speakers=contract.speakers,
        window_past=int(config.get("window_past", 2)),
        window_future=int(config.get("window_future", 2)),
        n_classes=contract.classes,
        dropout=float(config.get("dropout", 0.5)),
        time_attn=bool(config.get("time_attention", False)),
        no_cuda=device.type != "cuda",
    )
    state = _compatibility_state(checkpoint["model"])
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {result}")
    model.to(device).eval()
    return model, config


def _modality_losses(
    reconstruction: torch.Tensor,
    complete: torch.Tensor,
    availability: torch.Tensor,
    umask: torch.Tensor,
    dimensions: tuple[int, int, int],
) -> tuple[list[torch.Tensor], list[int]]:
    reconstructed = torch.split(reconstruction, dimensions, dim=-1)
    targets = torch.split(complete, dimensions, dim=-1)
    valid = umask.transpose(0, 1).bool()
    denominator = umask.sum().to(dtype=reconstruction.dtype)
    losses: list[torch.Tensor] = []
    counts: list[int] = []
    for index, dimension in enumerate(dimensions):
        selected = valid & availability[..., index].eq(0)
        error = (reconstructed[index] - targets[index]).square()
        loss = (error * selected.unsqueeze(-1)).sum() / dimension / denominator
        losses.append(loss)
        counts.append(int(selected.sum().item()))
    return losses, counts


def _schedule(dataset: str, fold: int, seed: int, rate: float) -> ConversationMaskSchedule:
    return ConversationMaskSchedule(
        dataset=dataset,
        split="test",
        fold=fold,
        requested_missing_rate=rate,
        mask_seed=SeedBundle(seed).derive("missing_mask"),
        freeze_evaluation=True,
    )


def audit_rate(
    model: OriginalGCNetControl,
    loader: Iterable[Sequence[object]],
    schedule: ConversationMaskSchedule,
    dimensions: tuple[int, int, int],
    device: torch.device,
) -> dict[str, object]:
    shared_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("linear_rec.") and not name.startswith("smax_fc.")
    ]
    aggregate = [torch.zeros(sum(p.numel() for p in shared_parameters)) for _ in MODALITIES]
    loss_sums = [0.0, 0.0, 0.0]
    missing_counts = [0, 0, 0]
    valid_count = 0
    batch_backbone_cosines = {(0, 1): [], (0, 2): [], (1, 2): []}
    batch_hidden_cosines = {(0, 1): [], (0, 2): [], (1, 2): []}
    overlap_hidden_cosines = {(0, 1): [], (0, 2): [], (1, 2): []}
    backbone_norms = [[], [], []]
    hidden_norms = [[], [], []]

    for raw in loader:
        data = _move_batch(raw, device)
        view = _prepare_view(data, schedule, epoch=0, dimensions=dimensions)
        _, reconstruction_list, hidden, _ = model(
            [view["incomplete"]],
            view["availability"],
            view["qmask"],
            view["umask"],
            view["lengths"],
            predict_missing=False,
        )
        losses, counts = _modality_losses(
            reconstruction_list[0],
            view["complete"],
            view["availability"],
            view["umask"],
            dimensions,
        )
        batch_valid = int(view["umask"].sum().item())
        valid_count += batch_valid
        hidden_gradients: list[torch.Tensor | None] = []
        backbone_gradients: list[torch.Tensor | None] = []
        for index, (loss, count) in enumerate(zip(losses, counts)):
            missing_counts[index] += count
            loss_sums[index] += float(loss.detach()) * batch_valid
            if count == 0:
                hidden_gradients.append(None)
                backbone_gradients.append(None)
                continue
            gradients = torch.autograd.grad(
                loss,
                [hidden, *shared_parameters],
                retain_graph=True,
                allow_unused=True,
            )
            hidden_gradient = gradients[0].detach().cpu()
            backbone_gradient = _flatten_gradients(
                gradients[1:], shared_parameters
            )
            hidden_gradients.append(hidden_gradient)
            backbone_gradients.append(backbone_gradient)
            aggregate[index].add_(backbone_gradient, alpha=batch_valid)
            hidden_norms[index].append(float(torch.linalg.vector_norm(hidden_gradient)))
            backbone_norms[index].append(float(torch.linalg.vector_norm(backbone_gradient)))

        valid = view["umask"].transpose(0, 1).bool().cpu()
        availability = view["availability"].detach().cpu()
        for pair in batch_backbone_cosines:
            left, right = pair
            if backbone_gradients[left] is None or backbone_gradients[right] is None:
                continue
            backbone_cosine = _cosine(backbone_gradients[left], backbone_gradients[right])
            hidden_cosine = _cosine(
                hidden_gradients[left].reshape(-1), hidden_gradients[right].reshape(-1)
            )
            if backbone_cosine is not None:
                batch_backbone_cosines[pair].append(backbone_cosine)
            if hidden_cosine is not None:
                batch_hidden_cosines[pair].append(hidden_cosine)
            overlap = valid & availability[..., left].eq(0) & availability[..., right].eq(0)
            if bool(overlap.any()):
                overlap_cosine = _cosine(
                    hidden_gradients[left][overlap].reshape(-1),
                    hidden_gradients[right][overlap].reshape(-1),
                )
                if overlap_cosine is not None:
                    overlap_hidden_cosines[pair].append(overlap_cosine)

    def summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "negative_fraction": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "negative_fraction": float(np.mean(array < 0)),
        }

    modality = {}
    for index, name in enumerate(MODALITIES):
        modality[name] = {
            "missing_utterances": missing_counts[index],
            "loss": None if valid_count == 0 else loss_sums[index] / valid_count,
            "hidden_gradient_norm_mean": (
                None if not hidden_norms[index] else float(np.mean(hidden_norms[index]))
            ),
            "backbone_gradient_norm_mean": (
                None if not backbone_norms[index] else float(np.mean(backbone_norms[index]))
            ),
            "aggregate_backbone_gradient_norm": float(torch.linalg.vector_norm(aggregate[index])),
        }
    pairs = {}
    for left, right in batch_backbone_cosines:
        key = f"{MODALITIES[left]}-{MODALITIES[right]}"
        pairs[key] = {
            "aggregate_backbone_cosine": _cosine(aggregate[left], aggregate[right]),
            "batch_backbone_cosine": summary(batch_backbone_cosines[(left, right)]),
            "batch_hidden_cosine": summary(batch_hidden_cosines[(left, right)]),
            "overlap_hidden_cosine": summary(overlap_hidden_cosines[(left, right)]),
        }
    return {"valid_utterances": valid_count, "modalities": modality, "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = dict(checkpoint["config"])
    dataset = str(checkpoint_config["dataset"])
    contract = CONTRACTS[dataset]
    seed = int(checkpoint_config.get("seed", 66))
    fold = int(checkpoint_config.get("fold", 5 if contract.folds == 5 else 1))
    feature_names = ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
    roots = [args.feature_root / name for name in feature_names]
    if not all(root.exists() for root in roots):
        raise FileNotFoundError(f"missing feature roots: {roots}")
    loaders = get_loaders(
        audio_root=str(roots[0]),
        text_root=str(roots[1]),
        video_root=str(roots[2]),
        num_folder=contract.folds,
        dataset=dataset,
        batch_size=int(checkpoint_config.get("batch_size", 32)),
        num_workers=args.num_workers,
        seed=seed,
        validation_fraction=float(checkpoint_config.get("validation_fraction", 0.1)),
        evaluation_protocol=str(checkpoint_config.get("evaluation_protocol", "official")),
    )
    _, _, test_loaders, adim, tdim, vdim = loaders
    dimensions = (adim, tdim, vdim)
    device = torch.device(args.device)
    # cuDNN intentionally rejects RNN backward while modules are in eval mode.
    # This is a gradient diagnostic, so retain deterministic eval semantics and
    # use PyTorch's differentiable RNN implementation instead of enabling dropout.
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
    model, loaded_config = _load_checkpoint(args.checkpoint, dimensions, device)
    test_loader = test_loaders[fold - 1 if contract.folds == 5 else 0]
    rates = {}
    for rate in RATES:
        rates[str(rate)] = audit_rate(
            model,
            test_loader,
            _schedule(dataset, fold, seed, rate),
            dimensions,
            device,
        )
        print(f"completed dataset={dataset} rate={rate}", flush=True)
    payload = {
        "dataset": dataset,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_config": loaded_config,
        "feature_root": str(args.feature_root),
        "dimensions": dimensions,
        "gradient_scope": {
            "hidden": "post-graph shared hidden representation",
            "backbone": "all parameters except linear_rec and smax_fc",
            "linear_rec_note": "head row blocks are disjoint and globally orthogonal by construction",
        },
        "rates": rates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
