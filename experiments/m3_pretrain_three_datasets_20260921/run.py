"""Three-dataset utterance-level M3/JEPA pretraining.

This module intentionally stops at modality-projector and predictor training.  It
does not import or execute an OSRAM/GCNet classifier path.  The command-line
entrypoint is explicit; importing the module never starts a dataset run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from gcnet_missing_m3.b2 import SourceOnlyM3Predictor
from gcnet_missing_m3.loss import missing_m3_loss
from gcnet_missing_m3.model import (
    MODALITIES,
    EMATeacherProjectors,
    ModalityProjector,
    _validate_observed_inputs,
)
from gcnet_modality_jepa.train_gcnet import generate_inputs, get_loaders, set_random_seed


DATASETS = ("CMUMOSI", "CMUMOSEI", "IEMOCAPSix")
SOURCE_PATTERNS = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
)
ROUTES = {
    "A->T": ((0,), 1),
    "A->V": ((0,), 2),
    "T->A": ((1,), 0),
    "T->V": ((1,), 2),
    "V->A": ((2,), 0),
    "V->T": ((2,), 1),
    "AT->V": ((0, 1), 2),
    "AV->T": ((0, 2), 1),
    "TV->A": ((1, 2), 0),
}


def sample_source_availability(
    umask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one of the six non-empty source sets per valid utterance.

    ``umask`` is the loader convention ``[B, L]`` and the returned mask is the
    model convention ``[L, B, 3]``.  Padding is always all-zero and no valid
    utterance receives the all-missing pattern.
    """

    if umask.ndim != 2:
        raise ValueError("umask must have shape [B, L]")
    if not bool(((umask == 0) | (umask == 1)).all()):
        raise ValueError("umask must be binary")
    length = int(umask.shape[1])
    batch = int(umask.shape[0])
    device = umask.device
    # Generate on CPU so callers can pass a reproducible CPU Generator even
    # when the model is on CUDA.
    ids = torch.randint(
        len(SOURCE_PATTERNS),
        (length, batch),
        generator=generator,
        device="cpu",
    )
    bank = torch.tensor(SOURCE_PATTERNS, dtype=umask.dtype)
    availability = bank[ids].to(device=device)
    availability = availability * umask.T.bool().unsqueeze(-1).to(umask.dtype)
    return availability


def uniform_dataset_schedule(
    dataset_names: Sequence[str],
    *,
    steps: int,
    generator: torch.Generator | None = None,
) -> list[str]:
    """Return a reproducible approximately-uniform dataset schedule."""

    names = tuple(str(name) for name in dataset_names)
    if not names:
        raise ValueError("dataset_names cannot be empty")
    if int(steps) < 0:
        raise ValueError("steps must be non-negative")
    ids = torch.randint(len(names), (int(steps),), generator=generator)
    return [names[int(index)] for index in ids.tolist()]


class ThreeDatasetJEPA(nn.Module):
    """Shared projector/EMA/predictor bank for three complete-view datasets."""

    def __init__(
        self,
        dimensions: Sequence[int],
        latent_dim: int = 256,
        projector_dropout: float = 0.1,
        predictor_dropout: float = 0.1,
        dropout: float | None = None,
        num_experts: int = 4,
        top_k: int = 2,
        mmoe_variant: str = "dual-gate",
        target_private_rank: int = 0,
    ) -> None:
        super().__init__()
        if len(tuple(dimensions)) != 3 or any(int(value) <= 0 for value in dimensions):
            raise ValueError("dimensions must contain three positive integers")
        self.dimensions = tuple(int(value) for value in dimensions)
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if dropout is not None:
            projector_dropout = float(dropout)
            predictor_dropout = float(dropout)
        self.projectors = nn.ModuleDict(
            {
                name: ModalityProjector(width, self.latent_dim, projector_dropout)
                for name, width in zip(MODALITIES, self.dimensions)
            }
        )
        self.teacher = EMATeacherProjectors(self.projectors)
        self.predictor = SourceOnlyM3Predictor(
            latent_dim=self.latent_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=predictor_dropout,
            mmoe_variant=mmoe_variant,
            target_private_rank=target_private_rank,
        )

    def _validate(
        self,
        full_features: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> torch.Tensor:
        return _validate_observed_inputs(
            full_features, availability, umask, self.dimensions
        )

    def source_latents(
        self,
        full_features: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode only true observed slots; missing raw features are ignored."""

        valid = self._validate(full_features, availability, umask)
        latent_shape = (*full_features.shape[:2], self.latent_dim)
        latents: dict[str, torch.Tensor] = {}
        start = 0
        for index, (name, width) in enumerate(zip(MODALITIES, self.dimensions)):
            block = full_features[..., start : start + width]
            selected = valid & availability[..., index].bool()
            latent = full_features.new_zeros(latent_shape)
            if bool(selected.any()):
                latent[selected] = self.projectors[name](block[selected])
            latents[name] = latent
            start += width
        return latents

    @torch.no_grad()
    def teacher_targets(
        self,
        full_features: torch.Tensor,
        umask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode every complete modality with the frozen EMA target bank."""

        if full_features.ndim != 3 or full_features.shape[-1] != sum(self.dimensions):
            raise ValueError("full_features must have shape [L, B, sum(dimensions)]")
        if umask.shape != full_features.shape[:2][::-1]:
            raise ValueError("umask must have shape [B, L]")
        valid = umask.T.bool()
        targets: dict[str, torch.Tensor] = {}
        start = 0
        for name, width in zip(MODALITIES, self.dimensions):
            block = full_features[..., start : start + width]
            target = full_features.new_zeros(
                *full_features.shape[:2], self.latent_dim
            )
            if bool(valid.any()):
                target[valid] = self.teacher[name](block[valid])
            targets[name] = target
            start += width
        return targets

    def forward(
        self,
        full_features: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ):
        latents = self.source_latents(full_features, availability, umask)
        predictions = self.predictor(latents, availability, umask)
        targets = self.teacher_targets(full_features, umask)
        return predictions, targets

    @torch.no_grad()
    def update_teacher(self, tau: float) -> None:
        self.teacher.update_from(self.projectors, tau)


def route_mask(
    availability: torch.Tensor,
    umask: torch.Tensor,
    source_indices: Sequence[int],
    target_index: int,
) -> torch.Tensor:
    """Select an exact source pattern for one directed route."""

    if availability.ndim != 3 or availability.shape[-1] != 3:
        raise ValueError("availability must have shape [L, B, 3]")
    valid = umask.T.bool()
    expected = torch.zeros_like(availability, dtype=torch.bool)
    expected[..., list(source_indices)] = True
    expected[..., target_index] = False
    return valid & (availability.bool() == expected).all(dim=-1)


def route_prediction(
    model: ThreeDatasetJEPA,
    latents: Mapping[str, torch.Tensor],
    availability: torch.Tensor,
    umask: torch.Tensor,
    source_indices: Sequence[int],
    target_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one exact M3 route, averaging sources only when needed."""

    mask = route_mask(availability, umask, source_indices, target_index)
    output = latents[MODALITIES[0]].new_zeros(
        *latents[MODALITIES[0]].shape[:2], model.latent_dim
    )
    if bool(mask.any()):
        values = []
        for source_index in source_indices:
            source = latents[MODALITIES[source_index]][mask]
            reg, _ = model.predictor.mmoe(
                model.predictor.input_norm(source), source_index, target_index
            )
            values.append(reg)
        output[mask] = torch.stack(values, dim=0).mean(dim=0)
    return output, mask


def latent_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    target_mean: torch.Tensor,
) -> dict[str, float | int | None]:
    """Compute prediction quality without using labels or a classifier."""

    prediction = prediction.detach().float().reshape(-1, prediction.shape[-1])
    target = target.detach().float().reshape(-1, target.shape[-1])
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    count = int(prediction.shape[0])
    if count == 0:
        return {
            "count": 0,
            "centered_cosine": None,
            "prediction_std": None,
            "target_std": None,
            "prediction_mse": None,
            "train_target_mean_baseline_gap": None,
        }
    centered_prediction = prediction - prediction.mean(dim=0, keepdim=True)
    centered_target = target - target.mean(dim=0, keepdim=True)
    numerator = (centered_prediction * centered_target).sum()
    denominator = centered_prediction.norm() * centered_target.norm()
    centered_cosine = numerator / denominator.clamp_min(1e-12)
    prediction_mse = (prediction - target).square().mean()
    target_mean = target_mean.detach().float().reshape(1, -1)
    baseline_mse = (target - target_mean).square().mean()
    return {
        "count": count,
        "centered_cosine": float(centered_cosine.item()),
        "prediction_std": float(prediction.std(unbiased=False).item()),
        "target_std": float(target.std(unbiased=False).item()),
        "prediction_mse": float(prediction_mse.item()),
        "train_target_mean_baseline_gap": float((baseline_mse - prediction_mse).item()),
    }


def _batch_to_features(raw_batch) -> tuple[torch.Tensor, torch.Tensor]:
    if len(raw_batch) < 8:
        raise ValueError("loader batch must contain at least eight fields")
    full = generate_inputs(*raw_batch[:7])[0]
    umask = raw_batch[7]
    return full, umask


def build_joint_loaders(
    feature_roots: Mapping[str, Sequence[str]],
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    iemocap_fold: int = 5,
    evaluation_protocol: str = "strict",
):
    """Load separate train/validation streams while sharing feature dimensions."""

    if set(feature_roots) != set(DATASETS):
        raise ValueError(f"feature_roots must contain exactly {DATASETS}")
    result = {}
    shared_dims = None
    for dataset_name in DATASETS:
        roots = tuple(feature_roots[dataset_name])
        if len(roots) != 3:
            raise ValueError("each feature root entry must contain audio/text/video")
        is_iemocap = dataset_name == "IEMOCAPSix"
        train, validation, _test, adim, tdim, vdim = get_loaders(
            audio_root=roots[0],
            text_root=roots[1],
            video_root=roots[2],
            num_folder=5 if is_iemocap else 1,
            dataset=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            evaluation_protocol=evaluation_protocol,
        )
        fold = int(iemocap_fold) - 1 if is_iemocap else 0
        if not 0 <= fold < len(train):
            raise ValueError("iemocap_fold must be in 1..5")
        dims = (int(adim), int(tdim), int(vdim))
        if shared_dims is None:
            shared_dims = dims
        elif dims != shared_dims:
            raise ValueError(
                "joint pretraining requires shared feature dimensions; "
                f"got {shared_dims} and {dims} for {dataset_name}"
            )
        result[dataset_name] = {
            "train": train[fold],
            "validation": validation[fold],
            "dimensions": dims,
        }
    return result, shared_dims


def _collect_routes(
    model: ThreeDatasetJEPA,
    loader,
    *,
    generator: torch.Generator,
    device: torch.device,
):
    model.eval()
    rows = {name: {"prediction": [], "target": []} for name in ROUTES}
    with torch.no_grad():
        for raw_batch in loader:
            full, umask = _batch_to_features(raw_batch)
            full = full.to(device)
            umask = umask.to(device)
            availability = sample_source_availability(umask, generator=generator)
            latents = model.source_latents(full, availability, umask)
            targets = model.teacher_targets(full, umask)
            for route_name, (sources, target_index) in ROUTES.items():
                prediction, mask = route_prediction(
                    model, latents, availability, umask, sources, target_index
                )
                if bool(mask.any()):
                    rows[route_name]["prediction"].append(prediction[mask].cpu())
                    rows[route_name]["target"].append(
                        targets[MODALITIES[target_index]][mask].cpu()
                    )
    return {
        route: {
            "prediction": torch.cat(values["prediction"])
            if values["prediction"]
            else torch.empty(0, model.latent_dim),
            "target": torch.cat(values["target"])
            if values["target"]
            else torch.empty(0, model.latent_dim),
        }
        for route, values in rows.items()
    }


@torch.no_grad()
def evaluate_train_validation(
    model: ThreeDatasetJEPA,
    dataset_loaders: Mapping[str, Mapping[str, object]],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Evaluate each dataset independently; train targets set the baseline mean."""

    output = {}
    for dataset_offset, (dataset_name, loaders) in enumerate(dataset_loaders.items()):
        train_generator = torch.Generator().manual_seed(seed + 1000 + dataset_offset)
        val_generator = torch.Generator().manual_seed(seed + 2000 + dataset_offset)
        train_rows = _collect_routes(
            model, loaders["train"], generator=train_generator, device=device
        )
        val_rows = _collect_routes(
            model, loaders["validation"], generator=val_generator, device=device
        )
        dataset_output = {}
        for route in ROUTES:
            train_target = train_rows[route]["target"]
            target_mean = (
                train_target.mean(dim=0)
                if train_target.shape[0]
                else torch.zeros(model.latent_dim)
            )
            dataset_output[route] = {
                "train": latent_metrics(
                    train_rows[route]["prediction"],
                    train_rows[route]["target"],
                    target_mean=target_mean,
                ),
                "validation": latent_metrics(
                    val_rows[route]["prediction"],
                    val_rows[route]["target"],
                    target_mean=target_mean,
                ),
            }
        output[dataset_name] = dataset_output
    return output


def train_joint(
    model: ThreeDatasetJEPA,
    loaders: Mapping[str, Mapping[str, object]],
    *,
    epochs: int,
    steps_per_epoch: int,
    seed: int,
    device: torch.device,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    temperature: float = 0.03,
    ema_tau: float = 0.99,
    clip_norm: float = 1.0,
) -> list[dict[str, float | int]]:
    """Explicit training loop; callers decide whether and how long to run it."""

    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    model.to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    iterators = {
        name: iter(loaders[name]["train"]) for name in DATASETS
    }
    history = []
    mask_generator = torch.Generator().manual_seed(seed + 9000)
    schedule_generator = torch.Generator().manual_seed(seed + 9100)
    for epoch in range(1, epochs + 1):
        model.train()
        dataset_schedule = uniform_dataset_schedule(
            DATASETS, steps=steps_per_epoch, generator=schedule_generator
        )
        losses = []
        target_counts = []
        for dataset_name in dataset_schedule:
            try:
                raw_batch = next(iterators[dataset_name])
            except StopIteration:
                iterators[dataset_name] = iter(loaders[dataset_name]["train"])
                raw_batch = next(iterators[dataset_name])
            full, umask = _batch_to_features(raw_batch)
            full = full.to(device)
            umask = umask.to(device)
            availability = sample_source_availability(umask, generator=mask_generator)
            predictions, targets = model(full, availability, umask)
            objective = missing_m3_loss(
                predictions,
                targets,
                temperature=temperature,
                include_contrastive=True,
            )
            optimizer.zero_grad(set_to_none=True)
            objective.total.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            model.update_teacher(ema_tau)
            losses.append(float(objective.total.detach().item()))
            target_counts.append(objective.target_count)
        history.append(
            {
                "epoch": epoch,
                "loss": sum(losses) / max(1, len(losses)),
                "target_count": sum(target_counts),
            }
        )
    return history


def _synthetic_smoke(device: str = "cpu") -> dict[str, object]:
    torch.manual_seed(123)
    target_device = torch.device(device)
    model = ThreeDatasetJEPA(
        dimensions=(4, 5, 6),
        latent_dim=8,
        projector_dropout=0.0,
        predictor_dropout=0.0,
        num_experts=2,
        top_k=1,
    ).to(target_device)
    length, batch = 5, 3
    features = torch.randn(length, batch, 15, device=target_device)
    umask = torch.ones(batch, length, device=target_device)
    availability = sample_source_availability(
        umask, generator=torch.Generator().manual_seed(7)
    )
    predictions, targets = model(features, availability, umask)
    objective = missing_m3_loss(
        predictions, targets, temperature=0.2, include_contrastive=True
    )
    objective.total.backward()
    model.update_teacher(0.99)
    text_mask = predictions.target_mask[..., 1]
    metrics = (
        latent_metrics(
            predictions.reg_predictions[..., 1, :][text_mask],
            targets["text"][text_mask],
            target_mean=targets["text"][text_mask].mean(dim=0),
        )
        if bool(text_mask.any())
        else {"count": 0}
    )
    return {
        "loss": float(objective.total.detach().item()),
        "target_count": objective.target_count,
        "metric_count": metrics["count"],
        "finite": bool(torch.isfinite(objective.total).item()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly start a data run; omitted by design",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="experiments/m3_pretrain_three_datasets_20260921/output")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--projector-dropout", type=float, default=0.1)
    parser.add_argument("--predictor-dropout", type=float, default=0.1)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--ema-tau", type=float, default=0.99)
    parser.add_argument("--iemocap-fold", type=int, default=5)
    parser.add_argument("--evaluation-protocol", choices=("strict", "official"), default="strict")
    parser.add_argument(
        "--feature-root",
        action="append",
        nargs=4,
        metavar=("DATASET", "AUDIO", "TEXT", "VISUAL"),
        help="repeat once for CMUMOSI, CMUMOSEI, and IEMOCAPSix",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        print(json.dumps(_synthetic_smoke(args.device), indent=2, sort_keys=True))
        return
    if not args.run:
        raise SystemExit(
            "No dataset run was started. Re-run with --run and one --feature-root "
            "entry for each dataset."
        )
    if not args.feature_root:
        raise SystemExit("--feature-root is required when --run is set")
    feature_roots = {entry[0]: entry[1:] for entry in args.feature_root}
    if set(feature_roots) != set(DATASETS):
        raise SystemExit(f"--feature-root is required once for each {DATASETS}")
    set_random_seed(args.seed)
    device = torch.device(args.device)
    loaders, dimensions = build_joint_loaders(
        feature_roots,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        iemocap_fold=args.iemocap_fold,
        evaluation_protocol=args.evaluation_protocol,
    )
    model_config = {
        "dimensions": dimensions,
        "latent_dim": args.latent_dim,
        "projector_dropout": args.projector_dropout,
        "predictor_dropout": args.predictor_dropout,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
    }
    model = ThreeDatasetJEPA(**model_config)
    history = train_joint(
        model,
        loaders,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        ema_tau=args.ema_tau,
    )
    metrics = evaluate_train_validation(model, loaders, seed=args.seed, device=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_config": model_config,
            "model_state": model.state_dict(),
            "dimensions": dimensions,
            "seed": args.seed,
            "evaluation_protocol": args.evaluation_protocol,
            "history": history,
        },
        checkpoint_path,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
