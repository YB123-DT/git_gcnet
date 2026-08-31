"""Audit whether Missing-M3 predictions retain utterance-level target information."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
from torch.nn import functional as F

from gcnet_modality_jepa.train_gcnet import get_loaders
from gcnet_missing_m3.model import MODALITIES, MissingM3GraphModel
from gcnet_missing_m3.train_gcnet import (
    TrainConfig,
    _build_schedule,
    _move_batch,
    _prepare_view,
    _resolve_task_contract,
)


def _effective_rank(value: torch.Tensor) -> float:
    centered = value.float() - value.float().mean(dim=0, keepdim=True)
    if hasattr(torch.linalg, "svdvals"):
        singular_values = torch.linalg.svdvals(centered)
    else:
        singular_values = torch.svd(centered, some=False).S
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum()
    return float(entropy.exp().item())


def _symmetric_info_nce(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(prediction.shape[0])
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    shuffle_seed: int,
) -> dict[str, float | int]:
    prediction = prediction.float().cpu()
    target = target.float().cpu()
    count = prediction.shape[0]
    centered_prediction = prediction - prediction.mean(dim=0, keepdim=True)
    centered_target = target - target.mean(dim=0, keepdim=True)
    real_cosine = F.cosine_similarity(prediction, target, dim=-1).mean()
    centered_cosine = F.cosine_similarity(
        centered_prediction, centered_target, dim=-1
    ).mean()
    real_smooth_l1 = F.smooth_l1_loss(prediction, target)
    real_nce = _symmetric_info_nce(prediction, target, temperature)
    shuffled_cosines = []
    shuffled_smooth_l1 = []
    shuffled_nce = []
    for repeat in range(8):
        generator = torch.Generator().manual_seed(shuffle_seed + repeat)
        permutation = torch.randperm(count, generator=generator)
        shuffled = target[permutation]
        shuffled_cosines.append(
            F.cosine_similarity(prediction, shuffled, dim=-1).mean()
        )
        shuffled_smooth_l1.append(F.smooth_l1_loss(prediction, shuffled))
        shuffled_nce.append(
            _symmetric_info_nce(prediction, shuffled, temperature)
        )
    shuffled_cosine = torch.stack(shuffled_cosines).mean()
    shuffled_smooth = torch.stack(shuffled_smooth_l1).mean()
    shuffled_contrastive = torch.stack(shuffled_nce).mean()
    similarities = F.normalize(centered_prediction, dim=-1) @ F.normalize(
        centered_target, dim=-1
    ).T
    retrieval = (
        similarities.argmax(dim=1) == torch.arange(count)
    ).float().mean()
    return {
        "count": count,
        "raw_cosine": float(real_cosine.item()),
        "centered_cosine": float(centered_cosine.item()),
        "real_minus_shuffle_cosine": float(
            (real_cosine - shuffled_cosine).item()
        ),
        "shuffle_minus_real_smooth_l1": float(
            (shuffled_smooth - real_smooth_l1).item()
        ),
        "shuffle_minus_real_nce": float(
            (shuffled_contrastive - real_nce).item()
        ),
        "retrieval_top1": float(retrieval.item()),
        "chance": 1.0 / count,
        "effective_rank": _effective_rank(prediction),
        "target_effective_rank": _effective_rank(target),
        "channel_std": float(
            prediction.std(dim=0, unbiased=False).mean().item()
        ),
        "target_channel_std": float(
            target.std(dim=0, unbiased=False).mean().item()
        ),
    }


def _load_config(checkpoint: dict[str, object]) -> TrainConfig:
    known = {field.name for field in fields(TrainConfig)}
    values = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in known
    }
    return TrainConfig(**values)


def _build_model(
    config: TrainConfig,
    dimensions: tuple[int, int, int],
    num_speakers: int,
    num_classes: int,
    device: torch.device,
) -> MissingM3GraphModel:
    adim, tdim, vdim = dimensions
    return MissingM3GraphModel(
        config.base_model,
        adim,
        tdim,
        vdim,
        config.hidden,
        config.hidden // 2,
        n_speakers=num_speakers,
        window_past=config.window_past,
        window_future=config.window_future,
        n_classes=num_classes,
        dropout=config.dropout,
        time_attn=config.time_attention,
        no_cuda=device.type != "cuda",
        latent_dim=config.latent_dim,
        num_experts=config.num_experts,
        top_k=config.top_k,
        projector_dropout=config.projector_dropout,
        predictor_dropout=config.predictor_dropout,
        fusion_type=config.fusion_type,
        local_context_residual=config.local_context_residual,
        local_fusion_hidden_dim=config.local_fusion_hidden_dim,
        local_fusion_dropout=config.local_fusion_dropout,
        graph_branch_mode=config.graph_branch_mode,
        mmoe_variant=config.mmoe_variant,
        classification_completion=config.classification_completion,
        representation_type=config.representation_type,
        node_interaction_residual=config.node_interaction_residual,
        readout_type=config.readout_type,
        readout_rank=config.readout_rank,
        recurrent_padding_mode=config.recurrent_padding_mode,
        postgraph_sequence_mode=config.postgraph_sequence_mode,
        graph_message_calibration=config.graph_message_calibration,
    ).to(device)


@torch.no_grad()
def audit(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = _load_config(checkpoint)
    contract = _resolve_task_contract(config.dataset, config.mosi_task_mode)
    roots = [
        str(Path(args.feature_root) / name)
        for name in (args.audio_feature, args.text_feature, args.video_feature)
    ]
    loaders = get_loaders(
        audio_root=roots[0],
        text_root=roots[1],
        video_root=roots[2],
        num_folder=int(contract["num_folds"]),
        dataset=config.dataset,
        batch_size=config.batch_size,
        num_workers=0,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol,
    )
    _, _, test_loaders, adim, tdim, vdim = loaders
    dimensions = (adim, tdim, vdim)
    device = torch.device(args.device)
    model = _build_model(
        config,
        dimensions,
        num_speakers=int(contract["num_speakers"]),
        num_classes=int(contract["num_classes"]),
        device=device,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    schedule = _build_schedule(config, "test", args.missing_rate)
    collected = {
        branch: {name: {"prediction": [], "target": []} for name in MODALITIES}
        for branch in ("regression", "contrastive")
    }
    for raw in test_loaders[config.fold - 1]:
        data = _move_batch(raw, device)
        view = _prepare_view(data, schedule, epoch=0, dimensions=dimensions)
        _, _, _, predictions = model(
            [view["incomplete"]],
            view["availability"],
            view["qmask"],
            view["umask"],
            view["lengths"],
            predict_missing=True,
        )
        teacher = model.encode_teacher_targets([view["complete"]])
        for target_index, name in enumerate(MODALITIES):
            selected = predictions.target_mask[..., target_index]
            for branch, value in (
                ("regression", predictions.reg_predictions),
                ("contrastive", predictions.cl_predictions),
            ):
                collected[branch][name]["prediction"].append(
                    value[..., target_index, :][selected].cpu()
                )
                collected[branch][name]["target"].append(
                    teacher[name][selected].cpu()
                )
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "missing_rate": args.missing_rate,
        "seed": config.seed,
        "jepa_contrastive_source": config.jepa_contrastive_source,
        "branches": {},
    }
    for branch, modalities in collected.items():
        result["branches"][branch] = {}
        for target_index, name in enumerate(MODALITIES):
            prediction = torch.cat(modalities[name]["prediction"])
            target = torch.cat(modalities[name]["target"])
            result["branches"][branch][name] = _metrics(
                prediction,
                target,
                config.temperature,
                shuffle_seed=config.seed + 100 * target_index,
            )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--audio-feature", required=True)
    parser.add_argument("--text-feature", required=True)
    parser.add_argument("--video-feature", required=True)
    parser.add_argument("--missing-rate", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
