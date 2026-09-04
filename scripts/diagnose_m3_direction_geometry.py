"""Diagnose six-direction gradient geometry and Top-K routing in Missing-M3."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gcnet_missing_m3.model import MODALITIES, MissingM3GraphModel
from gcnet_missing_m3.train_gcnet import (
    MISSING_RATES,
    _move_batch,
    _prepare_view,
    _resolve_task_contract,
    get_loaders,
)
from gcnet_modality_jepa.mask_schedule import ConversationMaskSchedule
from gcnet_modality_jepa.protocol import SeedBundle


def _compatibility_state(state):
    return {
        name.replace(".conv2.lin_l.", ".conv2.lin_rel.").replace(
            ".conv2.lin_r.", ".conv2.lin_root."
        ): value
        for name, value in state.items()
    }


def _cosine(left, right):
    denominator = left.norm() * right.norm()
    return None if float(denominator) == 0 else float(torch.dot(left, right) / denominator)


def _mean(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _build_model(config, dimensions, device):
    shape = _resolve_task_contract(config["dataset"], config.get("mosi_task_mode", "regression"))
    model = MissingM3GraphModel(
        config.get("base_model", "LSTM"),
        *dimensions,
        config.get("hidden", 200),
        config.get("hidden", 200) // 2,
        n_speakers=int(shape["num_speakers"]),
        window_past=config.get("window_past", 2),
        window_future=config.get("window_future", 2),
        n_classes=int(shape["num_classes"]),
        dropout=config.get("dropout", 0.5),
        time_attn=config.get("time_attention", False),
        no_cuda=device.type != "cuda",
        latent_dim=config.get("latent_dim", 256),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        projector_dropout=config.get("projector_dropout", 0.1),
        predictor_dropout=config.get("predictor_dropout", 0.1),
        fusion_type=config.get("fusion_type", "slot"),
        graph_branch_mode=config.get("graph_branch_mode", "both"),
        mmoe_variant=config.get("mmoe_variant", "dual-gate"),
        target_private_rank=config.get("target_private_rank", 0),
        representation_type=config.get("representation_type", "slot"),
        recurrent_padding_mode=config.get("recurrent_padding_mode", "legacy"),
        postgraph_sequence_mode=config.get("postgraph_sequence_mode", "independent"),
        graph_message_calibration=config.get("graph_message_calibration", "none"),
    ).to(device)
    return model


def _direction_loss(regression, contrastive, target, temperature):
    reg = F.smooth_l1_loss(regression, target)
    if target.shape[0] < 2:
        return reg
    prediction = F.normalize(contrastive, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(target.shape[0], device=target.device)
    cl = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return 0.5 * reg + 0.5 * cl


def _routing_counts(mmoe, conditioned):
    conditioned = conditioned
    result = torch.zeros(mmoe.num_experts, device=conditioned.device)
    for gate in (mmoe.reg_gate, mmoe.cl_gate):
        indices = gate(conditioned).topk(mmoe.top_k, dim=-1).indices
        result.scatter_add_(0, indices.reshape(-1), torch.ones(indices.numel(), device=result.device))
    return result


def _pair_metrics(records):
    directions = sorted(records)
    pairs = []
    for index, left_name in enumerate(directions):
        for right_name in directions[index + 1 :]:
            left, right = records[left_name], records[right_name]
            left_source, left_target = left_name.split("->")
            right_source, right_target = right_name.split("->")
            same_target = left_target == right_target
            global_cosine = _cosine(left["gradient"], right["gradient"])
            common = [
                expert
                for expert in range(len(left["expert_gradients"]))
                if left["expert_gradients"][expert].norm() > 1e-12
                and right["expert_gradients"][expert].norm() > 1e-12
            ]
            common_cosine = None
            if common:
                common_cosine = _cosine(
                    torch.cat([left["expert_gradients"][expert] for expert in common]),
                    torch.cat([right["expert_gradients"][expert] for expert in common]),
                )
            left_route = left["routing"] / left["routing"].sum().clamp_min(1)
            right_route = right["routing"] / right["routing"].sum().clamp_min(1)
            route_overlap = float(torch.minimum(left_route, right_route).sum())
            left_mass = torch.tensor([value.norm() for value in left["expert_gradients"]])
            right_mass = torch.tensor([value.norm() for value in right["expert_gradients"]])
            left_mass /= left_mass.sum().clamp_min(1e-12)
            right_mass /= right_mass.sum().clamp_min(1e-12)
            pairs.append(
                {
                    "pair": f"{left_name}|{right_name}",
                    "same_target": same_target,
                    "same_source": left_source == right_source,
                    "global_cosine": global_cosine,
                    "common_expert_cosine": common_cosine,
                    "common_expert_count": len(common),
                    "routing_overlap": route_overlap,
                    "gradient_mass_overlap": float(torch.minimum(left_mass, right_mass).sum()),
                }
            )
    groups = {}
    for label, selected in (
        ("same_target", [item for item in pairs if item["same_target"]]),
        ("cross_target", [item for item in pairs if not item["same_target"]]),
    ):
        groups[label] = {
            "pair_count": len(selected),
            "global_cosine_mean": _mean([item["global_cosine"] for item in selected]),
            "common_expert_cosine_mean": _mean([item["common_expert_cosine"] for item in selected]),
            "routing_overlap_mean": _mean([item["routing_overlap"] for item in selected]),
            "gradient_mass_overlap_mean": _mean([item["gradient_mass_overlap"] for item in selected]),
        }
    return pairs, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint["config"])
    device = torch.device(args.device)
    feature_names = ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
    roots = [args.feature_root / name for name in feature_names]
    shape = _resolve_task_contract(config["dataset"], config.get("mosi_task_mode", "regression"))
    loaders = get_loaders(
        *(str(root) for root in roots),
        num_folder=int(shape["num_folds"]),
        dataset=config["dataset"],
        batch_size=config.get("batch_size", 32),
        num_workers=0,
        seed=config.get("seed", 66),
        validation_fraction=config.get("validation_fraction", 0.1),
        evaluation_protocol=config.get("evaluation_protocol", "official"),
    )
    train_loaders, _, _, *dimensions = loaders
    dimensions = tuple(dimensions)
    model = _build_model(config, dimensions, device)
    model.load_state_dict(_compatibility_state(checkpoint["model"]), strict=True)
    model.eval()
    predictor = model.missing_predictor
    expert_parameter_groups = [list(expert.parameters()) for expert in predictor.mmoe.experts]
    expert_parameters = [parameter for group in expert_parameter_groups for parameter in group]
    epoch = int(checkpoint.get("epoch", 1)) - 1
    records = defaultdict(lambda: {"count": 0, "gradient": None, "expert_gradients": None, "routing": None})
    for rate in MISSING_RATES[1:]:
        schedule = ConversationMaskSchedule(
            dataset=config["dataset"], split="train", fold=config.get("fold", 1),
            requested_missing_rate=rate,
            mask_seed=SeedBundle(config.get("seed", 66)).derive("missing_mask"),
            freeze_evaluation=False,
        )
        for raw in train_loaders[config.get("fold", 1) - 1]:
            data = _move_batch(raw, device)
            view = _prepare_view(data, schedule, epoch, dimensions)
            with torch.no_grad():
                encoded, latents = model.observed_set(view["incomplete"], view["availability"], view["umask"])
                hidden = model.encode_hidden([encoded], view["qmask"], view["umask"], view["lengths"])
                teacher = model.encode_teacher_targets([view["complete"]])
            flat_valid = view["umask"].T.bool().reshape(-1)
            flat_availability = view["availability"].reshape(-1, 3).bool()
            singleton = flat_availability.sum(-1).eq(1)
            flat_hidden = hidden.detach().reshape(-1, hidden.shape[-1])
            for source_index, source_name in enumerate(MODALITIES):
                source_latent = latents[source_name].detach().reshape(-1, predictor.latent_dim)
                for target_index, target_name in enumerate(MODALITIES):
                    if source_index == target_index:
                        continue
                    selected = flat_valid & singleton & flat_availability[:, source_index] & ~flat_availability[:, target_index]
                    indices = selected.nonzero(as_tuple=False).flatten()
                    if indices.numel() < 2:
                        continue
                    source = source_latent[indices]
                    context = flat_hidden[indices]
                    reg, cl = predictor.direction_forward(source, context, source_index, target_index)
                    target = teacher[target_name].reshape(-1, predictor.latent_dim)[indices].detach()
                    loss = _direction_loss(reg, cl, target, config.get("temperature", 0.03))
                    gradients = torch.autograd.grad(loss, expert_parameters, retain_graph=False, allow_unused=True)
                    cursor = 0
                    blocks = []
                    for parameters in expert_parameter_groups:
                        values = []
                        for parameter in parameters:
                            gradient = gradients[cursor]
                            cursor += 1
                            values.append(torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1))
                        blocks.append(torch.cat(values).cpu() * indices.numel())
                    conditioned = predictor.input_norm(source + predictor.context_projection(context))
                    conditioned = conditioned + predictor.mmoe.source_embedding.weight[source_index] + predictor.mmoe.target_embedding.weight[target_index]
                    route = _routing_counts(predictor.mmoe, conditioned).detach().cpu()
                    name = f"{source_name[0].upper()}->{target_name[0].upper()}"
                    record = records[(str(rate), name)]
                    if record["gradient"] is None:
                        record["expert_gradients"] = [torch.zeros_like(block) for block in blocks]
                        record["gradient"] = torch.zeros(sum(block.numel() for block in blocks))
                        record["routing"] = torch.zeros_like(route)
                    record["count"] += indices.numel()
                    for expert, block in enumerate(blocks):
                        record["expert_gradients"][expert] += block
                    record["gradient"] += torch.cat(blocks)
                    record["routing"] += route

    output = {"checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"), "rates": {}}
    aggregate = {}
    for rate in MISSING_RATES[1:]:
        rate_records = {name: records[(str(rate), name)] for name in ("A->T", "A->V", "T->A", "T->V", "V->A", "V->T") if records[(str(rate), name)]["count"] > 0}
        pairs, groups = _pair_metrics(rate_records)
        output["rates"][str(rate)] = {
            "direction_counts": {name: value["count"] for name, value in rate_records.items()},
            "pairs": pairs,
            "groups": groups,
        }
        for name, value in rate_records.items():
            if name not in aggregate:
                aggregate[name] = {"count": 0, "gradient": torch.zeros_like(value["gradient"]), "expert_gradients": [torch.zeros_like(block) for block in value["expert_gradients"]], "routing": torch.zeros_like(value["routing"])}
            aggregate[name]["count"] += value["count"]
            aggregate[name]["gradient"] += value["gradient"]
            aggregate[name]["routing"] += value["routing"]
            for expert, block in enumerate(value["expert_gradients"]):
                aggregate[name]["expert_gradients"][expert] += block
    pairs, groups = _pair_metrics(aggregate)
    output["aggregate"] = {"direction_counts": {name: value["count"] for name, value in aggregate.items()}, "pairs": pairs, "groups": groups}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["aggregate"]["groups"], indent=2))


if __name__ == "__main__":
    main()
