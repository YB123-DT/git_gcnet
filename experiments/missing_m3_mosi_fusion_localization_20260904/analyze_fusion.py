"""Localize where non-text MOSI evidence is lost in Slot Missing-M3.

This is an inference-only checkpoint diagnostic.  It evaluates fixed modality
patterns and correspondence-breaking shuffles while capturing representations
along the unchanged GCNet path.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from gcnet_modality_jepa.train_gcnet import get_loaders
from gcnet_missing_m3.model import MissingM3GraphModel
from gcnet_missing_m3.train_gcnet import (
    TrainConfig,
    _build_schedule,
    _collect_predictions,
    _metrics,
    _move_batch,
    _prepare_view,
    _resolve_task_contract,
)


PATTERNS: Mapping[str, tuple[int, int, int]] = {
    "A": (1, 0, 0),
    "T": (0, 1, 0),
    "V": (0, 0, 1),
    "AT": (1, 1, 0),
    "AV": (1, 0, 1),
    "TV": (0, 1, 1),
    "ATV": (1, 1, 1),
}

VARIANTS: Mapping[str, tuple[str, tuple[int, ...]]] = {
    **{name: (name, ()) for name in PATTERNS},
    "AT_shuffle_A": ("AT", (0,)),
    "TV_shuffle_V": ("TV", (2,)),
    "AV_shuffle_A": ("AV", (0,)),
    "AV_shuffle_V": ("AV", (2,)),
    "ATV_shuffle_A": ("ATV", (0,)),
    "ATV_shuffle_V": ("ATV", (2,)),
    "ATV_shuffle_AV": ("ATV", (0, 2)),
}

PAIRS: Mapping[str, tuple[str, str]] = {
    "audio_given_text": ("T", "AT"),
    "visual_given_text": ("T", "TV"),
    "audio_visual_given_text": ("T", "ATV"),
    "audio_correspondence_given_text": ("AT_shuffle_A", "AT"),
    "visual_correspondence_given_text": ("TV_shuffle_V", "TV"),
    "audio_correspondence_in_atv": ("ATV_shuffle_A", "ATV"),
    "visual_correspondence_in_atv": ("ATV_shuffle_V", "ATV"),
    "joint_correspondence_in_atv": ("ATV_shuffle_AV", "ATV"),
}


def _load_config(checkpoint: Mapping[str, object]) -> TrainConfig:
    known = {field.name for field in fields(TrainConfig)}
    raw = checkpoint.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint is missing its training config")
    return TrainConfig(**{key: value for key, value in raw.items() if key in known})


def _build_model(
    config: TrainConfig,
    dimensions: tuple[int, int, int],
    device: torch.device,
) -> MissingM3GraphModel:
    contract = _resolve_task_contract(config.dataset, config.mosi_task_mode)
    adim, tdim, vdim = dimensions
    optional = {
        "local_context_residual": False,
        "local_fusion_hidden_dim": 256,
        "local_fusion_dropout": 0.2,
        "graph_branch_mode": "both",
        "mmoe_variant": "dual-gate",
        "target_private_rank": 0,
        "classification_completion": False,
        "representation_type": "slot",
        "node_interaction_residual": False,
        "readout_type": "shared",
        "readout_rank": 8,
        "recurrent_padding_mode": "legacy",
        "postgraph_sequence_mode": "independent",
        "graph_message_calibration": "none",
    }
    keyword = {
        "n_speakers": int(contract["num_speakers"]),
        "window_past": config.window_past,
        "window_future": config.window_future,
        "n_classes": int(contract["num_classes"]),
        "dropout": config.dropout,
        "time_attn": config.time_attention,
        "no_cuda": device.type != "cuda",
        "latent_dim": config.latent_dim,
        "num_experts": config.num_experts,
        "top_k": config.top_k,
        "projector_dropout": config.projector_dropout,
        "predictor_dropout": config.predictor_dropout,
        "fusion_type": config.fusion_type,
        **{
            name: getattr(config, name, default)
            for name, default in optional.items()
        },
    }
    supported = inspect.signature(MissingM3GraphModel.__init__).parameters
    keyword = {name: value for name, value in keyword.items() if name in supported}
    return MissingM3GraphModel(
        config.base_model,
        adim,
        tdim,
        vdim,
        config.hidden,
        config.hidden // 2,
        **keyword,
    ).to(device)


def _load_graphconv_compatible_state(
    model: MissingM3GraphModel,
    state: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Translate the PyG 2.x GraphConv rename without ignoring parameters."""
    rename = {
        ".conv2.lin_l.weight": ".conv2.lin_rel.weight",
        ".conv2.lin_l.bias": ".conv2.lin_rel.bias",
        ".conv2.lin_r.weight": ".conv2.lin_root.weight",
    }
    target = model.state_dict()
    translated: dict[str, torch.Tensor] = {}
    changed = []
    for source_name, value in state.items():
        target_name = source_name
        for old, new in rename.items():
            if source_name.endswith(old):
                target_name = source_name[: -len(old)] + new
                changed.append(source_name + " -> " + target_name)
                break
        if target_name in translated:
            raise ValueError("state translation produced duplicate key: " + target_name)
        if target_name not in target:
            raise ValueError("checkpoint contains unsupported key: " + target_name)
        if value.shape != target[target_name].shape:
            raise ValueError("checkpoint shape mismatch: " + target_name)
        translated[target_name] = value
    missing = sorted(set(target) - set(translated))
    if missing:
        raise ValueError("checkpoint is missing keys: " + ", ".join(missing))
    model.load_state_dict(translated, strict=True)
    return tuple(changed)


def _tensor_from_recurrent_output(output: object) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if isinstance(value, torch.nn.utils.rnn.PackedSequence):
        value, _ = torch.nn.utils.rnn.pad_packed_sequence(value)
    if not torch.is_tensor(value):
        raise TypeError("recurrent hook did not receive a tensor")
    return value.detach()


class LayerCapture:
    def __init__(self, model: MissingM3GraphModel) -> None:
        self.values: Dict[str, torch.Tensor] = {}
        self.handles = [
            model.observed_set.register_forward_hook(self._observed),
        ]
        recurrent = model.lstm if model.base_model == "LSTM" else model.gru
        self.handles.append(recurrent.register_forward_hook(self._pregraph))
        if model.graph_branch_mode in {"both", "temporal-only"}:
            self.handles.append(
                model.graph_net_temporal.register_forward_hook(
                    self._named("temporal")
                )
            )
        if model.graph_branch_mode in {"both", "speaker-only"}:
            self.handles.append(
                model.graph_net_speaker.register_forward_hook(
                    self._named("speaker")
                )
            )

    def _observed(self, _module, _inputs, output) -> None:
        encoded = output[0]
        if not torch.is_tensor(encoded):
            raise TypeError("Slot encoder output must be a tensor")
        self.values["slot"] = encoded.detach()

    def _pregraph(self, _module, _inputs, output) -> None:
        self.values["pregraph"] = _tensor_from_recurrent_output(output)

    def _named(self, name: str):
        def capture(_module, _inputs, output) -> None:
            if not torch.is_tensor(output):
                raise TypeError(name + " graph output must be a tensor")
            self.values[name] = output.detach()

        return capture

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _fixed_view(
    base: Mapping[str, object],
    dimensions: tuple[int, int, int],
    pattern: tuple[int, int, int],
    shuffled_modalities: tuple[int, ...],
    seed: int,
) -> dict[str, object]:
    complete = base["complete"].clone()
    umask = base["umask"]
    valid = umask.T.bool()
    offsets = np.cumsum((0, *dimensions))
    for modality in shuffled_modalities:
        start, end = int(offsets[modality]), int(offsets[modality + 1])
        block = complete[..., start:end]
        flat = block[valid].clone()
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 1009 * (modality + 1)
        )
        permutation = torch.randperm(flat.shape[0], generator=generator)
        block[valid] = flat[permutation.to(flat.device)]

    availability = complete.new_zeros(*complete.shape[:2], 3)
    availability[valid] = complete.new_tensor(pattern)
    widths = torch.tensor(dimensions, device=complete.device)
    expanded = torch.repeat_interleave(availability, widths, dim=-1)
    return {
        **base,
        "complete": complete,
        "incomplete": complete * expanded,
        "availability": availability,
    }


def _flatten(value: torch.Tensor, umask: torch.Tensor) -> torch.Tensor:
    if value.shape[:2] != umask.T.shape:
        raise ValueError("captured layer is not aligned with [L,B]")
    return value[umask.T.bool()].float().cpu()


def _layer_delta(base: torch.Tensor, treatment: torch.Tensor) -> dict[str, float]:
    if base.shape != treatment.shape:
        raise ValueError("paired representations have different shapes")
    difference = (treatment - base).norm(dim=-1)
    denominator = base.norm(dim=-1).mean().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(base, treatment, dim=-1)
    return {
        "normalized_l2": float((difference.mean() / denominator).item()),
        "mean_cosine": float(cosine.mean().item()),
    }


@torch.no_grad()
def analyze_checkpoint(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = _load_config(checkpoint)
    if config.dataset != "CMUMOSI" or config.fusion_type != "slot":
        raise ValueError("this diagnostic requires a CMUMOSI Slot checkpoint")
    roots = [
        str(Path(args.feature_root) / name)
        for name in (args.audio_feature, args.text_feature, args.video_feature)
    ]
    loaders = get_loaders(
        audio_root=roots[0],
        text_root=roots[1],
        video_root=roots[2],
        num_folder=1,
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
    model = _build_model(config, dimensions, device)
    translated_keys = _load_graphconv_compatible_state(model, checkpoint["model"])
    model.eval()
    capture = LayerCapture(model)
    schedule = _build_schedule(config, "test", 0.0)

    batch_cache = []
    for raw in test_loaders[0]:
        data = _move_batch(raw, device)
        batch_cache.append(_prepare_view(data, schedule, 0, dimensions))

    outputs: dict[str, dict[str, object]] = {}
    for variant, (pattern_name, shuffled) in VARIANTS.items():
        predictions: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        continuous: list[np.ndarray] = []
        layers: dict[str, list[torch.Tensor]] = {}
        for batch_index, base in enumerate(batch_cache):
            view = _fixed_view(
                base,
                dimensions,
                PATTERNS[pattern_name],
                shuffled,
                config.seed + 7919 * batch_index,
            )
            capture.clear()
            logits, hidden, _, missing = model(
                [view["incomplete"]],
                view["availability"],
                view["qmask"],
                view["umask"],
                view["lengths"],
                predict_missing=False,
            )
            if missing is not None:
                raise RuntimeError("inference unexpectedly returned JEPA predictions")
            predicted, expected, raw_labels = _collect_predictions(
                config.dataset,
                logits,
                view["labels"],
                view["umask"],
                config.mosi_task_mode,
            )
            predictions.append(predicted)
            labels.append(expected)
            continuous.append(raw_labels)
            capture.values["final"] = hidden.detach()
            capture.values["prediction"] = logits.detach()
            for name, value in capture.values.items():
                layers.setdefault(name, []).append(_flatten(value, view["umask"]))
        prediction = np.concatenate(predictions)
        expected = np.concatenate(labels)
        raw_labels = np.concatenate(continuous)
        outputs[variant] = {
            "metrics": _metrics(
                config.dataset, expected, prediction, config.mosi_task_mode
            ),
            "prediction": torch.from_numpy(prediction).float().reshape(-1, 1),
            "labels": torch.from_numpy(raw_labels).float(),
            "layers": {
                name: torch.cat(values, dim=0) for name, values in layers.items()
            },
        }
    capture.close()

    comparisons: dict[str, object] = {}
    for name, (base_name, treatment_name) in PAIRS.items():
        base = outputs[base_name]
        treatment = outputs[treatment_name]
        base_prediction = base["prediction"]
        treatment_prediction = treatment["prediction"]
        raw_labels = treatment["labels"]
        selected = raw_labels.ne(0)
        base_correct = base_prediction.flatten()[selected].gt(0).eq(
            raw_labels[selected].gt(0)
        )
        treatment_correct = treatment_prediction.flatten()[selected].gt(0).eq(
            raw_labels[selected].gt(0)
        )
        comparisons[name] = {
            "base": base_name,
            "treatment": treatment_name,
            "weighted_f1_delta_points": 100.0
            * (
                treatment["metrics"]["weighted_f1"]
                - base["metrics"]["weighted_f1"]
            ),
            "paired_accuracy_delta_points": 100.0
            * float(
                treatment_correct.float().mean().item()
                - base_correct.float().mean().item()
            ),
            "prediction_mean_absolute_change": float(
                (treatment_prediction - base_prediction).abs().mean().item()
            ),
            "layers": {
                layer: _layer_delta(base["layers"][layer], value)
                for layer, value in treatment["layers"].items()
            },
        }

    serializable_variants = {
        name: {"metrics": value["metrics"]} for name, value in outputs.items()
    }
    formal_path = checkpoint_path.parent / "predictions_miss_0p0.npz"
    formal = np.load(formal_path)
    replay_prediction = outputs["ATV"]["prediction"].flatten().numpy()
    replay_labels = outputs["ATV"]["labels"].numpy()
    if replay_prediction.shape != formal["predictions"].shape:
        raise ValueError("formal and replay predictions are not aligned")
    if replay_labels.shape != formal["labels"].shape:
        raise ValueError("formal and replay labels are not aligned")
    formal_replay = {
        "formal_prediction_path": str(formal_path),
        "label_max_absolute_error": float(
            np.max(np.abs(replay_labels - formal["labels"]))
        ),
        "prediction_mean_absolute_error": float(
            np.mean(np.abs(replay_prediction - formal["predictions"]))
        ),
        "prediction_sign_agreement": float(
            np.mean((replay_prediction > 0) == (formal["predictions"] > 0))
        ),
        "formal_weighted_f1": _metrics(
            config.dataset,
            formal["labels"],
            formal["predictions"],
            config.mosi_task_mode,
        )["weighted_f1"],
        "replay_weighted_f1": outputs["ATV"]["metrics"]["weighted_f1"],
    }
    return {
        "seed": config.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "state_key_translations": list(translated_keys),
        "test_batches": len(batch_cache),
        "variants": serializable_variants,
        "comparisons": comparisons,
        "formal_replay": formal_replay,
    }


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(array.mean())


def summarize(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    variants = {
        variant: {
            "weighted_f1_percent_mean": 100.0
            * _mean(
                result["variants"][variant]["metrics"]["weighted_f1"]
                for result in results
            ),
            "weighted_f1_percent_by_seed": {
                str(result["seed"]): 100.0
                * result["variants"][variant]["metrics"]["weighted_f1"]
                for result in results
            },
        }
        for variant in VARIANTS
    }
    comparisons = {}
    for comparison in PAIRS:
        rows = [result["comparisons"][comparison] for result in results]
        layer_names = rows[0]["layers"]
        comparisons[comparison] = {
            "weighted_f1_delta_points_mean": _mean(
                row["weighted_f1_delta_points"] for row in rows
            ),
            "weighted_f1_delta_points_by_seed": {
                str(result["seed"]): result["comparisons"][comparison][
                    "weighted_f1_delta_points"
                ]
                for result in results
            },
            "paired_accuracy_delta_points_mean": _mean(
                row["paired_accuracy_delta_points"] for row in rows
            ),
            "prediction_mean_absolute_change_mean": _mean(
                row["prediction_mean_absolute_change"] for row in rows
            ),
            "layers": {
                layer: {
                    "normalized_l2_mean": _mean(
                        row["layers"][layer]["normalized_l2"] for row in rows
                    ),
                    "mean_cosine": _mean(
                        row["layers"][layer]["mean_cosine"] for row in rows
                    ),
                }
                for layer in layer_names
            },
        }
    return {
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "selection": "existing validation-selected checkpoints",
        "training_performed": False,
        "seeds": [result["seed"] for result in results],
        "variants": variants,
        "comparisons": comparisons,
        "formal_replay": {
            "formal_weighted_f1_percent_mean": 100.0
            * _mean(result["formal_replay"]["formal_weighted_f1"] for result in results),
            "replay_weighted_f1_percent_mean": 100.0
            * _mean(result["formal_replay"]["replay_weighted_f1"] for result in results),
            "prediction_mean_absolute_error_mean": _mean(
                result["formal_replay"]["prediction_mean_absolute_error"]
                for result in results
            ),
            "prediction_sign_agreement_mean": _mean(
                result["formal_replay"]["prediction_sign_agreement"]
                for result in results
            ),
            "label_max_absolute_error": max(
                result["formal_replay"]["label_max_absolute_error"]
                for result in results
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--audio-feature", default="wav2vec-large-c-UTT")
    parser.add_argument("--text-feature", default="deberta-large-4-UTT")
    parser.add_argument("--video-feature", default="manet_UTT")
    parser.add_argument("--seeds", nargs="+", type=int, default=[66, 67, 68, 69, 70])
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.checkpoint_root)
    results = []
    for seed in args.seeds:
        checkpoint = root / ("seed_" + str(seed)) / "best.pt"
        print("analyzing", checkpoint, flush=True)
        results.append(analyze_checkpoint(args, checkpoint))
    payload = {"summary": summarize(results), "per_seed": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
