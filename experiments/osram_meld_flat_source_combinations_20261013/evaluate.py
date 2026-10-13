"""Evaluate all non-empty Local/Base/Gap combinations in one frozen Flat pass.

INTERNAL DIAGNOSTIC ONLY.  Memory is scanned exactly once for each batch.  The
seven predictions differ only in which captured Flat-readout source groups are
zeroed before the original adapter, normalization, and classifier.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SOURCE_COMBINATIONS = {
    "L": (True, False, False),
    "B": (False, True, False),
    "G": (False, False, True),
    "LB": (True, True, False),
    "LG": (True, False, True),
    "BG": (False, True, True),
    "LBG": (True, True, True),
}


def runtime_config_payload(source: dict) -> dict:
    """Adapt a historical config to the current constructor without changing masks.

    The exact historical schedule is instantiated explicitly below.  Replacing
    this one constructor-only value avoids re-enabling the retired protocol in
    the normal training surface.
    """
    payload = dict(source)
    if payload.get("missing_mask_protocol") == "gcnet-legacy-fixed-v1":
        payload["missing_mask_protocol"] = "conversation-mask-v2"
    return payload


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_flat_sources(
    local: torch.Tensor,
    flat_input: torch.Tensor,
    umask: torch.Tensor,
    *,
    local_dim: int,
    context_dim: int,
    combination: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask the three semantic source groups already presented to Flat."""
    if combination not in SOURCE_COMBINATIONS:
        raise ValueError(f"Unknown source combination: {combination}")
    expected_width = local_dim + 4 * context_dim
    if local.shape[-1] != local_dim or flat_input.shape[-1] != expected_width:
        raise ValueError(
            f"Unexpected Flat widths: local={local.shape[-1]}, "
            f"input={flat_input.shape[-1]}, expected={local_dim}/{expected_width}"
        )
    keep_local, keep_base, keep_gap = SOURCE_COMBINATIONS[combination]
    valid = umask.T.bool().unsqueeze(-1)
    selected_local = local if keep_local else torch.zeros_like(local)
    selected_input = flat_input.clone()
    if not keep_local:
        selected_input[..., :local_dim] = 0
    if not keep_base:
        selected_input[..., local_dim : local_dim + context_dim] = 0
    if not keep_gap:
        selected_input[..., local_dim + context_dim :] = 0
    return (
        torch.where(valid, selected_local, torch.zeros_like(selected_local)),
        torch.where(valid, selected_input, torch.zeros_like(selected_input)),
    )


def replay_flat_hidden(
    *,
    local_skip,
    emotion_adapter,
    emotion_norm,
    local: torch.Tensor,
    flat_input: torch.Tensor,
    umask: torch.Tensor,
    local_dim: int,
    context_dim: int,
    combination: str,
) -> torch.Tensor:
    selected_local, selected_input = select_flat_sources(
        local,
        flat_input,
        umask,
        local_dim=local_dim,
        context_dim=context_dim,
        combination=combination,
    )
    keep_local = SOURCE_COMBINATIONS[combination][0]
    skip = local_skip(selected_local) if keep_local else torch.zeros(
        (*selected_local.shape[:-1], local_skip.out_features),
        dtype=selected_local.dtype,
        device=selected_local.device,
    )
    hidden = emotion_norm(skip + emotion_adapter(selected_input))
    valid = umask.T.bool().unsqueeze(-1)
    return torch.where(valid, hidden, torch.zeros_like(hidden))


class FlatReadoutCapture:
    """Capture the exact tensors consumed by the original Flat readout."""

    def __init__(self, osram) -> None:
        self.osram = osram
        self.local_calls = 0
        self.adapter_calls = 0

    def _capture_local(self, _module, args) -> None:
        self.local = args[0].detach().clone()
        self.local_calls += 1

    def _capture_adapter(self, _module, args) -> None:
        self.flat_input = args[0].detach().clone()
        self.adapter_calls += 1

    def __enter__(self):
        self.handles = [
            self.osram.local_skip.register_forward_pre_hook(self._capture_local),
            self.osram.emotion_adapter.register_forward_pre_hook(self._capture_adapter),
        ]
        return self

    def __exit__(self, *_args) -> None:
        for handle in self.handles:
            handle.remove()


def _build_model(config, dimensions, device):
    from gcnet_missing_m3.model import MissingM3GraphModel
    from gcnet_missing_m3.train_gcnet import _dataset_shape
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import set_random_seed

    shape = _dataset_shape(config.dataset)
    set_random_seed(SeedBundle(config.seed).derive(f"missing_m3_model_init:fold:{config.fold}"))
    keywords = {
        name: getattr(config, name)
        for name in inspect.signature(MissingM3GraphModel).parameters
        if hasattr(config, name)
    }
    for name in ("base_model", "adim", "tdim", "vdim", "D_e", "graph_hidden_size"):
        keywords.pop(name, None)
    keywords.update(
        n_speakers=int(shape["num_speakers"]),
        n_classes=int(shape["num_classes"]),
        time_attn=config.time_attention,
        no_cuda=device.type != "cuda",
        complete_state_jepa=config.training_objective == "complete-state",
        write_state_completion=config.training_objective == "write-state",
        future_state_jepa=config.training_objective == "future-state",
    )
    return MissingM3GraphModel(
        config.base_model,
        *dimensions,
        config.hidden,
        config.hidden // 2,
        **keywords,
    ).to(device).eval().requires_grad_(False)


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "unweighted_accuracy": float(
            recall_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)
        ),
    }


def evaluate_rate(model, loader, config, dimensions, schedule) -> tuple[dict, dict]:
    from gcnet_missing_m3.train_gcnet import (
        _collect_predictions,
        _move_batch,
        _prepare_view,
    )

    labels_all: list[np.ndarray] = []
    predictions = {name: [] for name in SOURCE_COMBINATIONS}
    max_full_logit_error = 0.0
    scan_count = 0
    with torch.no_grad():
        for raw in loader:
            data = _move_batch(raw, next(model.parameters()).device)
            view = _prepare_view(data, schedule, epoch=0, dimensions=dimensions)
            with FlatReadoutCapture(model.osram) as capture:
                logits, _, _, predicted_missing = model(
                    [view["incomplete"]],
                    view["availability"],
                    view["qmask"],
                    view["umask"],
                    view["lengths"],
                    predict_missing=False,
                )
            if predicted_missing is not None:
                raise RuntimeError("Flat diagnostic unexpectedly produced missing-feature predictions")
            if capture.local_calls != 1 or capture.adapter_calls != 1:
                raise RuntimeError(
                    f"Expected one Flat readout, got local={capture.local_calls}, "
                    f"adapter={capture.adapter_calls}"
                )
            scan_count += 1
            original, labels, _ = _collect_predictions(
                config.dataset, logits, view["labels"], view["umask"], config.mosi_task_mode
            )
            labels_all.append(labels)
            for name in SOURCE_COMBINATIONS:
                hidden = replay_flat_hidden(
                    local_skip=model.osram.local_skip,
                    emotion_adapter=model.osram.emotion_adapter,
                    emotion_norm=model.osram.emotion_norm,
                    local=capture.local,
                    flat_input=capture.flat_input,
                    umask=view["umask"],
                    local_dim=config.latent_dim,
                    context_dim=model.osram.context_dim,
                    combination=name,
                )
                replay_logits = model.smax_fc(hidden)
                predicted, replay_labels, _ = _collect_predictions(
                    config.dataset,
                    replay_logits,
                    view["labels"],
                    view["umask"],
                    config.mosi_task_mode,
                )
                np.testing.assert_array_equal(labels, replay_labels)
                predictions[name].append(predicted)
                if name == "LBG":
                    valid = view["umask"].T.bool()
                    error = (replay_logits[valid] - logits[valid]).abs()
                    max_full_logit_error = max(
                        max_full_logit_error,
                        float(error.max()) if error.numel() else 0.0,
                    )
                    torch.testing.assert_close(
                        replay_logits[valid], logits[valid], rtol=2e-5, atol=2e-5
                    )
                    np.testing.assert_array_equal(predicted, original)
    labels = np.concatenate(labels_all)
    combined = {name: np.concatenate(parts) for name, parts in predictions.items()}
    scores = {name: _metrics(labels, values) for name, values in combined.items()}
    artifacts = {"labels": labels, **{f"pred_{name}": value for name, value in combined.items()}}
    checks = {
        "full_logit_max_abs_error": max_full_logit_error,
        "full_class_exact": True,
        "memory_scan_count": scan_count,
    }
    return {"metrics": scores, "checks": checks}, artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import (
        LegacyFixedMaskSchedule,
    )

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config_path = args.source / "config.json"
    source_config = json.loads(config_path.read_text())
    source_mask_protocol = source_config.get("missing_mask_protocol")
    if source_mask_protocol != "gcnet-legacy-fixed-v1":
        raise ValueError(f"Unexpected source mask protocol: {source_mask_protocol!r}")
    config = TrainConfig(**runtime_config_payload(source_config))
    if not (
        config.dataset == "MELD"
        and config.backbone_type == "osram"
        and config.osram_readout_fusion == "flat"
        and config.osram_meaningful_block == "none"
        and config.training_objective == "emotion-only"
        and not config.osram_bidirectional
    ):
        raise ValueError("Source is not the requested causal MELD no-JEPA Flat model")
    device = torch.device(args.device)
    config = replace(config, device=str(device))
    torch.set_num_threads(2)
    feature_root = args.dataset_root / "MELD" / "features"
    roots = [str(feature_root / name) for name in (
        "wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT"
    )]
    _, _, test, ad, td, vd = get_loaders(
        audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=int(_dataset_shape(config.dataset)["num_folds"]),
        dataset=config.dataset, batch_size=config.batch_size, num_workers=0,
        seed=config.seed, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol,
    )
    dimensions = (ad, td, vd)
    model = _build_model(config, dimensions, device)
    mask_seed = SeedBundle(config.seed).derive("missing_mask")
    schedules = {
        index / 10: LegacyFixedMaskSchedule(
            dataset=config.dataset,
            split="test",
            fold=config.fold,
            requested_missing_rate=index / 10,
            mask_seed=mask_seed,
            freeze_evaluation=True,
        )
        for index in range(8)
    }
    summary = {
        "status": "running",
        "training": False,
        "protocol": "INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle source checkpoints",
        "source": str(args.source),
        "source_config_sha256": _sha256(config_path),
        "source_missing_mask_protocol": source_mask_protocol,
        "seed": config.seed,
        "combinations": list(SOURCE_COMBINATIONS),
        "intervention": "one unchanged causal memory scan; zero selected captured Flat input groups only",
        "rates": {},
    }
    _write(args.output / "SUMMARY.json", summary)
    try:
        for index in range(8):
            rate = index / 10
            rate_name = f"{rate:.1f}"
            stem = f"best_miss_{rate_name}".replace(".", "p")
            checkpoint = args.source / f"{stem}.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            result, artifacts = evaluate_rate(
                model, test[config.fold - 1], config, dimensions, schedules[rate]
            )
            prediction_file = f"predictions_miss_{rate_name.replace('.', 'p')}.npz"
            np.savez_compressed(args.output / prediction_file, **artifacts)
            summary["rates"][rate_name] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "epoch": state.get("epoch"),
                "prediction_file": prediction_file,
                **result,
            }
            _write(args.output / "SUMMARY.json", summary)
            full = result["metrics"]["LBG"]
            print(
                f"seed={config.seed} rate={rate_name} "
                f"ACC={100*full['accuracy']:.3f} WF1={100*full['weighted_f1']:.3f}",
                flush=True,
            )
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        _write(args.output / "SUMMARY.json", summary)


if __name__ == "__main__":
    main()
