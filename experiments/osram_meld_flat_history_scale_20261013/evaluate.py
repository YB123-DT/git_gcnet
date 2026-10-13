"""Scale Base and Gap together at the frozen MELD Flat readout."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def scale_flat_history(
    flat_input: torch.Tensor,
    umask: torch.Tensor,
    *,
    local_dim: int,
    context_dim: int,
    alpha: float,
) -> torch.Tensor:
    """Keep Local fixed and scale the Base plus three Gap input blocks."""
    expected = local_dim + 4 * context_dim
    if flat_input.shape[-1] != expected:
        raise ValueError(f"expected Flat width {expected}, got {flat_input.shape[-1]}")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and nonnegative")
    scaled = flat_input.clone()
    scaled[..., local_dim:] = scaled[..., local_dim:] * float(alpha)
    valid = umask.T.bool().unsqueeze(-1)
    return torch.where(valid, scaled, torch.zeros_like(scaled))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluate_rate(model, loader, config, dimensions, schedule):
    from gcnet_missing_m3.train_gcnet import (
        _collect_predictions,
        _move_batch,
        _prepare_view_from_primary_masks,
    )
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
        FlatReadoutCapture,
        _metrics,
        materialize_legacy_masks,
    )

    labels_all = []
    predictions = {f"{alpha:g}": [] for alpha in ALPHAS}
    max_anchor_error = 0.0
    scans = 0
    with torch.no_grad():
        for raw in loader:
            data = _move_batch(raw, next(model.parameters()).device)
            host, guest = materialize_legacy_masks(schedule, data[-1], data[7], epoch=0)
            view = _prepare_view_from_primary_masks(data, host, guest, dimensions)
            with FlatReadoutCapture(model.osram) as capture:
                logits, _, _, predicted_missing = model(
                    [view["incomplete"]], view["availability"], view["qmask"],
                    view["umask"], view["lengths"], predict_missing=False,
                )
            if predicted_missing is not None:
                raise RuntimeError("unexpected missing-feature prediction")
            if capture.local_calls != 1 or capture.adapter_calls != 1:
                raise RuntimeError("expected exactly one unchanged Flat scan")
            scans += 1
            original, labels, _ = _collect_predictions(
                config.dataset, logits, view["labels"], view["umask"], config.mosi_task_mode
            )
            labels_all.append(labels)
            valid = view["umask"].T.bool()
            for alpha in ALPHAS:
                selected = scale_flat_history(
                    capture.flat_input,
                    view["umask"],
                    local_dim=config.latent_dim,
                    context_dim=model.osram.context_dim,
                    alpha=alpha,
                )
                hidden = model.osram.emotion_norm(
                    model.osram.local_skip(capture.local)
                    + model.osram.emotion_adapter(selected)
                )
                hidden = torch.where(
                    valid.unsqueeze(-1), hidden, torch.zeros_like(hidden)
                )
                replay_logits = model.smax_fc(hidden)
                replay, replay_labels, _ = _collect_predictions(
                    config.dataset, replay_logits, view["labels"], view["umask"],
                    config.mosi_task_mode,
                )
                np.testing.assert_array_equal(labels, replay_labels)
                predictions[f"{alpha:g}"].append(replay)
                if alpha == 1.0:
                    error = (replay_logits[valid] - logits[valid]).abs()
                    max_anchor_error = max(
                        max_anchor_error,
                        float(error.max()) if error.numel() else 0.0,
                    )
                    torch.testing.assert_close(
                        replay_logits[valid], logits[valid], rtol=2e-5, atol=2e-5
                    )
                    np.testing.assert_array_equal(replay, original)
    labels = np.concatenate(labels_all)
    combined = {key: np.concatenate(parts) for key, parts in predictions.items()}
    return (
        {
            "metrics": {key: _metrics(labels, pred) for key, pred in combined.items()},
            "checks": {
                "alpha_one_class_exact": True,
                "alpha_one_logit_max_abs_error": max_anchor_error,
                "memory_scan_count": scans,
            },
        },
        {"labels": labels, **{f"pred_alpha_{key}": pred for key, pred in combined.items()}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--meld-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.meld_pickle.is_file():
        raise FileNotFoundError(args.meld_pickle)
    args.output.mkdir(parents=True)
    os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
    os.environ["GCNET_MELD_PICKLE"] = str(args.meld_pickle)

    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
        _build_model,
        _sha256,
        runtime_config_payload,
    )
    from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import (
        LegacyFixedMaskSchedule,
    )

    config_path = args.source / "config.json"
    source_config = json.loads(config_path.read_text())
    if source_config.get("missing_mask_protocol") != "gcnet-legacy-fixed-v1":
        raise ValueError("unexpected source mask protocol")
    config = TrainConfig(**runtime_config_payload(source_config))
    if not (
        config.dataset == "MELD"
        and config.backbone_type == "osram"
        and config.osram_readout_fusion == "flat"
        and config.osram_meaningful_block == "none"
        and config.training_objective == "emotion-only"
        and not config.osram_bidirectional
    ):
        raise ValueError("source is not the requested causal MELD no-JEPA Flat model")
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
            config.dataset, "test", config.fold, index / 10, mask_seed, True
        )
        for index in range(8)
    }
    summary = {
        "status": "running",
        "training": False,
        "protocol": "INTERNAL DIAGNOSTIC ONLY; frozen per-rate Test-oracle checkpoints",
        "source": str(args.source),
        "source_config_sha256": _sha256(config_path),
        "dataset_pickle": str(args.meld_pickle),
        "seed": config.seed,
        "alphas": list(ALPHAS),
        "intervention": "scale Base and already-masked Gap together at Flat input",
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
            anchor = result["metrics"]["1"]
            print(
                f"seed={config.seed} rate={rate_name} "
                f"ACC={100*anchor['accuracy']:.3f} WF1={100*anchor['weighted_f1']:.3f}",
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
