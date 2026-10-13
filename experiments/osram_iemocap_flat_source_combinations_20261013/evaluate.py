"""Replay all Local/Base/Gap combinations from one frozen IEMOCAP Flat run."""

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

from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (  # noqa: E402
    SOURCE_COMBINATIONS,
    FlatReadoutCapture,
    _build_model,
    _sha256,
    _write,
    replay_flat_hidden,
)


def _runtime_config(source: dict) -> dict:
    payload = dict(source)
    if payload.get("osram_meaningful_block") is None:
        payload["osram_meaningful_block"] = "none"
    return payload


def _metrics(labels: np.ndarray, predictions: np.ndarray, classes: int) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "unweighted_accuracy": float(
            recall_score(
                labels,
                predictions,
                labels=list(range(classes)),
                average="macro",
                zero_division=0,
            )
        ),
    }


def flatten_metric_availability(
    availability: torch.Tensor, umask: torch.Tensor
) -> torch.Tensor:
    """Match `_collect_predictions` and saved-artifact batch-major ordering."""
    if availability.ndim != 3 or availability.shape[:2] != umask.T.shape:
        raise ValueError("availability must be [time,batch,3] for [batch,time] umask")
    return availability.transpose(0, 1)[umask.bool()]


def authoritative_combination_prediction(
    combination: str, replayed: np.ndarray, original: np.ndarray
) -> np.ndarray:
    """Use the untouched forward as the full-combination numerical anchor."""
    return original if combination == "LBG" else replayed


def evaluate_rate(model, loader, config, dimensions, schedule, source_prediction, classes):
    from gcnet_missing_m3.train_gcnet import (
        _collect_predictions,
        _move_batch,
        _prepare_view,
    )

    labels_all: list[np.ndarray] = []
    availability_all: list[np.ndarray] = []
    predictions = {name: [] for name in SOURCE_COMBINATIONS}
    max_full_logit_error = 0.0
    full_replay_class_mismatches = 0
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
                raise RuntimeError("Flat diagnostic unexpectedly predicted missing features")
            if capture.local_calls != 1 or capture.adapter_calls != 1:
                raise RuntimeError(
                    f"Expected one Flat readout, got local={capture.local_calls}, "
                    f"adapter={capture.adapter_calls}"
                )
            scan_count += 1
            original, labels, _ = _collect_predictions(
                config.dataset, logits, view["labels"], view["umask"], config.mosi_task_mode
            )
            valid = view["umask"].T.bool()
            labels_all.append(labels)
            availability_all.append(
                flatten_metric_availability(view["availability"], view["umask"])
                .cpu()
                .numpy()
            )
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
                if name == "LBG":
                    error = (replay_logits[valid] - logits[valid]).abs()
                    max_full_logit_error = max(
                        max_full_logit_error,
                        float(error.max()) if error.numel() else 0.0,
                    )
                    torch.testing.assert_close(
                        replay_logits[valid], logits[valid], rtol=2e-5, atol=2e-5
                    )
                    full_replay_class_mismatches += int(np.not_equal(predicted, original).sum())
                predictions[name].append(
                    authoritative_combination_prediction(name, predicted, original)
                )

    labels = np.concatenate(labels_all)
    availability = np.concatenate(availability_all)
    combined = {name: np.concatenate(parts) for name, parts in predictions.items()}
    source = np.load(source_prediction)
    np.testing.assert_array_equal(labels, source["labels"])
    np.testing.assert_array_equal(combined["LBG"], source["predictions"])
    np.testing.assert_array_equal(availability, source["availability"])
    scores = {
        name: _metrics(labels, values, classes) for name, values in combined.items()
    }
    artifacts = {
        "labels": labels,
        "availability": availability,
        **{f"pred_{name}": value for name, value in combined.items()},
    }
    checks = {
        "full_logit_max_abs_error": max_full_logit_error,
        "full_class_exact": True,
        "full_replay_class_mismatches": full_replay_class_mismatches,
        "source_prediction_exact": True,
        "source_availability_exact": True,
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

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)

    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape, _schedules
    from gcnet_modality_jepa.train_gcnet import get_loaders

    config_path = args.source / "config.json"
    source_config = json.loads(config_path.read_text())
    config = TrainConfig(**_runtime_config(source_config))
    if not (
        config.dataset in {"IEMOCAPFour", "IEMOCAPSix"}
        and config.backbone_type == "osram"
        and config.osram_readout_fusion == "flat"
        and config.osram_meaningful_block == "none"
        and config.training_objective == "emotion-only"
        and not config.osram_bidirectional
    ):
        raise ValueError("Source is not a causal IEMOCAP no-JEPA Flat run")
    classes = int(_dataset_shape(config.dataset)["num_classes"])
    device = torch.device(args.device)
    config = replace(config, device=str(device))
    torch.set_num_threads(2)
    feature_root = args.dataset_root / "IEMOCAP" / "features"
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
    schedules = _schedules(config, "test")
    summary = {
        "status": "running",
        "training": False,
        "protocol": "INTERNAL DIAGNOSTIC ONLY; frozen per-rate Test-oracle checkpoints",
        "source": str(args.source),
        "source_config_sha256": _sha256(config_path),
        "dataset": config.dataset,
        "seed": config.seed,
        "fold": config.fold,
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
            source_prediction = args.source / f"predictions_miss_{rate_name.replace('.', 'p')}.npz"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            result, artifacts = evaluate_rate(
                model,
                test[config.fold - 1],
                config,
                dimensions,
                schedules[rate],
                source_prediction,
                classes,
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
                f"{config.dataset} seed={config.seed} fold={config.fold} "
                f"rate={rate_name} ACC={100*full['accuracy']:.3f} "
                f"UA={100*full['unweighted_accuracy']:.3f}",
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
