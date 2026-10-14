"""Ablate Base and Gap in the canonical GCNet-random no-skip checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
    _build_model,
    _metrics,
    _sha256,
    _write,
)
from experiments.osram_meld_no_skip_evidence_ablation_20261014.evaluate import (
    EVIDENCE_CONDITIONS,
    AdapterInputCapture,
    replay_no_skip_hidden,
)


CANONICAL_MASK_PROTOCOL = "gcnet-random-mask-v1"


def replay_view_from_availability(data, availability: torch.Tensor, dimensions):
    """Build the ordinary model view from an already-saved selected-speaker mask."""
    from gcnet_missing_m3.train_gcnet import _prepare_view_from_primary_masks

    if availability.ndim != 3 or availability.shape[-1] != 3:
        raise ValueError("availability must have shape [length, batch, 3]")
    # Only the side selected by qmask contributes at each utterance. Supplying
    # the same selected-speaker availability to both sides reconstructs exactly
    # the saved incomplete input without inventing an unobserved speaker mask.
    return _prepare_view_from_primary_masks(
        data, availability, availability, dimensions
    )


def scatter_saved_availability(
    saved: np.ndarray,
    cursor: int,
    umask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Restore evaluate_rate's valid-utterance flattening order for one batch."""
    # evaluate_rate stores metric availability after transposing to [B, L, 3]
    # and applying the [B, L] umask. Restore that batch-major order first,
    # then return the model-facing [L, B, 3] tensor.
    valid = umask.bool()
    count = int(valid.sum().item())
    stop = cursor + count
    if stop > len(saved):
        raise ValueError("saved availability ended before the test loader")
    batch_major = torch.zeros(
        (*valid.shape, 3), dtype=torch.float32, device=umask.device
    )
    batch_major[valid] = torch.as_tensor(
        saved[cursor:stop], dtype=batch_major.dtype, device=batch_major.device
    )
    return batch_major.transpose(0, 1), stop


def evaluate_rate(
    model,
    loader,
    config,
    dimensions,
    saved_artifact,
    *,
    require_saved_predictions_exact: bool = True,
):
    from gcnet_missing_m3.train_gcnet import _collect_predictions, _move_batch

    saved_availability = np.asarray(saved_artifact["availability"])
    saved_labels = np.asarray(saved_artifact["labels"])
    saved_predictions = np.asarray(saved_artifact["predictions"])
    labels_all: list[np.ndarray] = []
    predictions = {name: [] for name in EVIDENCE_CONDITIONS}
    max_full_logit_error = 0.0
    scans = 0
    cursor = 0
    with torch.no_grad():
        for raw in loader:
            data = _move_batch(raw, next(model.parameters()).device)
            availability, cursor = scatter_saved_availability(
                saved_availability, cursor, data[7]
            )
            view = replay_view_from_availability(data, availability, dimensions)
            with AdapterInputCapture(model.osram) as capture:
                logits, _, _, predicted_missing = model(
                    [view["incomplete"]], view["availability"], view["qmask"],
                    view["umask"], view["lengths"], predict_missing=False,
                )
            if predicted_missing is not None or capture.calls != 1:
                raise RuntimeError("expected one unchanged no-completion Adapter call")
            original, labels, _ = _collect_predictions(
                config.dataset, logits, view["labels"], view["umask"],
                config.mosi_task_mode,
            )
            labels_all.append(labels)
            for name in EVIDENCE_CONDITIONS:
                hidden = replay_no_skip_hidden(
                    model.osram.emotion_adapter,
                    model.osram.emotion_norm,
                    capture.flat_input,
                    view["umask"],
                    local_dim=config.latent_dim,
                    context_dim=model.osram.context_dim,
                    condition=name,
                )
                replay_logits = model.smax_fc(hidden)
                predicted, replay_labels, _ = _collect_predictions(
                    config.dataset, replay_logits, view["labels"],
                    view["umask"], config.mosi_task_mode,
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
            scans += 1
    if cursor != len(saved_availability):
        raise ValueError(
            f"unused saved availability rows: {len(saved_availability) - cursor}"
        )
    labels = np.concatenate(labels_all)
    combined = {name: np.concatenate(parts) for name, parts in predictions.items()}
    np.testing.assert_array_equal(labels, saved_labels)
    saved_prediction_disagreements = int(
        np.count_nonzero(combined["LBG"] != saved_predictions)
    )
    if require_saved_predictions_exact:
        np.testing.assert_array_equal(combined["LBG"], saved_predictions)
    full_correct = combined["LBG"] == labels
    scores = {name: _metrics(labels, values) for name, values in combined.items()}
    changes = {}
    for name, values in combined.items():
        correct = values == labels
        changes[name] = {
            "corrections_vs_full": int((~full_correct & correct).sum()),
            "harms_vs_full": int((full_correct & ~correct).sum()),
            "prediction_changes_vs_full": int((values != combined["LBG"]).sum()),
        }
    return {
        "metrics": scores,
        "changes": changes,
        "checks": {
            "full_logit_max_abs_error": max_full_logit_error,
            "full_class_exact": True,
            "saved_predictions_exact": saved_prediction_disagreements == 0,
            "saved_prediction_disagreements": saved_prediction_disagreements,
            "saved_prediction_count": int(len(saved_predictions)),
            "saved_availability_rows": cursor,
            "memory_scan_count": scans,
        },
    }, {"labels": labels, **{f"pred_{k}": v for k, v in combined.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--meld-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--allow-saved-prediction-mismatch", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal --gpu-uuid")
    os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
    os.environ["GCNET_MELD_PICKLE"] = str(args.meld_pickle)

    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape
    from gcnet_modality_jepa.train_gcnet import get_loaders

    source_config = json.loads((args.source / "config.json").read_text())
    if source_config.get("missing_mask_protocol") != CANONICAL_MASK_PROTOCOL:
        raise ValueError(f"source must use {CANONICAL_MASK_PROTOCOL}")
    config = TrainConfig(**source_config)
    if not (
        config.dataset == "MELD" and config.seed in (66, 67, 68)
        and config.backbone_type == "osram" and not config.osram_bidirectional
        and config.osram_readout_fusion == "flat"
        and config.training_objective == "emotion-only"
        and config.osram_disable_local_skip
    ):
        raise ValueError("source must be a canonical causal MELD no-skip model")
    device = torch.device("cuda")
    config = replace(config, device=str(device))
    torch.set_num_threads(2)
    feature_root = args.dataset_root / "MELD" / "features"
    roots = [str(feature_root / name) for name in (
        "wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT",
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
    output = args.output
    output.mkdir(parents=True)
    summary = {
        "status": "running",
        "training": False,
        "protocol": "INTERNAL DIAGNOSTIC; per-rate Test-best checkpoints; saved paired masks",
        "source": str(args.source),
        "source_config_sha256": _sha256(args.source / "config.json"),
        "source_missing_mask_protocol": CANONICAL_MASK_PROTOCOL,
        "seed": config.seed,
        "conditions": list(EVIDENCE_CONDITIONS),
        "intervention": "one Memory scan; preserve Adapter Local; zero Base and/or Gap only at readout",
        "rates": {},
    }
    _write(output / "SUMMARY.json", summary)
    try:
        for index in range(8):
            rate_name = f"{index / 10:.1f}"
            suffix = rate_name.replace(".", "p")
            checkpoint = args.source / f"best_miss_{suffix}.pt"
            prediction_source = args.source / f"predictions_miss_{suffix}.npz"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            with np.load(prediction_source) as saved:
                result, artifacts = evaluate_rate(
                    model,
                    test[config.fold - 1],
                    config,
                    dimensions,
                    saved,
                    require_saved_predictions_exact=(
                        not args.allow_saved_prediction_mismatch
                    ),
                )
            prediction_file = f"predictions_miss_{suffix}.npz"
            np.savez_compressed(output / prediction_file, **artifacts)
            summary["rates"][rate_name] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "source_predictions": str(prediction_source),
                "source_predictions_sha256": _sha256(prediction_source),
                "epoch": state.get("epoch"),
                "prediction_file": prediction_file,
                **result,
            }
            _write(output / "SUMMARY.json", summary)
            print(rate_name, result["metrics"], flush=True)
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        _write(output / "SUMMARY.json", summary)


if __name__ == "__main__":
    main()
