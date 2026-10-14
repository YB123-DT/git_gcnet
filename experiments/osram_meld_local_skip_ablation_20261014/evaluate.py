"""Disable only Flat's final pre-LayerNorm Local shortcut on MELD seed66."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from experiments.osram_meld_information_availability_20261014.run import (
    _sha256,
    _write,
    model_state_hash,
    source_epoch_metrics,
)
from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
    FlatReadoutCapture,
    _build_model,
    materialize_legacy_masks,
    runtime_config_payload,
)


LABEL = "INTERNAL DIAGNOSTIC ONLY; frozen Test-oracle MELD checkpoints"


def replay_flat_readout(
    local_skip,
    emotion_adapter,
    emotion_norm,
    local: torch.Tensor,
    flat_input: torch.Tensor,
    umask: torch.Tensor,
    *,
    keep_local_shortcut: bool,
) -> torch.Tensor:
    """Replay Flat while preserving Adapter([Local,Base,Gap]) byte-for-byte."""
    if local.shape[:2] != flat_input.shape[:2] or umask.T.shape != local.shape[:2]:
        raise ValueError("Local, Flat input, and umask are not aligned")
    valid = umask.T.bool().unsqueeze(-1)
    safe_local = torch.where(valid, local, torch.zeros_like(local))
    safe_input = torch.where(valid, flat_input, torch.zeros_like(flat_input))
    adapter_value = emotion_adapter(safe_input)
    if keep_local_shortcut:
        pre_norm = local_skip(safe_local) + adapter_value
    else:
        pre_norm = adapter_value
    hidden = emotion_norm(pre_norm)
    return torch.where(valid, hidden, torch.zeros_like(hidden))


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def evaluate_rate(model, loader, schedule, dimensions, config) -> tuple[dict, dict]:
    from gcnet_missing_m3.train_gcnet import _collect_predictions, _move_batch, _prepare_view_from_primary_masks

    labels_parts, original_parts, off_parts = [], [], []
    replay_max_error = 0.0
    scans = 0
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("model must be frozen and in eval mode")
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
            if predicted_missing is not None or capture.local_calls != 1 or capture.adapter_calls != 1:
                raise RuntimeError("expected one unchanged no-completion Flat forward")
            original, labels, _ = _collect_predictions(
                config.dataset, logits, view["labels"], view["umask"], config.mosi_task_mode,
            )
            replay_hidden = replay_flat_readout(
                model.osram.local_skip, model.osram.emotion_adapter, model.osram.emotion_norm,
                capture.local, capture.flat_input, view["umask"], keep_local_shortcut=True,
            )
            replay_logits = model.smax_fc(replay_hidden)
            replay, replay_labels, _ = _collect_predictions(
                config.dataset, replay_logits, view["labels"], view["umask"], config.mosi_task_mode,
            )
            np.testing.assert_array_equal(labels, replay_labels)
            np.testing.assert_array_equal(original, replay)
            valid = view["umask"].T.bool()
            torch.testing.assert_close(replay_logits[valid], logits[valid], rtol=2e-5, atol=2e-5)
            replay_max_error = max(replay_max_error, float((replay_logits[valid] - logits[valid]).abs().max()))
            off_hidden = replay_flat_readout(
                model.osram.local_skip, model.osram.emotion_adapter, model.osram.emotion_norm,
                capture.local, capture.flat_input, view["umask"], keep_local_shortcut=False,
            )
            off_logits = model.smax_fc(off_hidden)
            off, off_labels, _ = _collect_predictions(
                config.dataset, off_logits, view["labels"], view["umask"], config.mosi_task_mode,
            )
            np.testing.assert_array_equal(labels, off_labels)
            labels_parts.append(labels)
            original_parts.append(original)
            off_parts.append(off)
            scans += 1
    labels = np.concatenate(labels_parts)
    original = np.concatenate(original_parts)
    shortcut_off = np.concatenate(off_parts)
    original_correct = original == labels
    off_correct = shortcut_off == labels
    result = {
        "original": _metrics(labels, original),
        "shortcut_off": _metrics(labels, shortcut_off),
        "delta_off_minus_original": {
            "accuracy": _metrics(labels, shortcut_off)["accuracy"] - _metrics(labels, original)["accuracy"],
            "weighted_f1": _metrics(labels, shortcut_off)["weighted_f1"] - _metrics(labels, original)["weighted_f1"],
        },
        "corrections": int((~original_correct & off_correct).sum()),
        "harms": int((original_correct & ~off_correct).sum()),
        "prediction_changes": int((original != shortcut_off).sum()),
        "memory_scans": scans,
        "original_replay_max_abs_error": replay_max_error,
    }
    artifacts = {"labels": labels, "original": original, "shortcut_off": shortcut_off}
    return result, artifacts


def _render(summary: dict) -> str:
    rates = [f"{index / 10:.1f}" for index in range(8)]
    rows = [
        "# MELD Local-shortcut-only frozen ablation",
        "",
        "**INTERNAL DIAGNOSTIC ONLY.** Evaluation-only intervention on retained per-rate Test-oracle checkpoints.",
        "",
        "Only `local_skip(Local)` immediately before the final LayerNorm is removed. "
        "The Local prefix inside `emotion_adapter([Local,Base,Gap])` remains unchanged; Memory is scanned once and is unchanged.",
        "",
        "| Metric | " + " | ".join(rates) + " | 8-rate mean | high mean |",
        "|---|" + "---:|" * 10,
    ]
    for key, label in (("accuracy", "ACC original"), ("off_accuracy", "ACC shortcut-off"),
                       ("weighted_f1", "W-F1 original"), ("off_weighted_f1", "W-F1 shortcut-off")):
        values = []
        for rate in rates:
            record = summary["rates"][rate]
            if key.startswith("off_"):
                values.append(record["shortcut_off"][key.removeprefix("off_")])
            else:
                values.append(record["original"][key])
        mean = float(np.mean(values))
        high = float(np.mean(values[5:]))
        rows.append(f"| {label} | " + " | ".join(f"{100*value:.3f}" for value in values) + f" | {100*mean:.3f} | {100*high:.3f} |")
    corrections = sum(summary["rates"][rate]["corrections"] for rate in rates)
    harms = sum(summary["rates"][rate]["harms"] for rate in rates)
    changes = sum(summary["rates"][rate]["prediction_changes"] for rate in rates)
    rows.extend(["", f"Across rate evaluations: corrections={corrections}, harms={harms}, prediction changes={changes}.", ""])
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--meld-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.gpu_uuid and os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal the selected healthy GPU UUID")
    args.output.mkdir(parents=True)
    summary = {"label": LABEL, "status": "running", "source": str(args.source), "rates": {}}
    _write(args.output / "SUMMARY.json", summary)
    try:
        os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
        os.environ["GCNET_MELD_PICKLE"] = str(args.meld_pickle)
        from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape
        from gcnet_modality_jepa.protocol import SeedBundle
        from gcnet_modality_jepa.train_gcnet import get_loaders
        from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import LegacyFixedMaskSchedule

        source_config = json.loads((args.source / "config.json").read_text())
        if source_config.get("missing_mask_protocol") != "gcnet-legacy-fixed-v1":
            raise ValueError("source must use gcnet-legacy-fixed-v1")
        config = TrainConfig(**runtime_config_payload(source_config))
        if not (config.dataset == "MELD" and config.seed == 66 and config.backbone_type == "osram"
                and config.osram_readout_fusion == "flat" and config.training_objective == "emotion-only"
                and not config.osram_bidirectional):
            raise ValueError("source must be MELD seed66 causal no-JEPA Flat")
        device = torch.device(args.device)
        config = replace(config, device=str(device))
        feature_root = args.dataset_root / "MELD" / "features"
        roots = [str(feature_root / name) for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")]
        _, _, test, ad, td, vd = get_loaders(
            audio_root=roots[0], text_root=roots[1], video_root=roots[2],
            num_folder=int(_dataset_shape(config.dataset)["num_folds"]), dataset=config.dataset,
            batch_size=config.batch_size, num_workers=0, seed=config.seed,
            validation_fraction=config.validation_fraction, evaluation_protocol=config.evaluation_protocol,
        )
        dimensions = (ad, td, vd)
        model = _build_model(config, dimensions, device)
        history = json.loads((args.source / "history.json").read_text())
        mask_seed = SeedBundle(config.seed).derive("missing_mask")
        summary.update(
            source_config_sha256=_sha256(args.source / "config.json"), seed=66,
            intervention="remove only local_skip(Local) before final emotion_norm",
            adapter_local_retained=True, memory_unchanged=True, memory_scans_per_batch=1,
        )
        for index in range(8):
            rate = index / 10
            rate_name = f"{rate:.1f}"
            checkpoint = args.source / f"best_miss_{rate_name.replace('.', 'p')}.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"], strict=True)
            model.eval().requires_grad_(False)
            before = model_state_hash(model)
            schedule = LegacyFixedMaskSchedule(
                dataset="MELD", split="test", fold=config.fold, requested_missing_rate=rate,
                mask_seed=mask_seed, freeze_evaluation=True,
            )
            result, artifacts = evaluate_rate(model, test[config.fold - 1], schedule, dimensions, config)
            reference = source_epoch_metrics(history, epoch=int(saved["epoch"]), rate=rate)
            for metric in ("accuracy", "weighted_f1"):
                if abs(result["original"][metric] - reference[metric]) > 1e-12:
                    raise AssertionError(f"source parity failed rate={rate_name} metric={metric}")
            if model_state_hash(model) != before:
                raise AssertionError("frozen model changed")
            prediction_file = f"predictions_miss_{rate_name.replace('.', 'p')}.npz"
            np.savez_compressed(args.output / prediction_file, **artifacts)
            summary["rates"][rate_name] = {
                **result, "source_parity": True, "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint), "checkpoint_epoch": int(saved["epoch"]),
                "prediction_file": prediction_file,
            }
            _write(args.output / "SUMMARY.json", summary)
            print(json.dumps({"rate": rate_name, **result}), flush=True)
        summary["status"] = "complete"
        (args.output / "RESULT.md").write_text(_render(summary))
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        _write(args.output / "SUMMARY.json", summary)


if __name__ == "__main__":
    main()
