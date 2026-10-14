"""Run frozen MELD information-availability probes.

INTERNAL DIAGNOSTIC ONLY: retained backbones and offline probes use test-oracle
selection.  This script never updates the OSRAM backbone.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

from .features import match_donors, mask_forward_memory, strict_past_summaries, validate_cache
from .probe import PROBE_PAIRS, build_probe_inputs, classification_metrics, fit_probe


LABEL = "INTERNAL DIAGNOSTIC ONLY; Test-oracle backbone and probe selection"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_epoch_metrics(history: list[dict], *, epoch: int, rate: float) -> dict:
    matches = [row for row in history if int(row.get("epoch", -1)) == int(epoch)]
    if len(matches) != 1:
        raise ValueError(f"expected one source history row for epoch {epoch}")
    metrics = matches[0].get("test_oracle", {}).get(f"{rate:.1f}")
    if not isinstance(metrics, dict) or "accuracy" not in metrics or "weighted_f1" not in metrics:
        raise ValueError(f"source history lacks test-oracle metrics for rate {rate:.1f}")
    return {"accuracy": float(metrics["accuracy"]), "weighted_f1": float(metrics["weighted_f1"])}


def assert_source_parity_scope(max_batches: int | None) -> None:
    if max_batches is not None:
        raise ValueError("source metric parity requires the full test split")


def flatten_batch_cache(view: dict, capture, logits: torch.Tensor, *, local_dim: int, context_dim: int) -> dict:
    """Flatten valid utterances conversation-first from one unchanged forward."""
    if capture.local.shape[:2] != logits.shape[:2] or capture.flat_input.shape[:2] != logits.shape[:2]:
        raise ValueError("captured tensors and logits must be time-major and aligned")
    if capture.local.shape[-1] != local_dim:
        raise ValueError("unexpected Local width")
    expected = local_dim + 4 * context_dim
    if capture.flat_input.shape[-1] != expected:
        raise ValueError("unexpected Flat input width")
    torch.testing.assert_close(capture.flat_input[..., :local_dim], capture.local, rtol=0, atol=0)
    umask = view["umask"].bool()
    if umask.ndim != 2 or len(view["conversation_ids"]) != umask.shape[0]:
        raise ValueError("invalid batch mask or conversation IDs")
    fields = {name: [] for name in (
        "conversation_ids", "utterance_indices", "speaker", "labels", "availability",
        "local", "base_full", "gap_full", "prediction",
    )}
    flat = capture.flat_input
    base = flat[..., local_dim : local_dim + context_dim]
    gap = flat[..., local_dim + context_dim :].reshape(*flat.shape[:2], 3, context_dim)
    for batch, conversation in enumerate(view["conversation_ids"]):
        length = int(umask[batch].sum())
        if length < 1 or not torch.equal(umask[batch], torch.arange(umask.shape[1], device=umask.device) < length):
            raise ValueError("valid utterances must form a nonempty prefix")
        for position in range(length):
            fields["conversation_ids"].append(str(conversation))
            fields["utterance_indices"].append(position)
            fields["speaker"].append(int(view["qmask"][batch, position].item()))
            fields["labels"].append(int(view["labels"][batch, position].item()))
            fields["availability"].append(view["availability"][position, batch].detach().cpu().numpy())
            fields["local"].append(capture.local[position, batch].detach().cpu().numpy())
            fields["base_full"].append(base[position, batch].detach().cpu().numpy())
            fields["gap_full"].append(gap[position, batch].detach().cpu().numpy())
            fields["prediction"].append(logits[position, batch].detach().cpu().numpy())
    result = {name: np.asarray(values) for name, values in fields.items()}
    result["utterance_indices"] = result["utterance_indices"].astype(np.int64)
    result["speaker"] = result["speaker"].astype(np.int64)
    result["labels"] = result["labels"].astype(np.int64)
    result["availability"] = result["availability"].astype(np.int64)
    result["base"], result["gap"] = mask_forward_memory(
        result.pop("base_full"), result.pop("gap_full"), result["availability"]
    )
    result.update(strict_past_summaries(result))
    return result


def _concatenate(parts: list[dict]) -> dict:
    if not parts:
        raise ValueError("no extracted batches")
    keys = parts[0].keys()
    if any(part.keys() != parts[0].keys() for part in parts[1:]):
        raise ValueError("batch cache fields differ")
    result = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    # Loaders may interleave conversations across batches; restore strict identity order.
    order = np.lexsort((result["utterance_indices"], result["conversation_ids"].astype(str)))
    return {key: values[order] for key, values in result.items()}


def extract_split(model, loader, schedule, dimensions, config, *, max_batches: int | None = None) -> tuple[dict, int]:
    from gcnet_missing_m3.train_gcnet import _move_batch, _prepare_view_from_primary_masks
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
        FlatReadoutCapture,
        materialize_legacy_masks,
    )

    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("backbone must be frozen and in eval mode")
    parts = []
    scans = 0
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
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
            parts.append(
                flatten_batch_cache(
                    view, capture, logits, local_dim=config.latent_dim,
                    context_dim=model.osram.context_dim,
                )
            )
            scans += 1
    cache = _concatenate(parts)
    validate_cache(cache)
    first = cache["utterance_indices"] == 0
    if np.count_nonzero(cache["base"][first]) or np.count_nonzero(cache["gap"][first]):
        raise AssertionError("read-before-write first turns must have zero Memory")
    return cache, scans


def _fit_rate_probes(
    split_data: dict[str, dict], destination: Path, *, seeds: tuple[int, ...], epochs: int,
    device: str, batch_size: int,
) -> dict:
    labels = {split: values["labels"] for split, values in split_data.items()}
    summary = {"probe_seeds": list(seeds), "epochs": epochs, "batch_size": batch_size, "runs": [], "coverage": {}}
    for seed in seeds:
        donor = {split: match_donors(values, seed + 1009 * index) for index, (split, values) in enumerate(split_data.items())}
        inputs = {split: build_probe_inputs(split_data[split], donor[split]) for split in split_data}
        summary["coverage"][str(seed)] = {
            split: {"total": match.total, "matched": match.matched, "fraction": match.matched / match.total}
            for split, match in donor.items()
        }
        donor_payload = {
            split: {
                "indices": match.indices.tolist(), "included": match.included.tolist(),
                "target_conversation": split_data[split]["conversation_ids"].astype(str).tolist(),
                "donor_conversation": [
                    str(split_data[split]["conversation_ids"][index]) if index >= 0 else None
                    for index in match.indices
                ],
            }
            for split, match in donor.items()
        }
        _write(destination / f"donors_seed{seed}.json", donor_payload)
        variants = ["Local", *[name for pair in PROBE_PAIRS.items() for name in pair]]
        for variant in variants:
            pair_real = next((real for real, control in PROBE_PAIRS.items() if variant in (real, control)), None)
            masks = {
                split: (np.ones(len(labels[split]), dtype=bool) if variant == "Local" else donor[split].included)
                for split in split_data
            }
            output = destination / f"seed_{seed}" / variant
            result = fit_probe(
                inputs["train"][variant][masks["train"]], labels["train"][masks["train"]],
                inputs["test"][variant][masks["test"]], labels["test"][masks["test"]],
                seed=seed, epochs=epochs, destination=output, device=device, batch_size=batch_size,
            )
            result.update(
                variant=variant, paired_family=pair_real,
                coverage={split: int(mask.sum()) for split, mask in masks.items()},
            )
            summary["runs"].append(result)
            _write(destination / "PROBES.json", summary)
        del inputs
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--meld-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[index / 10 for index in range(8)])
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[66, 67, 68])
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    allowed_rates = {index / 10 for index in range(8)}
    if len(set(args.rates)) != len(args.rates) or not set(args.rates) <= allowed_rates:
        parser.error("rates must be unique members of 0.0,...,0.7")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.gpu_uuid and os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal the explicitly selected healthy GPU UUID")
    args.output.mkdir(parents=True)
    state = {
        "label": LABEL, "status": "running", "pid": os.getpid(), "started_utc": _now(),
        "command": sys.argv, "rates": args.rates, "probe_seeds": args.probe_seeds,
    }
    _write(args.output / "STATUS.json", state)
    try:
        os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
        os.environ["GCNET_MELD_PICKLE"] = str(args.meld_pickle)
        from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape
        from gcnet_modality_jepa.protocol import SeedBundle
        from gcnet_modality_jepa.train_gcnet import get_loaders
        from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
            _build_model, runtime_config_payload,
        )
        from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import LegacyFixedMaskSchedule

        source_config_path = args.source / "config.json"
        source_config = json.loads(source_config_path.read_text())
        if source_config.get("missing_mask_protocol") != "gcnet-legacy-fixed-v1":
            raise ValueError("source must use the retained MELD legacy fixed mask")
        config = TrainConfig(**runtime_config_payload(source_config))
        if not (
            config.dataset == "MELD" and config.seed == 66 and config.backbone_type == "osram"
            and config.osram_readout_fusion == "flat" and config.training_objective == "emotion-only"
            and not config.osram_bidirectional
        ):
            raise ValueError("source must be MELD seed66 causal no-JEPA Flat")
        device = torch.device(args.device)
        config = replace(config, device=str(device))
        torch.set_num_threads(2)
        root = args.dataset_root / "MELD" / "features"
        feature_roots = [str(root / name) for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")]
        train, _, test, ad, td, vd = get_loaders(
            audio_root=feature_roots[0], text_root=feature_roots[1], video_root=feature_roots[2],
            num_folder=int(_dataset_shape(config.dataset)["num_folds"]), dataset=config.dataset,
            batch_size=config.batch_size, num_workers=0, seed=config.seed,
            validation_fraction=config.validation_fraction, evaluation_protocol=config.evaluation_protocol,
        )
        loaders = {"train": train[config.fold - 1], "test": test[config.fold - 1]}
        dimensions = (ad, td, vd)
        model = _build_model(config, dimensions, device)
        mask_seed = SeedBundle(config.seed).derive("missing_mask")
        provenance = {
            "label": LABEL, "host": platform.node(), "python": sys.version, "torch": torch.__version__,
            "source": str(args.source), "source_config_sha256": _sha256(source_config_path),
            "meld_pickle": str(args.meld_pickle), "meld_pickle_sha256": _sha256(args.meld_pickle),
            "gpu_uuid": args.gpu_uuid, "mask_epoch": 0, "model_frozen": True,
            "probe_selection_split": "test", "probe_gradient_split": "train",
        }
        _write(args.output / "PROVENANCE.json", provenance)
        source_history = json.loads((args.source / "history.json").read_text())
        state["records"] = []
        for rate in args.rates:
            rate_name = f"{rate:.1f}"
            rate_dir = args.output / f"rate_{rate_name.replace('.', 'p')}"
            rate_dir.mkdir()
            checkpoint = args.source / f"best_miss_{rate_name.replace('.', 'p')}.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"], strict=True)
            model.eval().requires_grad_(False)
            frozen_hash = model_state_hash(model)
            split_data = {}
            scans = {}
            for split, loader in loaders.items():
                schedule = LegacyFixedMaskSchedule(
                    dataset="MELD", split=split, fold=config.fold, requested_missing_rate=rate,
                    mask_seed=mask_seed, freeze_evaluation=True,
                )
                extraction_limit = None
                if split == "test":
                    assert_source_parity_scope(extraction_limit)
                split_data[split], scans[split] = extract_split(
                    model, loader, schedule, dimensions, config,
                    max_batches=extraction_limit,
                )
                np.savez_compressed(rate_dir / f"{split}_features.npz", **split_data[split])
            if set(split_data["train"]["conversation_ids"]) & set(split_data["test"]["conversation_ids"]):
                raise AssertionError("train/test conversation overlap")
            test_predictions = split_data["test"]["prediction"].argmax(axis=1)
            baseline = classification_metrics(split_data["test"]["labels"], test_predictions)
            reference = source_epoch_metrics(source_history, epoch=int(saved["epoch"]), rate=rate)
            for metric in ("accuracy", "weighted_f1"):
                if abs(baseline[metric] - reference[metric]) > 1e-12:
                    raise AssertionError(
                        f"source prediction parity failed at rate={rate_name} {metric}: "
                        f"{baseline[metric]} != {reference[metric]}"
                    )
            probes = _fit_rate_probes(
                split_data, rate_dir / "probes",
                seeds=tuple(args.probe_seeds[:1] if args.check_only else args.probe_seeds),
                epochs=2 if args.check_only else args.probe_epochs,
                device=str(device), batch_size=args.probe_batch_size,
            )
            if model_state_hash(model) != frozen_hash or any(parameter.requires_grad for parameter in model.parameters()):
                raise AssertionError("frozen backbone changed during extraction/probe fitting")
            record = {
                "rate": rate, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_epoch": saved.get("epoch"), "frozen_state_sha256": frozen_hash,
                "baseline": baseline, "source_reference": reference, "source_parity": True,
                "scans": scans, "probe_runs": len(probes["runs"]),
            }
            _write(rate_dir / "RESULT.json", record)
            state["records"].append(record)
            _write(args.output / "STATUS.json", state)
            print(json.dumps({"phase": "rate_complete", **record}), flush=True)
        state.update(status="complete", finished_utc=_now())
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), finished_utc=_now())
        raise
    finally:
        _write(args.output / "STATUS.json", state)


if __name__ == "__main__":
    main()
