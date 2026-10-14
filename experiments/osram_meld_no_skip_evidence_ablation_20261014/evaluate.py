"""Ablate Base and Gap at the readout of retrained MELD no-skip checkpoints."""

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
    materialize_legacy_masks,
    runtime_config_payload,
)


EVIDENCE_CONDITIONS = {
    "LBG": (True, True),
    "LG": (False, True),
    "LB": (True, False),
    "L": (False, False),
}


def replay_no_skip_hidden(
    emotion_adapter,
    emotion_norm,
    flat_input: torch.Tensor,
    umask: torch.Tensor,
    *,
    local_dim: int,
    context_dim: int,
    condition: str,
) -> torch.Tensor:
    """Replay Adapter(Local, Base, Gap) without ever adding Local Skip."""
    if condition not in EVIDENCE_CONDITIONS:
        raise ValueError(f"unknown evidence condition: {condition}")
    if flat_input.shape[-1] != local_dim + 4 * context_dim:
        raise ValueError("unexpected no-skip Flat input width")
    keep_base, keep_gap = EVIDENCE_CONDITIONS[condition]
    selected = flat_input.clone()
    if not keep_base:
        selected[..., local_dim : local_dim + context_dim] = 0
    if not keep_gap:
        selected[..., local_dim + context_dim :] = 0
    valid = umask.T.bool().unsqueeze(-1)
    selected = torch.where(valid, selected, torch.zeros_like(selected))
    hidden = emotion_norm(emotion_adapter(selected))
    return torch.where(valid, hidden, torch.zeros_like(hidden))


class AdapterInputCapture:
    def __init__(self, osram) -> None:
        self.osram = osram
        self.calls = 0

    def _capture(self, _module, args) -> None:
        self.flat_input = args[0].detach().clone()
        self.calls += 1

    def __enter__(self):
        self.handle = self.osram.emotion_adapter.register_forward_pre_hook(self._capture)
        return self

    def __exit__(self, *_args) -> None:
        self.handle.remove()


def evaluate_rate(model, loader, config, dimensions, schedule) -> tuple[dict, dict]:
    from gcnet_missing_m3.train_gcnet import (
        _collect_predictions,
        _move_batch,
        _prepare_view_from_primary_masks,
    )

    labels_all: list[np.ndarray] = []
    predictions = {name: [] for name in EVIDENCE_CONDITIONS}
    max_full_logit_error = 0.0
    scans = 0
    with torch.no_grad():
        for raw in loader:
            data = _move_batch(raw, next(model.parameters()).device)
            host, guest = materialize_legacy_masks(schedule, data[-1], data[7], epoch=0)
            view = _prepare_view_from_primary_masks(data, host, guest, dimensions)
            with AdapterInputCapture(model.osram) as capture:
                logits, _, _, predicted_missing = model(
                    [view["incomplete"]], view["availability"], view["qmask"],
                    view["umask"], view["lengths"], predict_missing=False,
                )
            if predicted_missing is not None or capture.calls != 1:
                raise RuntimeError("expected one unchanged no-completion Adapter call")
            original, labels, _ = _collect_predictions(
                config.dataset, logits, view["labels"], view["umask"], config.mosi_task_mode,
            )
            labels_all.append(labels)
            for name in EVIDENCE_CONDITIONS:
                hidden = replay_no_skip_hidden(
                    model.osram.emotion_adapter, model.osram.emotion_norm,
                    capture.flat_input, view["umask"], local_dim=config.latent_dim,
                    context_dim=model.osram.context_dim, condition=name,
                )
                replay_logits = model.smax_fc(hidden)
                predicted, replay_labels, _ = _collect_predictions(
                    config.dataset, replay_logits, view["labels"], view["umask"],
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
                        replay_logits[valid], logits[valid], rtol=2e-5, atol=2e-5,
                    )
                    np.testing.assert_array_equal(predicted, original)
            scans += 1
    labels = np.concatenate(labels_all)
    combined = {name: np.concatenate(parts) for name, parts in predictions.items()}
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
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal --gpu-uuid")
    os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
    os.environ["GCNET_MELD_PICKLE"] = str(args.meld_pickle)

    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import (
        LegacyFixedMaskSchedule,
    )

    source_config = json.loads((args.source / "config.json").read_text())
    if source_config.get("missing_mask_protocol") != "gcnet-legacy-fixed-v1":
        raise ValueError("source must use gcnet-legacy-fixed-v1")
    config = TrainConfig(**runtime_config_payload(source_config))
    if not (
        config.dataset == "MELD" and config.seed == 66
        and config.backbone_type == "osram" and not config.osram_bidirectional
        and config.osram_readout_fusion == "flat"
        and config.training_objective == "emotion-only"
        and config.osram_disable_local_skip
    ):
        raise ValueError("source must be the retrained seed66 causal MELD no-skip model")
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
    mask_seed = SeedBundle(config.seed).derive("missing_mask")
    output = args.output
    output.mkdir(parents=True)
    summary = {
        "status": "running",
        "training": False,
        "protocol": "INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle checkpoints",
        "source": str(args.source),
        "source_config_sha256": _sha256(args.source / "config.json"),
        "seed": config.seed,
        "conditions": list(EVIDENCE_CONDITIONS),
        "intervention": "one Memory scan; preserve Adapter Local; zero Base and/or Gap only at readout",
        "rates": {},
    }
    _write(output / "SUMMARY.json", summary)
    try:
        for index in range(8):
            rate_name = f"{index / 10:.1f}"
            checkpoint = args.source / f"best_miss_{rate_name.replace('.', 'p')}.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            schedule = LegacyFixedMaskSchedule(
                dataset=config.dataset, split="test", fold=config.fold,
                requested_missing_rate=index / 10, mask_seed=mask_seed,
                freeze_evaluation=True,
            )
            result, artifacts = evaluate_rate(
                model, test[config.fold - 1], config, dimensions, schedule,
            )
            prediction_file = f"predictions_miss_{rate_name.replace('.', 'p')}.npz"
            np.savez_compressed(output / prediction_file, **artifacts)
            summary["rates"][rate_name] = {
                "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                "epoch": state.get("epoch"), "prediction_file": prediction_file, **result,
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
