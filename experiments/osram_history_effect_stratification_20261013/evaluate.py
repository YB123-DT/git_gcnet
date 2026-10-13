"""Save frozen L/LB/LBG logits, strict-past metadata, and effect strata.

Internal diagnostic only: source checkpoints may use per-rate Test-oracle
selection. Zeroing captured Flat inputs does not rerun or alter memory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.osram_iemocap_flat_source_combinations_20261013.evaluate import (  # noqa: E402
    _metrics,
    _runtime_config,
    flatten_metric_availability,
)
from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (  # noqa: E402
    FlatReadoutCapture,
    _build_model,
    _sha256,
    _write,
    materialize_legacy_masks,
    replay_flat_hidden,
    runtime_config_payload,
)

COMBINATIONS = ("L", "LB", "LBG")


def batch_metadata(view: dict) -> dict[str, np.ndarray]:
    """Flatten in metric order; histories exclude the current utterance.

    Speaker IDs are the actual qmask category (including all nine MELD IDs).
    Label relations are analysis-only strata, never model inputs.
    """
    valid = view["umask"].detach().cpu().numpy().astype(bool)
    speakers = view["qmask"].detach().cpu().numpy()
    labels = view["labels"].detach().cpu().numpy()
    availability = view["availability"].detach().cpu().numpy().transpose(1, 0, 2)
    if speakers.shape != valid.shape or labels.shape != valid.shape:
        raise ValueError("qmask, labels, and umask must have shape [batch,time]")
    if availability.shape != (*valid.shape, 3):
        raise ValueError("availability must have shape [time,batch,3]")
    if len(view["conversation_ids"]) != valid.shape[0]:
        raise ValueError("conversation IDs must match the batch size")
    fields = {name: [] for name in (
        "conversation_id", "position", "conversation_length", "speaker",
        "prior_total", "prior_same_speaker", "speaker_switch", "history_modality_seen",
        "same_speaker_history_modality_seen", "current_availability",
        "adjacent_label_relation", "same_speaker_label_relation",
    )}
    for batch, conversation in enumerate(view["conversation_ids"]):
        length = int(valid[batch].sum())
        if length == 0 or not np.array_equal(valid[batch], np.arange(valid.shape[1]) < length):
            raise ValueError("each conversation must have a nonempty contiguous valid prefix")
        history_seen = np.zeros(3, dtype=bool)
        same_seen: dict[int, np.ndarray] = {}
        counts: dict[int, int] = {}
        previous_labels: dict[int, int] = {}
        for position in range(length):
            speaker_value = speakers[batch, position]
            if not np.isfinite(speaker_value) or speaker_value != int(speaker_value):
                raise ValueError("speaker IDs must be finite integers")
            speaker = int(speaker_value)
            label = int(labels[batch, position])
            current = availability[batch, position]
            if not np.isin(current, [0, 1]).all():
                raise ValueError("availability must be binary")
            seen = same_seen.setdefault(speaker, np.zeros(3, dtype=bool))
            row = {
                "conversation_id": str(conversation),
                "position": position,
                "conversation_length": length,
                "speaker": speaker,
                "prior_total": position,
                "prior_same_speaker": counts.get(speaker, 0),
                "speaker_switch": -1 if position == 0 else int(speakers[batch, position - 1] != speaker),
                "history_modality_seen": history_seen.copy(),
                "same_speaker_history_modality_seen": seen.copy(),
                "current_availability": current.astype(bool),
                "adjacent_label_relation": "start" if position == 0 else (
                    "same" if labels[batch, position - 1] == label else "different"
                ),
                "same_speaker_label_relation": "first_for_speaker" if speaker not in previous_labels else (
                    "same" if previous_labels[speaker] == label else "different"
                ),
            }
            for name, value in row.items():
                fields[name].append(value)
            history_seen |= current.astype(bool)
            seen |= current.astype(bool)
            counts[speaker] = counts.get(speaker, 0) + 1
            previous_labels[speaker] = label
    return {name: np.asarray(values) for name, values in fields.items()}


def per_utterance_effects(labels: np.ndarray, logits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Positive Delta = improved true-label CE; corrections and harms are paired."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or labels.dtype.kind not in "iu":
        raise ValueError("labels must be a nonempty integer vector")
    expected_shape = np.asarray(logits["L"]).shape
    if len(expected_shape) != 2 or expected_shape[0] != len(labels):
        raise ValueError("logits must have shape [utterance,class]")
    if labels.min() < 0 or labels.max() >= expected_shape[1]:
        raise ValueError("label outside classifier class range")
    result = {}
    for name in COMBINATIONS:
        values = np.asarray(logits[name], dtype=np.float64)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError("all logits must share a finite [utterance,class] shape")
        shifted = values - values.max(axis=1, keepdims=True)
        log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        ce = -log_probability[np.arange(len(labels)), labels]
        result[f"ce_{name}"] = ce
        result[f"true_probability_{name}"] = np.exp(-ce)
        result[f"pred_{name}"] = values.argmax(axis=1)
        result[f"correct_{name}"] = result[f"pred_{name}"] == labels
    for effect, before, after in (("Base", "L", "LB"), ("Gap", "LB", "LBG"), ("Total", "L", "LBG")):
        result[f"Delta_{effect}"] = result[f"ce_{before}"] - result[f"ce_{after}"]
        result[f"probability_change_{effect}"] = result[f"true_probability_{after}"] - result[f"true_probability_{before}"]
        result[f"correction_{effect}"] = ~result[f"correct_{before}"] & result[f"correct_{after}"]
        result[f"harm_{effect}"] = result[f"correct_{before}"] & ~result[f"correct_{after}"]
    return result


def history_bins(counts: np.ndarray) -> np.ndarray:
    boundaries = [0, 1, 4, 9, 19]
    names = np.asarray(["0", "1", "2-4", "5-9", "10-19", "20+"])
    return names[np.searchsorted(boundaries, counts, side="left")]


def modality_patterns(availability: np.ndarray) -> np.ndarray:
    return np.asarray(["".join(letter for letter, present in zip("ATV", row) if present) or "none" for row in availability])


def summarize_strata(artifacts: dict[str, np.ndarray]) -> dict:
    """Descriptive utterance summaries; no independent-sample significance claims."""
    size = len(artifacts["labels"])
    groupings = {
        "prior_total": history_bins(artifacts["prior_total"]),
        "prior_same_speaker": history_bins(artifacts["prior_same_speaker"]),
        "speaker_switch": np.asarray(["start" if value == -1 else ("switch" if value else "same") for value in artifacts["speaker_switch"]]),
        "current_availability": modality_patterns(artifacts["current_availability"]),
        "history_modality_seen": modality_patterns(artifacts["history_modality_seen"]),
        "same_speaker_history_modality_seen": modality_patterns(artifacts["same_speaker_history_modality_seen"]),
        "adjacent_label_relation": artifacts["adjacent_label_relation"],
        "same_speaker_label_relation": artifacts["same_speaker_label_relation"],
    }
    groupings["current_availability_by_prior_total"] = np.asarray([
        f"{pattern}|history={history}" for pattern, history in zip(groupings["current_availability"], groupings["prior_total"])
    ])
    groupings["current_availability_by_history_seen"] = np.asarray([
        f"{pattern}|seen={seen}" for pattern, seen in zip(groupings["current_availability"], groupings["history_modality_seen"])
    ])

    def summarize(selected):
        count = int(selected.sum())
        row = {"count": count, "accuracy": {}, "mean_ce": {}, "effects": {}}
        for name in COMBINATIONS:
            row["accuracy"][name] = float(artifacts[f"correct_{name}"][selected].mean())
            row["mean_ce"][name] = float(artifacts[f"ce_{name}"][selected].mean())
        for effect, before in (("Base", "L"), ("Gap", "LB"), ("Total", "L")):
            delta = artifacts[f"Delta_{effect}"][selected]
            correction = int(artifacts[f"correction_{effect}"][selected].sum())
            harm = int(artifacts[f"harm_{effect}"][selected].sum())
            before_correct = int(artifacts[f"correct_{before}"][selected].sum())
            row["effects"][effect] = {
                "mean_delta_ce": float(delta.mean()),
                "std_delta_ce": float(delta.std()),
                "positive_delta_count": int((delta > 0).sum()),
                "negative_delta_count": int((delta < 0).sum()),
                "zero_delta_count": int((delta == 0).sum()),
                "corrections": correction,
                "harms": harm,
                "net_corrections": correction - harm,
                "correction_rate": correction / count,
                "harm_rate": harm / count,
                "correction_rate_among_before_errors": correction / (count - before_correct) if before_correct < count else None,
                "harm_rate_among_before_correct": harm / before_correct if before_correct else None,
                "mean_true_probability_change": float(artifacts[f"probability_change_{effect}"][selected].mean()),
            }
        return row

    if not size:
        raise ValueError("cannot summarize zero utterances")
    return {
        "overall": summarize(np.ones(size, dtype=bool)),
        "strata": {name: {str(value): summarize(groups == value) for value in np.unique(groups)} for name, groups in groupings.items()},
    }


def evaluate_rate(model, loader, config, dimensions, schedule, source_prediction=None, *, source_missing_mask_protocol=None):
    from gcnet_missing_m3.train_gcnet import _move_batch, _prepare_view, _prepare_view_from_primary_masks

    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("P0 requires eval mode and frozen parameters")
    parts: dict[str, list[np.ndarray]] = {}
    max_error = 0.0
    replay_mismatches = 0
    scan_count = 0
    with torch.no_grad():
        for raw in loader:
            data = _move_batch(raw, next(model.parameters()).device)
            if config.dataset == "MELD" and source_missing_mask_protocol == "gcnet-legacy-fixed-v1":
                host, guest = materialize_legacy_masks(schedule, data[-1], data[7], epoch=0)
                view = _prepare_view_from_primary_masks(data, host, guest, dimensions)
            else:
                view = _prepare_view(data, schedule, epoch=0, dimensions=dimensions)
            with FlatReadoutCapture(model.osram) as capture:
                original, _, _, missing = model(
                    [view["incomplete"]], view["availability"], view["qmask"],
                    view["umask"], view["lengths"], predict_missing=False,
                )
            if missing is not None or capture.local_calls != 1 or capture.adapter_calls != 1:
                raise RuntimeError("expected one unchanged Flat readout with no missing prediction")
            scan_count += 1
            valid = view["umask"].bool()
            batch = batch_metadata(view)
            batch["labels"] = view["labels"][valid].cpu().numpy()
            logits = {}
            for name in COMBINATIONS:
                hidden = replay_flat_hidden(
                    local_skip=model.osram.local_skip, emotion_adapter=model.osram.emotion_adapter,
                    emotion_norm=model.osram.emotion_norm, local=capture.local,
                    flat_input=capture.flat_input, umask=view["umask"],
                    local_dim=config.latent_dim, context_dim=model.osram.context_dim, combination=name,
                )
                replay = model.smax_fc(hidden).transpose(0, 1)[valid]
                if name == "LBG":
                    full = original.transpose(0, 1)[valid]
                    torch.testing.assert_close(replay, full, rtol=2e-5, atol=2e-5)
                    max_error = max(max_error, float((replay - full).abs().max()))
                    replay_mismatches += int((replay.argmax(-1) != full.argmax(-1)).sum())
                    replay = full
                logits[name] = replay.cpu().numpy()
                batch[f"logits_{name}"] = logits[name]
            batch.update(per_utterance_effects(batch["labels"], logits))
            expected_availability = flatten_metric_availability(view["availability"], view["umask"]).cpu().numpy()
            np.testing.assert_array_equal(batch["current_availability"], expected_availability)
            for name, values in batch.items():
                parts.setdefault(name, []).append(values)
    if not parts:
        raise ValueError("test loader has no batches")
    artifacts = {name: np.concatenate(values) for name, values in parts.items()}
    source_checked = source_prediction is not None
    availability_checked = False
    if source_checked:
        with np.load(source_prediction, allow_pickle=False) as source:
            np.testing.assert_array_equal(artifacts["labels"], source["labels"])
            np.testing.assert_array_equal(artifacts["pred_LBG"], source["predictions"])
            if "availability" in source:
                np.testing.assert_array_equal(artifacts["current_availability"], source["availability"])
                availability_checked = True
    classes = artifacts["logits_L"].shape[1]
    summary = summarize_strata(artifacts)
    summary["metrics"] = {name: _metrics(artifacts["labels"], artifacts[f"pred_{name}"], classes) for name in COMBINATIONS}
    summary["checks"] = {
        "full_logit_max_abs_error": max_error, "full_replay_class_mismatches": replay_mismatches,
        "LBG_uses_original_logits": True, "memory_scan_count": scan_count,
        "source_prediction_exact": source_checked, "source_availability_exact": availability_checked,
    }
    return summary, artifacts


def validate_source(source: dict) -> None:
    if not (
        source.get("dataset") in {"IEMOCAPFour", "IEMOCAPSix", "MELD"}
        and source.get("backbone_type") == "osram"
        and source.get("osram_readout_fusion") == "flat"
        and source.get("osram_meaningful_block") in {None, "none"}
        and source.get("training_objective") == "emotion-only"
        and not source.get("osram_bidirectional", False)
    ):
        raise ValueError("P0 requires a causal emotion-only no-JEPA Flat source")


def code_provenance() -> dict:
    files = [
        "experiments/osram_history_effect_stratification_20261013/evaluate.py",
        "experiments/osram_iemocap_flat_source_combinations_20261013/evaluate.py",
        "experiments/osram_meld_flat_source_combinations_20261013/evaluate.py",
        "experiments/osram_meld_flat_source_combinations_20261013/legacy_schedule.py",
        "gcnet_missing_m3/model.py", "gcnet_missing_m3/osram.py",
        "gcnet_missing_m3/train_gcnet.py", "gcnet_modality_jepa/train_gcnet.py",
        "gcnet_modality_jepa/dataloader_iemocap.py", "gcnet_modality_jepa/dataloader_meld.py",
    ]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    diff = subprocess.run(["git", "diff", "HEAD", "--", *files], cwd=ROOT, capture_output=True, text=True, check=False)
    return {"git_head": revision.stdout.strip() if revision.returncode == 0 else None,
            "tracked_diff": diff.stdout if diff.returncode == 0 else None,
            "file_sha256": {name: _sha256(ROOT / name) for name in files}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--meld-pickle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rates", type=float, nargs="+", default=[index / 10 for index in range(8)])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.rates)) != len(args.rates) or any(rate not in {index / 10 for index in range(8)} for rate in args.rates):
        raise ValueError("rates must be unique members of 0.0--0.7")
    config_path = args.source / "config.json"
    source = json.loads(config_path.read_text())
    validate_source(source)
    source_protocol = source.get("missing_mask_protocol", "conversation-mask-v2")
    if source["dataset"] == "MELD" and (args.meld_pickle is None or not args.meld_pickle.is_file()):
        raise ValueError("MELD requires the source run's --meld-pickle")
    os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
    if args.meld_pickle is not None:
        os.environ["GCNET_MELD_PICKLE"] = str(args.meld_pickle)
    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape, _schedules
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import LegacyFixedMaskSchedule

    config = TrainConfig(**_runtime_config(runtime_config_payload(source)))
    config = replace(config, device=args.device)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    feature_root = args.dataset_root / ("MELD" if config.dataset == "MELD" else "IEMOCAP") / "features"
    roots = [str(feature_root / name) for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")]
    _, _, test, ad, td, vd = get_loaders(
        audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=int(_dataset_shape(config.dataset)["num_folds"]), dataset=config.dataset,
        batch_size=config.batch_size, num_workers=0, seed=config.seed,
        validation_fraction=config.validation_fraction, evaluation_protocol=config.evaluation_protocol,
    )
    model = _build_model(config, (ad, td, vd), device)
    if config.dataset == "MELD" and source_protocol == "gcnet-legacy-fixed-v1":
        schedules = {rate: LegacyFixedMaskSchedule(config.dataset, "test", config.fold, rate, SeedBundle(config.seed).derive("missing_mask"), freeze_evaluation=True) for rate in args.rates}
    else:
        schedules = _schedules(config, "test")
    args.output.mkdir(parents=True)
    _write(args.output / "runtime_config.json", asdict(config))
    _write(args.output / "code_provenance.json", code_provenance())
    summary = {
        "status": "running", "training": False,
        "protocol": "INTERNAL DIAGNOSTIC ONLY; frozen source per-rate Test-oracle checkpoints; descriptive strata",
        "source": str(args.source), "source_config_sha256": _sha256(config_path),
        "source_missing_mask_protocol": source_protocol,
        "dataset": config.dataset, "seed": config.seed, "fold": config.fold,
        "combinations": list(COMBINATIONS), "device": str(device),
        "runtime_config_file": "runtime_config.json", "code_provenance_file": "code_provenance.json",
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "delta_definition": "Delta_Base=CE(L)-CE(LB); Delta_Gap=CE(LB)-CE(LBG); positive means improvement",
        "history_definition": "strict past valid utterances; current availability excluded; A/T/V order",
        "intervention": "one unchanged memory scan; zero selected Flat input groups only",
        "rates": {},
    }
    if args.meld_pickle:
        summary.update(dataset_pickle=str(args.meld_pickle), dataset_pickle_sha256=_sha256(args.meld_pickle))
    _write(args.output / "SUMMARY.json", summary)
    try:
        for rate in args.rates:
            rate_name = f"{rate:.1f}"
            suffix = rate_name.replace(".", "p")
            checkpoint = args.source / f"best_miss_{suffix}.pt"
            source_prediction = args.source / f"predictions_miss_{suffix}.npz"
            if config.dataset != "MELD" and not source_prediction.is_file():
                raise FileNotFoundError(source_prediction)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            result, artifacts = evaluate_rate(
                model, test[config.fold - 1], config, (ad, td, vd), schedules[rate],
                source_prediction if source_prediction.is_file() else None,
                source_missing_mask_protocol=source_protocol,
            )
            artifact_name = f"utterances_miss_{suffix}.npz"
            np.savez_compressed(args.output / artifact_name, **artifacts)
            summary["rates"][rate_name] = {
                "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                "source_prediction": str(source_prediction) if source_prediction.is_file() else None,
                "epoch": state.get("epoch"), "utterance_file": artifact_name, **result,
            }
            _write(args.output / "SUMMARY.json", summary)
            print(f"{config.dataset} seed={config.seed} fold={config.fold} rate={rate_name} n={len(artifacts['labels'])}", flush=True)
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        _write(args.output / "SUMMARY.json", summary)


if __name__ == "__main__":
    main()
