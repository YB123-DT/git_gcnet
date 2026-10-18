"""Frozen Local/Memory readout interventions on exact ACC-selected masks.

Every batch scans the original CHM once. L/BG/LBG differ only in the final
Flat input and Local Skip, not in any state, query, key/value or write.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
    FlatReadoutCapture, replay_flat_hidden,
)
from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
    replay_view_from_availability, scatter_saved_availability,
)
from experiments.osram_meld_write_tuning_20261017.capture import _metrics


CONDITIONS = ("L", "BG", "LBG")
METADATA = ("labels", "conversation_ids", "utterance_indices", "speakers")


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _metadata(data):
    valid = _numpy(data[7]).astype(bool)
    ids = np.asarray([str(value) for value in data[-1]])
    if ids.shape != (valid.shape[0],):
        raise ValueError("Conversation IDs do not match the loader batch")
    return dict(labels=_numpy(data[8])[valid].astype(np.int64),
        speakers=_numpy(data[6])[valid],
        conversation_ids=np.broadcast_to(ids[:, None], valid.shape)[valid],
        utterance_indices=np.broadcast_to(np.arange(valid.shape[1]), valid.shape)[valid])


def _validate_saved(saved):
    required = set(METADATA) | {"availability", "logits", "predictions"}
    if not required.issubset(saved):
        raise ValueError("Exact selection artifact is missing: " + str(sorted(required - set(saved))))
    saved = {key: np.asarray(saved[key]) for key in required}
    n = len(saved["labels"])
    if n == 0 or any(len(value) != n for value in saved.values()):
        raise ValueError("Empty or inconsistent selected-artifact lengths")
    availability = saved["availability"]
    if (availability.shape != (n, 3) or not np.isin(availability, [0, 1]).all()
            or np.any(availability.sum(1) < 1)):
        raise ValueError("Selected availability must be nonempty binary [N,3]")
    if saved["logits"].shape != (n, 7) or not np.isfinite(saved["logits"]).all():
        raise ValueError("Selected logits must be finite [N,7]")
    if saved["labels"].shape != (n,) or not np.isin(saved["labels"], np.arange(7)).all():
        raise ValueError("Expected seven-class MELD labels")
    np.testing.assert_array_equal(saved["predictions"], saved["logits"].argmax(1))
    return saved


def evaluate_rate(model, loader, cfg, dimensions, saved, trainer):
    """Return per-condition metrics and aligned, small prediction artifacts.

    ``saved`` must be the checkpoint-linked selection artifact, never the
    independently resampled final evaluation. ACC/UA/WF1/recall use percent.
    """
    if cfg.dataset != "MELD":
        raise ValueError("This diagnostic is restricted to seven-class MELD")
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Provide a fully frozen eval-mode model")
    saved = _validate_saved(saved)
    parts = {name: [] for name in CONDITIONS}
    metadata_parts = {name: [] for name in METADATA}
    cursor = scans = 0
    saved_logit_error = full_logit_error = 0.
    with torch.no_grad():
        for raw in loader:
            data = trainer._move_batch(raw, next(model.parameters()).device)
            metadata = _metadata(data)
            count = len(metadata["labels"])
            for name, actual in metadata.items():
                np.testing.assert_array_equal(actual, saved[name][cursor:cursor + count],
                    err_msg="Selection artifact/loader alignment failed for " + name)
            availability, stop = scatter_saved_availability(saved["availability"], cursor, data[7])
            view = replay_view_from_availability(data, availability, dimensions)
            np.testing.assert_array_equal(
                _numpy(view["availability"].transpose(0, 1))[_numpy(view["umask"]).astype(bool)],
                saved["availability"][cursor:stop])
            with FlatReadoutCapture(model.osram) as capture:
                logits, _, _, missing = model([view["incomplete"]], view["availability"],
                    view["qmask"], view["umask"], view["lengths"], predict_missing=False)
            if missing is not None or capture.local_calls != 1 or capture.adapter_calls != 1:
                raise RuntimeError("Expected one original Flat call with no completion")
            scans += 1
            valid = view["umask"].bool()
            original = _numpy(logits.transpose(0, 1)[valid])
            expected = saved["logits"][cursor:stop]
            np.testing.assert_allclose(original, expected, rtol=2e-5, atol=2e-5,
                err_msg="Original forward no longer reproduces selected logits")
            np.testing.assert_array_equal(original.argmax(1), saved["predictions"][cursor:stop])
            saved_logit_error = max(saved_logit_error, float(np.abs(original - expected).max(initial=0.)))
            for name in CONDITIONS:
                hidden = replay_flat_hidden(local_skip=model.osram.local_skip,
                    emotion_adapter=model.osram.emotion_adapter, emotion_norm=model.osram.emotion_norm,
                    local=capture.local, flat_input=capture.flat_input, umask=view["umask"],
                    local_dim=cfg.latent_dim, context_dim=model.osram.context_dim, combination=name)
                replay = _numpy(model.smax_fc(hidden).transpose(0, 1)[valid])
                if not np.isfinite(replay).all():
                    raise ValueError("Nonfinite readout intervention logits")
                parts[name].append(replay)
                if name == "LBG":
                    np.testing.assert_allclose(replay, original, rtol=2e-5, atol=2e-5)
                    np.testing.assert_array_equal(replay.argmax(1), original.argmax(1))
                    full_logit_error = max(full_logit_error, float(np.abs(replay - original).max(initial=0.)))
            for name, actual in metadata.items():
                metadata_parts[name].append(actual)
            cursor = stop
    if cursor != len(saved["labels"]):
        raise ValueError("Loader did not consume every selected-artifact row")
    combined = {name: np.concatenate(values) for name, values in parts.items()}
    artifacts = {name: np.concatenate(values) for name, values in metadata_parts.items()}
    artifacts["availability"] = saved["availability"].copy()
    labels = artifacts["labels"]
    predictions = {name: logits.argmax(1) for name, logits in combined.items()}
    scores = {name: _metrics(dict(labels=labels, logits=logits)) for name, logits in combined.items()}
    correct = {name: values == labels for name, values in predictions.items()}
    changes = {name: dict(corrections_vs_full=int((correct[name] & ~correct["LBG"]).sum()),
        harms_vs_full=int((~correct[name] & correct["LBG"]).sum()),
        prediction_changes_vs_full=int((predictions[name] != predictions["LBG"]).sum())) for name in CONDITIONS}
    local_to_full = dict(corrections=int((~correct["L"] & correct["LBG"]).sum()),
        harms=int((correct["L"] & ~correct["LBG"]).sum()),
        prediction_changes=int((predictions["L"] != predictions["LBG"]).sum()))
    cross = dict(both_correct=int((correct["L"] & correct["BG"]).sum()),
        local_only_correct=int((correct["L"] & ~correct["BG"]).sum()),
        memory_only_correct=int((~correct["L"] & correct["BG"]).sum()),
        both_wrong=int((~correct["L"] & ~correct["BG"]).sum()))
    for name in CONDITIONS:
        artifacts["logits_" + name] = combined[name]
        artifacts["pred_" + name] = predictions[name]
    checks = dict(memory_scan_count=scans, saved_availability_rows=cursor,
        saved_predictions_exact=True, full_class_exact=True, metadata_exact=True,
        saved_logit_max_abs_error=saved_logit_error, full_logit_max_abs_error=full_logit_error)
    return dict(metrics=scores, changes=changes, local_to_full=local_to_full,
                local_memory_cross=cross, checks=checks), artifacts
