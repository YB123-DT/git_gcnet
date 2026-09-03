"""Pure tensor utilities for the frozen missing-latent oracle diagnostic."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .model import MODALITIES


__all__ = [
    "OracleState",
    "build_sample_keys",
    "concatenate_oracle_states",
    "compute_path_output",
    "conversation_cluster_bootstrap",
    "effective_rank",
    "extract_oracle_batch",
    "flatten_valid_lbd",
    "fusion_path_diagnostics",
    "metric_mean_std",
    "regression_metrics",
    "restore_named_buffers",
    "shuffle_targets_by_modality",
    "snapshot_named_buffers",
    "stable_seed",
    "stack_teacher_targets",
    "state_dict_sha256",
    "tensor_sha256",
]


@dataclass(frozen=True)
class OracleState:
    """Conversation-major tensors collected for frozen oracle paths."""

    sample_keys: Tuple[str, ...]
    labels: torch.Tensor
    availability: torch.Tensor
    graph_hidden: torch.Tensor
    predicted_latents: torch.Tensor
    teacher_latents: torch.Tensor
    target_mask: torch.Tensor
    native_logits: torch.Tensor

    def __post_init__(self) -> None:
        keys = tuple(self.sample_keys)
        object.__setattr__(self, "sample_keys", keys)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate sample key in OracleState")
        sample_count = len(keys)
        if self.labels.ndim != 1 or self.labels.shape[0] != sample_count:
            raise ValueError("labels must have shape [N]")
        if tuple(self.availability.shape) != (sample_count, len(MODALITIES)):
            raise ValueError("availability must have shape [N, 3]")
        if self.graph_hidden.ndim != 2 or self.graph_hidden.shape[0] != sample_count:
            raise ValueError("graph_hidden must have shape [N, H]")
        latent_prefix = (sample_count, len(MODALITIES))
        if (
            self.predicted_latents.ndim != 3
            or tuple(self.predicted_latents.shape[:2]) != latent_prefix
        ):
            raise ValueError("predicted_latents must have shape [N, 3, D]")
        if self.teacher_latents.shape != self.predicted_latents.shape:
            raise ValueError("teacher_latents must match predicted_latents")
        if tuple(self.target_mask.shape) != latent_prefix:
            raise ValueError("target_mask must have shape [N, 3]")
        if self.native_logits.ndim != 2 or self.native_logits.shape[0] != sample_count:
            raise ValueError("native_logits must have shape [N, C]")


def concatenate_oracle_states(states: Sequence[OracleState]) -> OracleState:
    """Concatenate batch states without changing their metric order."""
    values = tuple(states)
    if not values:
        raise ValueError("states must contain at least one OracleState")
    if any(not isinstance(state, OracleState) for state in values):
        raise TypeError("states must contain only OracleState values")
    sample_keys = tuple(key for state in values for key in state.sample_keys)
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("duplicate sample key across OracleState values")

    def concatenate(name: str) -> torch.Tensor:
        return torch.cat([getattr(state, name) for state in values], dim=0)

    return OracleState(
        sample_keys=sample_keys,
        labels=concatenate("labels"),
        availability=concatenate("availability"),
        graph_hidden=concatenate("graph_hidden"),
        predicted_latents=concatenate("predicted_latents"),
        teacher_latents=concatenate("teacher_latents"),
        target_mask=concatenate("target_mask"),
        native_logits=concatenate("native_logits"),
    )


def flatten_valid_lbd(value: torch.Tensor, umask: torch.Tensor) -> torch.Tensor:
    """Flatten ``[L, B, ...]`` in conversation-major metric order."""
    if value.ndim < 2:
        raise ValueError("value must have shape [L, B, ...]")
    if umask.ndim != 2 or tuple(umask.shape) != (value.shape[1], value.shape[0]):
        raise ValueError("umask must have shape [B, L]")
    valid = umask.to(device=value.device, dtype=torch.bool)
    return value.transpose(0, 1)[valid]


def build_sample_keys(
    conversation_ids: Sequence[object], umask: torch.Tensor
) -> Tuple[str, ...]:
    """Return ``conversation_id:utterance_index`` keys in metric order."""
    if umask.ndim != 2:
        raise ValueError("umask must have shape [B, L]")
    ids = tuple(conversation_ids)
    if len(ids) != umask.shape[0]:
        raise ValueError("conversation_ids must contain one id per batch item")
    valid = umask.detach().to(device="cpu", dtype=torch.bool).tolist()
    return tuple(
        f"{conversation_id}:{utterance_index}"
        for conversation_id, conversation_mask in zip(ids, valid)
        for utterance_index, is_valid in enumerate(conversation_mask)
        if is_valid
    )


def stack_teacher_targets(mapping: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Stack teacher tensors immediately before the feature axis."""
    if set(mapping) != set(MODALITIES):
        raise ValueError("mapping must contain audio, text, and visual")
    values = [mapping[name] for name in MODALITIES]
    if any(not torch.is_tensor(value) for value in values):
        raise TypeError("teacher targets must be tensors")
    if any(value.ndim < 1 for value in values):
        raise ValueError("teacher targets must have a feature axis")
    if any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("teacher targets must share one shape")
    return torch.stack(values, dim=-2)


def stable_seed(*parts: object) -> int:
    """Derive a process-independent 64-bit seed from printable components."""
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _hash_part(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    contiguous = value.detach().to(device="cpu").contiguous()
    try:
        return contiguous.numpy().tobytes(order="C")
    except TypeError:
        return contiguous.view(torch.uint8).numpy().tobytes(order="C")


def _shape_bytes(value: torch.Tensor) -> bytes:
    return ",".join(str(int(size)) for size in value.shape).encode("ascii")


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and logical contiguous bytes."""
    if not torch.is_tensor(value):
        raise TypeError("value must be a tensor")
    digest = hashlib.sha256()
    _hash_part(digest, str(value.dtype).encode("ascii"))
    _hash_part(digest, _shape_bytes(value))
    _hash_part(digest, _tensor_bytes(value))
    return digest.hexdigest()


def state_dict_sha256(
    value: Union[nn.Module, Mapping[str, torch.Tensor]]
) -> str:
    """Hash a state dict in sorted key, dtype, shape, and byte order."""
    state = value.state_dict() if isinstance(value, nn.Module) else value
    if not isinstance(state, Mapping):
        raise TypeError("value must be a module or tensor mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise TypeError("state_dict values must be tensors")
        _hash_part(digest, str(name).encode("utf-8"))
        _hash_part(digest, str(tensor.dtype).encode("ascii"))
        _hash_part(digest, _shape_bytes(tensor))
        _hash_part(digest, _tensor_bytes(tensor))
    return digest.hexdigest()


def snapshot_named_buffers(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone every persistent and non-persistent named buffer."""
    return {
        name: buffer.detach().clone()
        for name, buffer in module.named_buffers()
    }


@torch.no_grad()
def restore_named_buffers(
    module: nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> None:
    """Restore a complete named-buffer snapshot without replacing buffers."""
    current = dict(module.named_buffers())
    if set(current) != set(snapshot):
        raise ValueError("snapshot names do not match module named buffers")
    for name, saved in snapshot.items():
        destination = current[name]
        if destination.shape != saved.shape or destination.dtype != saved.dtype:
            raise ValueError(f"snapshot metadata differs for buffer {name!r}")
    for name, saved in snapshot.items():
        destination = current[name]
        destination.copy_(saved.to(device=destination.device))


def effective_rank(value: torch.Tensor) -> float:
    """Return entropy effective rank for a finite matrix."""
    if value.ndim != 2:
        raise ValueError("value must have shape [N, D]")
    matrix = value.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("value must contain only finite entries")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return 0.0
    if matrix.shape[0] < 2:
        return 1.0
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    if hasattr(torch.linalg, "svdvals"):
        singular_values = torch.linalg.svdvals(matrix)
    else:
        singular_values = torch.svd(matrix, some=False).S
    total = singular_values.sum()
    if singular_values.numel() == 0 or float(total.item()) <= 1e-12:
        return 1.0
    probabilities = singular_values / total
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum()
    return float(entropy.exp().item())


def shuffle_targets_by_modality(
    teacher: torch.Tensor,
    target_mask: torch.Tensor,
    master_seed: int,
    rate: float,
    shuffle_index: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Derange each missing-target pool independently and deterministically."""
    if teacher.ndim != 3 or teacher.shape[1] != len(MODALITIES):
        raise ValueError("teacher must have shape [N, 3, D]")
    if target_mask.ndim != 2 or tuple(target_mask.shape) != tuple(teacher.shape[:2]):
        raise ValueError("target_mask must have shape [N, 3]")

    output = teacher.clone()
    mask_cpu = target_mask.detach().to(device="cpu", dtype=torch.bool)
    target_metadata: Dict[str, Dict[str, Any]] = {}
    for target_index, name in enumerate(MODALITIES):
        selected = torch.nonzero(
            mask_cpu[:, target_index], as_tuple=False
        ).flatten()
        count = int(selected.numel())
        order_seed = stable_seed(
            master_seed, rate, shuffle_index, name, "order"
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(order_seed)
        order = torch.randperm(count, generator=generator)
        source_positions = order.clone()
        if count > 1:
            shift = 1 + stable_seed(
                master_seed, rate, shuffle_index, name, "shift"
            ) % (count - 1)
            source_positions = torch.empty_like(order)
            source_positions[order] = order.roll(int(shift))
            destinations = selected.to(device=teacher.device)
            sources = selected[source_positions].to(device=teacher.device)
            output[destinations, target_index] = teacher[sources, target_index]
        source_indices = selected[source_positions]
        fixed_points = int(
            (source_positions == torch.arange(count, dtype=torch.long)).sum().item()
        )
        target_metadata[name] = {
            "seed": order_seed,
            "count": count,
            "fixed_points": fixed_points,
            "unshufflable": count == 1,
            "permutation_sha256": tensor_sha256(source_indices),
        }

    metadata: Dict[str, Any] = {
        "master_seed": int(master_seed),
        "rate": float(rate),
        "shuffle_index": int(shuffle_index),
        "modalities": target_metadata,
    }
    return output, metadata


def compute_path_output(
    graph_hidden: torch.Tensor,
    target_latents: Union[torch.Tensor, None],
    target_mask: torch.Tensor,
    fusion: nn.Module,
    classifier: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute one frozen classifier path from conversation-major tensors."""
    if graph_hidden.ndim != 2:
        raise ValueError("graph_hidden must have shape [N, H]")
    sample_count, hidden_dim = graph_hidden.shape
    if target_mask.ndim != 2 or tuple(target_mask.shape) != (
        sample_count,
        len(MODALITIES),
    ):
        raise ValueError("target_mask must have shape [N, 3]")

    if target_latents is None:
        residual = torch.zeros_like(graph_hidden)
    else:
        if target_latents.ndim != 3 or tuple(target_latents.shape[:2]) != (
            sample_count,
            len(MODALITIES),
        ):
            raise ValueError("target_latents must have shape [N, 3, D]")
        if target_latents.device != graph_hidden.device:
            raise ValueError("graph_hidden and target_latents must share a device")
        sequence_latents = target_latents.unsqueeze(1)
        sequence_mask = target_mask.to(
            device=target_latents.device, dtype=torch.bool
        ).unsqueeze(1)
        umask = torch.ones(
            (1, sample_count), dtype=torch.bool, device=target_latents.device
        )
        sequence_residual = fusion(sequence_latents, sequence_mask, umask)
        expected_shape = (sample_count, 1, hidden_dim)
        if tuple(sequence_residual.shape) != expected_shape:
            raise ValueError(
                "fusion must return residual with shape [N, 1, H]"
            )
        residual = sequence_residual[:, 0]

    logits = classifier(graph_hidden + residual)
    if logits.ndim != 2 or logits.shape[0] != sample_count:
        raise ValueError("classifier must return logits with shape [N, C]")
    return logits, residual


@torch.no_grad()
def extract_oracle_batch(
    model: nn.Module, view: Mapping[str, Any]
) -> Tuple[OracleState, Dict[str, float]]:
    """Extract one batch and audit manual completion against native forward."""
    buffer_snapshot = snapshot_named_buffers(model)
    try:
        if model.training:
            raise ValueError("model must be in evaluation mode (training=False)")
        if bool(getattr(model, "local_context_residual", False)):
            raise ValueError(
                "extract_oracle_batch does not support local_context_residual=True"
            )
        if not bool(getattr(model, "classification_completion", False)):
            raise ValueError("model must enable classification completion")

        incomplete = view["incomplete"]
        complete = view["complete"]
        availability = view["availability"]
        qmask = view["qmask"]
        umask = view["umask"]
        labels = view["labels"]
        lengths = view["lengths"]

        encoded, observed_latents = model.observed_set(
            incomplete, availability, umask
        )
        graph_lbd = model.encode_hidden([encoded], qmask, umask, lengths)
        predictions = model.missing_predictor(
            observed_latents, graph_lbd, availability, umask
        )
        teacher_lbd = stack_teacher_targets(
            model.encode_teacher_targets([complete])
        )
        native_logits_lbd, native_hidden_lbd, _, returned_predictions = model(
            [incomplete],
            availability,
            qmask,
            umask,
            lengths,
            predict_missing=False,
        )
        if returned_predictions is not None:
            raise RuntimeError("native inference unexpectedly returned predictions")

        graph_hidden = flatten_valid_lbd(graph_lbd, umask)
        predicted_latents = flatten_valid_lbd(
            predictions.reg_predictions, umask
        )
        teacher_latents = flatten_valid_lbd(teacher_lbd, umask)
        target_mask = flatten_valid_lbd(predictions.target_mask, umask)
        native_logits = flatten_valid_lbd(native_logits_lbd, umask)
        native_hidden = flatten_valid_lbd(native_hidden_lbd, umask)
        flat_availability = flatten_valid_lbd(availability, umask)
        flat_labels = flatten_valid_lbd(labels.transpose(0, 1), umask)

        predicted_logits, predicted_residual = compute_path_output(
            graph_hidden,
            predicted_latents,
            target_mask,
            model.missing_latent_fusion,
            model.smax_fc,
        )
        predicted_hidden = graph_hidden + predicted_residual

        def max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
            if left.shape != right.shape:
                raise ValueError("manual and native tensors must share a shape")
            if left.numel() == 0:
                return 0.0
            return float((left - right).abs().max().item())

        audit = {
            "predicted_hidden_max_abs_error": max_abs_error(
                predicted_hidden, native_hidden
            ),
            "predicted_logits_max_abs_error": max_abs_error(
                predicted_logits, native_logits
            ),
        }
        state = OracleState(
            sample_keys=build_sample_keys(view["conversation_ids"], umask),
            labels=flat_labels,
            availability=flat_availability,
            graph_hidden=graph_hidden,
            predicted_latents=predicted_latents,
            teacher_latents=teacher_latents,
            target_mask=target_mask,
            native_logits=native_logits,
        )
        return state, audit
    finally:
        restore_named_buffers(model, buffer_snapshot)


def _numpy_vector(value: Any, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().to(device="cpu").numpy()
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    try:
        finite = np.isfinite(array).all()
    except TypeError as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if not bool(finite):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float64, copy=False)


def regression_metrics(labels: Any, predictions: Any) -> Dict[str, float]:
    """Return the historical CMU-MOSI regression evaluation metrics."""
    target = _numpy_vector(labels, "labels")
    predicted = _numpy_vector(predictions, "predictions")
    if target.shape[0] != predicted.shape[0]:
        raise ValueError("labels and predictions must have equal length")

    nonzero = target != 0
    if bool(nonzero.any()):
        binary_target = target[nonzero] > 0
        binary_predicted = predicted[nonzero] > 0
        weighted_f1 = float(
            f1_score(binary_target, binary_predicted, average="weighted")
        )
        macro_f1 = float(
            f1_score(binary_target, binary_predicted, average="macro")
        )
        accuracy = float(accuracy_score(binary_target, binary_predicted))
    else:
        weighted_f1 = 0.0
        macro_f1 = 0.0
        accuracy = 0.0

    correlation = (
        float(np.corrcoef(target, predicted)[0, 1])
        if target.size >= 2
        and float(np.std(target)) > 0.0
        and float(np.std(predicted)) > 0.0
        else 0.0
    )
    return {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "mae": float(np.mean(np.abs(target - predicted))),
        "correlation": correlation,
    }


def _weighted_sign_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Fast equivalent of sklearn weighted F1 for MOSI sign labels."""
    selected = labels != 0
    if not bool(selected.any()):
        return 0.0
    target = labels[selected] > 0
    predicted = predictions[selected] > 0
    score = 0.0
    for label in np.unique(np.concatenate((target, predicted))):
        target_is_label = target == label
        predicted_is_label = predicted == label
        true_positive = int(np.sum(target_is_label & predicted_is_label))
        false_positive = int(np.sum(~target_is_label & predicted_is_label))
        false_negative = int(np.sum(target_is_label & ~predicted_is_label))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
        score += int(np.sum(target_is_label)) * f1
    return float(score / target.shape[0])


def _rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.float().square().mean().sqrt().item())


def _latent_statistics(value: torch.Tensor, prefix: str) -> Dict[str, float]:
    if value.numel() == 0:
        return {
            f"{prefix}_effective_rank": 0.0,
            f"{prefix}_channel_std": 0.0,
            f"{prefix}_rms": 0.0,
        }
    channel_std = value.float().std(dim=0, unbiased=False).mean()
    return {
        f"{prefix}_effective_rank": effective_rank(value),
        f"{prefix}_channel_std": float(channel_std.item()),
        f"{prefix}_rms": _rms(value),
    }


@torch.no_grad()
def fusion_path_diagnostics(
    latents: torch.Tensor,
    target_mask: torch.Tensor,
    fusion: nn.Module,
    graph_hidden: torch.Tensor,
    graph_logits: torch.Tensor,
    classifier: nn.Module,
) -> Dict[str, Any]:
    """Measure latent geometry and its frozen residual/logit contribution."""
    if latents.ndim != 3 or latents.shape[1] != len(MODALITIES):
        raise ValueError("latents must have shape [N, 3, D]")
    sample_count = latents.shape[0]
    if target_mask.ndim != 2 or tuple(target_mask.shape) != (
        sample_count,
        len(MODALITIES),
    ):
        raise ValueError("target_mask must have shape [N, 3]")
    if graph_hidden.ndim != 2 or graph_hidden.shape[0] != sample_count:
        raise ValueError("graph_hidden must have shape [N, H]")
    if graph_logits.ndim != 2 or graph_logits.shape[0] != sample_count:
        raise ValueError("graph_logits must have shape [N, C]")
    for name, value in (
        ("latents", latents),
        ("graph_hidden", graph_hidden),
        ("graph_logits", graph_logits),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    projections = getattr(fusion, "target_projections", None)
    if projections is None or len(projections) != len(MODALITIES):
        raise ValueError("fusion must expose three target_projections")

    fusion_buffers = snapshot_named_buffers(fusion)
    classifier_buffers = snapshot_named_buffers(classifier)
    try:
        selected_mask = target_mask.to(device=latents.device, dtype=torch.bool)
        target_statistics: Dict[str, Dict[str, Union[int, float]]] = {}
        for target_index, modality in enumerate(MODALITIES):
            selected = latents[selected_mask[:, target_index], target_index]
            count = int(selected.shape[0])
            if count == 0:
                target_statistics[modality] = {
                    "count": 0,
                    "raw_effective_rank": 0.0,
                    "raw_channel_std": 0.0,
                    "raw_rms": 0.0,
                    "layer_norm_effective_rank": 0.0,
                    "layer_norm_channel_std": 0.0,
                    "layer_norm_rms": 0.0,
                    "linear_output_rms": 0.0,
                    "tanh_saturation_fraction": 0.0,
                }
                continue
            projection = projections[target_index]
            try:
                normalized = projection[0](selected)
                linear_output = projection[1](normalized)
            except (IndexError, TypeError) as error:
                raise ValueError(
                    "each target projection must expose LayerNorm then Linear"
                ) from error
            statistics: Dict[str, Union[int, float]] = {"count": count}
            statistics.update(_latent_statistics(selected, "raw"))
            statistics.update(_latent_statistics(normalized, "layer_norm"))
            statistics["linear_output_rms"] = _rms(linear_output)
            statistics["tanh_saturation_fraction"] = float(
                (torch.tanh(linear_output).abs() >= 0.95)
                .float()
                .mean()
                .item()
            )
            target_statistics[modality] = statistics

        path_logits, residual = compute_path_output(
            graph_hidden,
            latents,
            selected_mask,
            fusion,
            classifier,
        )
        if path_logits.shape != graph_logits.shape:
            raise ValueError("path logits and graph_logits must share a shape")
        logit_shift = path_logits - graph_logits.to(device=path_logits.device)
        diagnostics: Dict[str, Any] = {
            "targets": target_statistics,
            "residual": {
                "rms": _rms(residual),
                "mean_l2": float(
                    residual.float().norm(dim=-1).mean().item()
                    if residual.shape[0]
                    else 0.0
                ),
            },
            "logit_shift": {
                "mean_abs": float(
                    logit_shift.abs().mean().item()
                    if logit_shift.numel()
                    else 0.0
                ),
                "max_abs": float(
                    logit_shift.abs().max().item()
                    if logit_shift.numel()
                    else 0.0
                ),
                "rms": _rms(logit_shift),
            },
        }
        return diagnostics
    finally:
        restore_named_buffers(fusion, fusion_buffers)
        restore_named_buffers(classifier, classifier_buffers)


def conversation_cluster_bootstrap(
    labels: Any,
    left_predictions: Any,
    right_predictions: Any,
    sample_keys: Sequence[str],
    seed: int,
    resamples: int = 2000,
) -> Dict[str, Union[int, float]]:
    """Bootstrap a MOSI W-F1 contrast by whole conversation clusters."""
    target = _numpy_vector(labels, "labels")
    left = _numpy_vector(left_predictions, "left_predictions")
    if left.shape[0] != target.shape[0]:
        raise ValueError("labels and left_predictions must have equal length")
    if torch.is_tensor(right_predictions):
        right_predictions = right_predictions.detach().to(device="cpu").numpy()
    right = np.asarray(right_predictions)
    if right.ndim == 1:
        right = right[None, :]
    if right.ndim != 2:
        raise ValueError("right_predictions must have shape [N] or [K, N]")
    if right.shape[0] == 0 or right.shape[1] != target.shape[0]:
        raise ValueError("right_predictions must contain one or more length-N rows")
    try:
        right_is_finite = np.isfinite(right).all()
    except TypeError as error:
        raise ValueError(
            "right_predictions must contain finite numeric values"
        ) from error
    if not bool(right_is_finite):
        raise ValueError("right_predictions must contain only finite values")
    right = right.astype(np.float64, copy=False)
    keys = tuple(str(key) for key in sample_keys)
    if len(keys) != target.shape[0]:
        raise ValueError("sample_keys must have the same length as labels")
    if len(keys) != len(set(keys)):
        raise ValueError("sample_keys must be unique")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")

    cluster_members: Dict[str, list] = {}
    for index, key in enumerate(keys):
        pieces = key.rsplit(":", 1)
        if len(pieces) != 2 or not pieces[0] or not pieces[1]:
            raise ValueError("sample_keys must end in ':utterance_index'")
        cluster_members.setdefault(pieces[0], []).append(index)
    clusters = tuple(
        np.asarray(indices, dtype=np.int64)
        for indices in cluster_members.values()
    )

    def contrast(indices: np.ndarray) -> float:
        sampled_target = target[indices]
        left_score = _weighted_sign_f1(sampled_target, left[indices])
        right_score = float(
            np.mean(
                [
                    _weighted_sign_f1(sampled_target, row[indices])
                    for row in right
                ]
            )
        )
        return float(left_score - right_score)

    all_indices = np.arange(target.shape[0], dtype=np.int64)
    point_estimate = contrast(all_indices)
    generator = np.random.RandomState(int(seed) % (2 ** 32))
    replicates = np.empty(resamples, dtype=np.float64)
    cluster_count = len(clusters)
    for replicate in range(resamples):
        draw = generator.randint(0, cluster_count, size=cluster_count)
        selected = np.concatenate([clusters[index] for index in draw])
        replicates[replicate] = contrast(selected)
    return {
        "point_estimate": point_estimate,
        "ci_low": float(np.quantile(replicates, 0.025)),
        "ci_high": float(np.quantile(replicates, 0.975)),
        "resamples": int(resamples),
        "conversation_count": int(cluster_count),
    }


def metric_mean_std(
    records: Sequence[Mapping[str, Union[int, float]]]
) -> Dict[str, Dict[str, float]]:
    """Summarize flat metric records with population mean and std."""
    values = tuple(records)
    if not values:
        raise ValueError("records must contain at least one metric mapping")
    keys = set(values[0])
    if any(set(record) != keys for record in values[1:]):
        raise ValueError("records must have the same metric keys")
    summary: Dict[str, Dict[str, float]] = {}
    for key in sorted(keys):
        metric_values = np.asarray(
            [record[key] for record in values], dtype=np.float64
        )
        if not bool(np.isfinite(metric_values).all()):
            raise ValueError("metric values must be finite")
        summary[key] = {
            "mean": float(metric_values.mean()),
            "std": float(metric_values.std(ddof=0)),
        }
    return summary
