"""Pure tensor utilities for the frozen missing-latent oracle diagnostic."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import torch
from torch import nn

from .model import MODALITIES


__all__ = [
    "build_sample_keys",
    "compute_path_output",
    "effective_rank",
    "flatten_valid_lbd",
    "restore_named_buffers",
    "shuffle_targets_by_modality",
    "snapshot_named_buffers",
    "stable_seed",
    "stack_teacher_targets",
    "state_dict_sha256",
    "tensor_sha256",
]


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
