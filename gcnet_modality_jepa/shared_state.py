from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional, Union

import torch
from torch import nn


SHARED_CHECKPOINT_FORMAT = "gcnet-shared-state"
SHARED_CHECKPOINT_VERSION = 1
SHARED_CHECKPOINT_FIELDS = frozenset(
    {"format", "version", "seed", "shared_hash", "tensors"}
)
SHARED_STATE_PREFIXES = (
    "lstm.",
    "gru.",
    "graph_net_temporal.",
    "graph_net_speaker.",
    "smax_fc.",
)

StateSource = Union[nn.Module, Mapping[str, torch.Tensor]]


class SharedStateParityError(AssertionError):
    """Raised when two GCNet shared states are not exactly equal."""


def extract_shared_state(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """Return sorted, detached CPU clones of the shared GCNet state."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    return OrderedDict(
        (name, tensor.detach().cpu().contiguous().clone())
        for name, tensor in sorted(model.state_dict().items())
        if name.startswith(SHARED_STATE_PREFIXES)
    )


def _state_mapping(source: StateSource) -> Mapping[str, torch.Tensor]:
    if isinstance(source, nn.Module):
        return extract_shared_state(source)
    if not isinstance(source, Mapping):
        raise TypeError("shared state must be a module or a tensor mapping")
    return source


def _canonical_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("shared state value for {!r} must be a tensor".format(name))
    if tensor.layout != torch.strided:
        raise TypeError("shared state tensor {!r} must use strided layout".format(name))
    return tensor.detach().cpu().contiguous()


def _update_field(digest: "hashlib._Hash", value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def shared_state_hash(source: StateSource) -> str:
    """Hash sorted names, dtypes, shapes, and raw contiguous tensor bytes."""
    state = _state_mapping(source)
    digest = hashlib.sha256()
    for name in sorted(state):
        if not isinstance(name, str):
            raise TypeError("shared state keys must be strings")
        tensor = _canonical_tensor(state[name], name)
        _update_field(digest, name.encode("utf-8"))
        _update_field(digest, str(tensor.dtype).encode("ascii"))
        shape = ",".join(str(dimension) for dimension in tensor.shape)
        _update_field(digest, shape.encode("ascii"))
        _update_field(digest, tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _first_shared_state_mismatch(
    expected: StateSource, actual: StateSource
) -> Optional[str]:
    expected_state = _state_mapping(expected)
    actual_state = _state_mapping(actual)
    expected_names = set(expected_state)
    actual_names = set(actual_state)

    missing = sorted(expected_names - actual_names)
    if missing:
        return "missing shared key {!r}".format(missing[0])
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        return "unexpected shared key {!r}".format(unexpected[0])

    for name in sorted(expected_names):
        expected_tensor = _canonical_tensor(expected_state[name], name)
        actual_tensor = _canonical_tensor(actual_state[name], name)
        if expected_tensor.dtype != actual_tensor.dtype:
            return "shared tensor {!r} dtype differs: {} != {}".format(
                name, expected_tensor.dtype, actual_tensor.dtype
            )
        if expected_tensor.shape != actual_tensor.shape:
            return "shared tensor {!r} shape differs: {} != {}".format(
                name, tuple(expected_tensor.shape), tuple(actual_tensor.shape)
            )
        if not torch.equal(expected_tensor, actual_tensor):
            different = torch.ne(expected_tensor, actual_tensor).reshape(-1)
            first_index = int(torch.nonzero(different, as_tuple=False)[0].item())
            expected_value = expected_tensor.reshape(-1)[first_index].item()
            actual_value = actual_tensor.reshape(-1)[first_index].item()
            return (
                "shared tensor {!r} values differ at flat index {}: {!r} != {!r}"
            ).format(name, first_index, expected_value, actual_value)
    return None


def compare_shared_state(expected: StateSource, actual: StateSource) -> bool:
    """Return whether two modules or shared-state mappings are exactly equal."""
    return _first_shared_state_mismatch(expected, actual) is None


def assert_shared_state_parity(expected: StateSource, actual: StateSource) -> None:
    """Raise with the first useful mismatch when shared states are not equal."""
    mismatch = _first_shared_state_mismatch(expected, actual)
    if mismatch is not None:
        raise SharedStateParityError(mismatch)


def save_shared_checkpoint(
    path: Union[str, os.PathLike], model: nn.Module, seed: int
) -> str:
    """Atomically save shared tensors and return their canonical SHA-256 hash."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors = extract_shared_state(model)
    state_hash = shared_state_hash(tensors)
    payload = {
        "format": SHARED_CHECKPOINT_FORMAT,
        "version": SHARED_CHECKPOINT_VERSION,
        "seed": seed,
        "shared_hash": state_hash,
        "tensors": tensors,
    }

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(destination.parent),
            prefix=".{}.".format(destination.name),
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(str(temporary_path), str(destination))
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise
    return state_hash


def _load_payload(path: Union[str, os.PathLike]) -> Mapping[str, object]:
    try:
        payload = torch.load(str(path), map_location="cpu")
    except Exception as error:
        raise ValueError("shared checkpoint is unreadable or corrupt") from error
    if not isinstance(payload, Mapping):
        raise ValueError("shared checkpoint payload must be a mapping")

    payload_fields = set(payload)
    missing_fields = sorted(SHARED_CHECKPOINT_FIELDS - payload_fields)
    if missing_fields:
        raise ValueError(
            "shared checkpoint is missing fields: {}".format(", ".join(missing_fields))
        )
    unexpected_fields = sorted(payload_fields - SHARED_CHECKPOINT_FIELDS)
    if unexpected_fields:
        raise ValueError(
            "shared checkpoint has unexpected fields: {}".format(
                ", ".join(unexpected_fields)
            )
        )
    if payload["format"] != SHARED_CHECKPOINT_FORMAT:
        raise ValueError(
            "unexpected shared checkpoint format: {!r}".format(payload["format"])
        )
    if (
        type(payload["version"]) is not int
        or payload["version"] != SHARED_CHECKPOINT_VERSION
    ):
        raise ValueError(
            "unsupported shared checkpoint version: {!r}".format(
                payload["version"]
            )
        )
    if type(payload["seed"]) is not int:
        raise ValueError("shared checkpoint seed must be an integer")
    if not isinstance(payload["shared_hash"], str):
        raise ValueError("shared checkpoint shared_hash must be a string")
    if not isinstance(payload["tensors"], Mapping):
        raise ValueError("shared checkpoint tensors must be a mapping")
    return payload


def load_shared_checkpoint(
    path: Union[str, os.PathLike],
    model: nn.Module,
    expected_hash: Optional[str] = None,
) -> str:
    """Validate and strictly load shared keys without changing variant heads."""
    payload = _load_payload(path)
    tensors = payload["tensors"]
    checkpoint_hash = payload["shared_hash"]
    actual_hash = shared_state_hash(tensors)
    if checkpoint_hash != actual_hash:
        raise ValueError(
            "shared checkpoint hash mismatch; checkpoint may be corrupt: {} != {}".format(
                checkpoint_hash, actual_hash
            )
        )
    if expected_hash is not None and checkpoint_hash != expected_hash:
        raise ValueError(
            "shared checkpoint hash does not match required hash: {} != {}".format(
                checkpoint_hash, expected_hash
            )
        )

    expected_state = extract_shared_state(model)
    expected_names = set(expected_state)
    checkpoint_names = set(tensors)
    missing = sorted(expected_names - checkpoint_names)
    if missing:
        raise ValueError("missing shared keys: {}".format(", ".join(missing)))
    unexpected = sorted(checkpoint_names - expected_names)
    if unexpected:
        raise ValueError("unexpected shared keys: {}".format(", ".join(unexpected)))

    for name in sorted(expected_names):
        checkpoint_tensor = _canonical_tensor(tensors[name], name)
        expected_tensor = expected_state[name]
        if checkpoint_tensor.dtype != expected_tensor.dtype:
            raise ValueError(
                "shared tensor {!r} dtype mismatch: {} != {}".format(
                    name, checkpoint_tensor.dtype, expected_tensor.dtype
                )
            )
        if checkpoint_tensor.shape != expected_tensor.shape:
            raise ValueError(
                "shared tensor {!r} shape mismatch: {} != {}".format(
                    name, tuple(checkpoint_tensor.shape), tuple(expected_tensor.shape)
                )
            )

    full_state = model.state_dict()
    for name in sorted(expected_names):
        full_state[name] = tensors[name]
    model.load_state_dict(full_state, strict=True)
    return checkpoint_hash
