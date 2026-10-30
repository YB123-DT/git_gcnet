"""Original frozen Nested stage two; checkpoint weights are never a parent.

Reuse the exact tested model rather than fork its numerical implementation.
Only that model file is needed from the historical experiment, not its runner,
datasets, Signature caches, or checkpoint setup chain.
"""
import hashlib
from pathlib import Path

import torch

from experiments.osram_frozen_nested_gcnet_20261025.model import FrozenNestedModel as _FrozenNestedModel

METHOD = 'nested_gnn_rooted_evidence'
BLOCK_PARAMETERS = 159235


class FrozenNestedModel(_FrozenNestedModel):
    """Same original model/state tree, with the unified runner's view adapter."""

    def forward_view(self, view):
        return self(view)['prediction']


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def restore_parent(parent, path, *, expected_sha256=None):
    """Restore native parent weights strictly, not optimizer/RNG continuation.

    Architecture, non-parameter configuration and data must already have been
    reconstructed from the matching parent run by the caller. Strict loading
    alone cannot validate settings such as raw versus residualized Gap queries.
    """
    path = Path(path)
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Parent checkpoint SHA256 differs from required identity')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError('Parent checkpoint must contain a model state dictionary')
    if 'block' in payload and not any(k in payload for k in ('model', 'model_state')):
        raise ValueError('Nested block-only checkpoint cannot restore an original Flat parent')
    keys = [k for k in ('model', 'model_state') if k in payload]
    if len(keys) != 1 or not isinstance(payload[keys[0]], dict):
        raise ValueError('Require exactly one native model/model_state entry')
    parent.load_state_dict(payload[keys[0]], strict=True)
    return dict(path=str(path.resolve()), sha256=digest, epoch=payload.get('epoch'),
                restoration='strict model weights only; not optimizer/RNG resume')


def _validate_parent_id(parent_sha256):
    if (not isinstance(parent_sha256, str) or len(parent_sha256) != 64
            or any(c not in '0123456789abcdef' for c in parent_sha256)):
        raise ValueError('Require exact lowercase parent checkpoint SHA256')


def _validate_block(model):
    count = sum(p.numel() for p in model.block.parameters())
    if count != BLOCK_PARAMETERS:
        raise ValueError('Only original Nested dim64/depth3/MLP64 (159235 parameters) is supported')


def block_checkpoint(model, parent_sha256, *, epoch, **metadata):
    """Create a block-only artifact tied to one immutable parent checkpoint."""
    _validate_parent_id(parent_sha256)
    _validate_block(model)
    if set(metadata) & {'block', 'parent_id', 'method', 'version', 'epoch'}:
        raise ValueError('Metadata cannot override checkpoint ownership fields')
    return dict(metadata, version=1, method=METHOD, epoch=int(epoch),
                parent_id=parent_sha256, block=model.block.state_dict())


def restore_block(model, path, parent_sha256):
    """Load original or unified block artifacts, enforcing parent SHA identity."""
    _validate_parent_id(parent_sha256)
    _validate_block(model)
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get('block'), dict):
        raise ValueError('Require a Nested block-only checkpoint, not parent weights')
    if payload.get('parent_id') != parent_sha256:
        raise ValueError('Nested checkpoint belongs to a different parent SHA256')
    identity = payload.get('identity', {})
    method = payload.get('method', identity.get('method', METHOD))
    if method != METHOD:
        raise ValueError('Checkpoint is not the original Nested mechanism')
    model.block.load_state_dict(payload['block'], strict=True)
    return dict(path=str(path.resolve()), sha256=file_sha256(path),
                parent_id=parent_sha256, epoch=payload.get('epoch'), method=METHOD)
