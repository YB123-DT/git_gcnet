"""Strict transfer of supervised ONLINE projectors into a fixed target bank."""
from pathlib import Path

import torch
from .b2_training import file_sha256, state_sha256, subset

PREFIX = 'observed_set.projectors.'
FORMAT = 'supervised-modality-projectors-v1'


def read_source(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    if (checkpoint.get('selection_split') != 'validation'
            or config.get('checkpoint_selection') != 'validation'
            or config.get('training_objective') != 'emotion-only'
            or config.get('train_rate_mode') != 'fixed'
            or config.get('fixed_missing_rate') != 0.
            or config.get('backbone_type') != 'osram'
            or config.get('osram_bidirectional', True)
            or config.get('osram_write_step') != .6
            or config.get('osram_readout_fusion', 'flat') != 'flat'
            or config.get('fusion_type') != 'mean'
            or config.get('classification_completion', False)
            or config.get('completion_path', 'none') != 'none'):
        raise ValueError('Teacher source must be complete-view emotion-only causal .6 Flat, validation-selected')
    if not isinstance(checkpoint.get('model'), dict):
        raise ValueError('Teacher source has no model state')
    projectors = subset(checkpoint['model'], PREFIX)
    if not projectors:
        raise ValueError('Teacher source lacks observed_set.projectors.*; never use teacher.*')
    for name, value in projectors.items():
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid Teacher tensor: {name}')
    fingerprint = state_sha256(projectors)
    if checkpoint.get('format') == FORMAT:
        if checkpoint.get('projector_sha256') != fingerprint:
            raise ValueError('Exported Teacher projector hash mismatch')
        if set(checkpoint['model']) != {PREFIX+k for k in projectors}:
            raise ValueError('Projector export contains unexpected model keys')
    return checkpoint, projectors, fingerprint


def load_pretrained_teacher(teacher, path):
    checkpoint, projectors, fingerprint = read_source(path)
    expected = teacher.state_dict()
    if set(projectors) != set(expected):
        raise ValueError(f'Teacher projector keys differ: missing={sorted(set(expected)-set(projectors))}, '
                         f'unexpected={sorted(set(projectors)-set(expected))}')
    for name, value in projectors.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f'Teacher shape/dtype mismatch: {name}')
    teacher.load_state_dict(projectors, strict=True)
    teacher.requires_grad_(False).eval()
    if state_sha256(teacher.state_dict()) != fingerprint:
        raise RuntimeError('Teacher transfer hash differs from exported projectors')
    return dict(checkpoint=str(Path(path).resolve()), checkpoint_sha256=file_sha256(path),
                source_prefix=PREFIX, projector_sha256=fingerprint,
                source_config=checkpoint['config'], source_epoch=checkpoint.get('epoch'),
                selection_split=checkpoint['selection_split'],
                source_validation_weighted_f1=checkpoint.get('validation_mean_weighted_f1'),
                diagnostic_only=checkpoint.get('diagnostic_only', False),
                parent_checkpoint_sha256=checkpoint.get('parent_checkpoint_sha256'))


def export_teacher_projectors(source, destination):
    """Export selected online weights; retain task and checkpoint provenance."""
    checkpoint, projectors, fingerprint = read_source(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format=FORMAT, model={PREFIX+k: v for k,v in projectors.items()},
                   config=checkpoint['config'], epoch=checkpoint.get('epoch'),
                   selection_split='validation',
                   validation_mean_weighted_f1=checkpoint.get('validation_mean_weighted_f1'),
                   projector_sha256=fingerprint, source_prefix=PREFIX,
                   parent_checkpoint=str(Path(source).resolve()),
                   parent_checkpoint_sha256=file_sha256(source),
                   diagnostic_only=checkpoint.get('diagnostic_only', False))
    # Refuse to silently replace a previously exported Teacher.
    with destination.open('xb') as stream:
        torch.save(payload, stream)
    return payload
