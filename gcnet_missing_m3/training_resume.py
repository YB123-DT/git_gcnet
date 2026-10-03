"""Atomic epoch-boundary recovery, separate from inference BEST checkpoints.

One immutable model version serves all rate improvements in an epoch. Canonical
BEST files and history are repairable views of the last committed full state.
Versions are intentionally retained; disk admission must budget for them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import uuid

import numpy as np
import torch


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(path, data):
    encoded = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _atomic(path, lambda stream: stream.write(encoded))


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


class TrainingState:
    def __init__(self, output_dir, *, identity, schedule_identity=None):
        if not isinstance(identity, dict) or not identity:
            raise ValueError('A nonempty source/protocol/data identity is required')
        self.output = Path(output_dir)
        self.identity = copy.deepcopy(identity)
        self.schedule_identity = copy.deepcopy(schedule_identity)
        self.references = {}
        self._epoch_models = {}
        self.model = self.optimizer = self.scheduler = self.scaler = None

    def bind(self, model, optimizer, *, scheduler=None, scaler=None):
        self.model, self.optimizer = model, optimizer
        self.scheduler, self.scaler = scheduler, scaler
        return self

    def _bound(self):
        if self.model is None or self.optimizer is None:
            raise RuntimeError('bind model and optimizer before training-state operations')

    def _reference(self, path):
        return {'path': str(path.relative_to(self.output)), 'sha256': _sha(path)}

    def _checked(self, reference):
        path = (self.output / reference['path']).resolve()
        if not path.is_relative_to(self.output.resolve()):
            raise ValueError('Version reference escapes output directory')
        if not path.is_file() or _sha(path) != reference['sha256']:
            raise ValueError(f'Checkpoint version integrity/hash mismatch: {path.name}')
        return path

    def _canonical(self, rate, reference):
        model = torch.load(self._checked(reference['model']), map_location='cpu', weights_only=False)
        metadata = json.loads(self._checked(reference['selection']).read_text())
        checkpoint = dict(metadata['checkpoint'], model=model)
        path = self.output / ('best_miss_' + rate.replace('.', 'p') + '.pt')
        _atomic(path, lambda stream: torch.save(checkpoint, stream))

    def save_best(self, *, epoch, rate, checkpoint):
        self._bound()
        if not isinstance(epoch, int) or epoch < 1 or rate not in {f'{i / 10:.1f}' for i in range(8)}:
            raise ValueError('Expected one-based epoch and canonical missing rate')
        if checkpoint.get('epoch') != epoch or 'model' not in checkpoint:
            raise ValueError('BEST payload must contain matching epoch and model')
        versions = self.output / 'training_versions'
        versions.mkdir(parents=True, exist_ok=True)
        if epoch not in self._epoch_models:
            model_path = versions / f'epoch_{epoch:06d}_{uuid.uuid4().hex}_model.pt'
            model_state = _cpu(checkpoint['model'])
            _atomic(model_path, lambda stream: torch.save(model_state, stream))
            self._epoch_models[epoch] = self._reference(model_path)
        model_reference = self._epoch_models[epoch]
        metadata_path = versions / f'epoch_{epoch:06d}_{rate.replace(".", "p")}_{uuid.uuid4().hex}.json'
        metadata = {key: value for key, value in checkpoint.items() if key != 'model'}
        _json(metadata_path, {'checkpoint': metadata, 'model': model_reference})
        reference = {'model': copy.deepcopy(model_reference), 'selection': self._reference(metadata_path)}
        self._canonical(rate, reference)
        self.references[rate] = reference

    def commit_epoch(self, *, next_epoch, history, selection_state, schedule_state=None):
        self._bound()
        if not isinstance(next_epoch, int) or next_epoch < 1:
            raise ValueError('next_epoch must be the completed epoch count')
        if [row.get('epoch') for row in history] != list(range(1, next_epoch + 1)):
            raise ValueError('Committed history must contain each completed epoch exactly once')
        for reference in self.references.values():
            self._checked(reference['model'])
            self._checked(reference['selection'])
        state = {
            'schema_version': 1, 'identity': self.identity,
            'schedule_identity': self.schedule_identity,
            'model': _cpu(self.model.state_dict()), 'optimizer': _cpu(self.optimizer.state_dict()),
            'scheduler': None if self.scheduler is None else _cpu(self.scheduler.state_dict()),
            'scaler': None if self.scaler is None else _cpu(self.scaler.state_dict()),
            'next_epoch': next_epoch, 'history': copy.deepcopy(history),
            'selection_state': copy.deepcopy(selection_state),
            'schedule_state': copy.deepcopy(schedule_state),
            'best_references': copy.deepcopy(self.references),
            'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None},
        }
        _json(self.output / 'history.json', history)
        _atomic(self.output / 'last_training.pt', lambda stream: torch.save(state, stream))

    def restore(self):
        self._bound()
        path = self.output / 'last_training.pt'
        if not path.exists():
            if list(self.output.glob('best_miss_*.pt')) or (self.output / 'history.json').exists():
                raise ValueError('Existing artifacts lack complete last_training state; use a new attempt')
            return None
        state = torch.load(path, map_location='cpu', weights_only=False)
        required = {'schema_version', 'identity', 'schedule_identity', 'model', 'optimizer',
                    'scheduler', 'scaler', 'next_epoch', 'history', 'selection_state',
                    'schedule_state', 'best_references', 'rng'}
        if not isinstance(state, dict) or required - state.keys() or state.get('schema_version') != 1:
            raise ValueError('Not a compatible complete last_training checkpoint')
        if state['identity'] != self.identity or state['schedule_identity'] != self.schedule_identity:
            raise ValueError('Training source/protocol/data/schedule identity mismatch')
        for key in ('scheduler', 'scaler'):
            if (getattr(self, key) is None) != (state[key] is None):
                raise ValueError(f'Applicable {key} state mismatch')
        if [row.get('epoch') for row in state['history']] != list(range(1, state['next_epoch'] + 1)):
            raise ValueError('Committed history/epoch mismatch')
        for reference in state['best_references'].values():
            self._checked(reference['model'])
            self._checked(reference['selection'])
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        for key in ('scheduler', 'scaler'):
            if getattr(self, key) is not None:
                getattr(self, key).load_state_dict(state[key])
        self.references = copy.deepcopy(state['best_references'])
        self._epoch_models = {}
        # Repair all canonical files before restoring RNG. Uncommitted model
        # versions remain available for audit but never enter selected metrics.
        for rate, reference in self.references.items():
            self._canonical(rate, reference)
        _json(self.output / 'history.json', state['history'])
        rng = state['rng']
        if not {'python', 'numpy', 'torch', 'cuda'} <= rng.keys():
            raise ValueError('Incomplete RNG state')
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if rng['cuda'] is not None:
            if not torch.cuda.is_available() or len(rng['cuda']) != torch.cuda.device_count():
                raise ValueError('CUDA RNG device count mismatch')
            torch.cuda.set_rng_state_all(rng['cuda'])
        return {key: copy.deepcopy(state[key]) for key in
                ('next_epoch', 'history', 'selection_state', 'schedule_state')}
