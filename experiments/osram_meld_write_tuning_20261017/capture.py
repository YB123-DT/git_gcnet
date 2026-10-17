"""Observe historical training without changing its forwards or random stream.

ACC-selected artifacts are copied from the evaluation that selected the weight,
not from the historical trainer's independently resampled final evaluation.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _array(tensor):
    return tensor.detach().cpu().numpy().copy()


def _metrics(artifacts):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    labels, logits = artifacts['labels'], artifacts['logits'].astype(np.float64)
    predictions = logits.argmax(1)
    shifted = logits - logits.max(1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(labels)), labels]
    recall = recall_score(labels, predictions, labels=np.arange(7), average=None, zero_division=0)
    return dict(ACC=100 * float(accuracy_score(labels, predictions)),
                UA=100 * float(recall.mean()),
                WF1=100 * float(f1_score(labels, predictions, average='weighted', zero_division=0)),
                CE=float(ce.mean()), recall=(100 * recall).tolist(),
                support=np.bincount(labels, minlength=7).tolist(), N=len(labels))


class TrainingCapture:
    """Temporary, read-only hooks around b4a3138's existing training calls.

    ``latest`` contains at most eight selection evaluations. Final ``collect=True``
    calls are traced separately and never replace them. ``bind`` should be called
    by TrainingState.bind after the model has been completely initialized.
    """
    def __init__(self, trainer, output):
        self.trainer, self.output = trainer, Path(output)
        self.latest, self.selected_scores = {}, {}
        self.trace = {'train': {}, 'selection': {}, 'final': []}
        self.initial_state_sha256 = None
        self.modelparameters = None
        self._counts, self._digests = {}, {}
        self._active_eval = None
        self._entered = False

    def bind(self, model):
        if self.initial_state_sha256 is not None:
            raise ValueError('Initialization already bound')
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            array = _array(value.contiguous())
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(array.tobytes())
        self.initial_state_sha256 = digest.hexdigest()
        self.modelparameters = sum(p.numel() for p in model.parameters())
        return self

    def _record_mask(self, view, row):
        availability = _array(view['availability'])
        valid = _array(view['umask']).astype(bool)
        speakers = _array(view['qmask'])
        ids = [str(value) for value in view['conversation_ids']]
        if availability.shape != (*valid.T.shape, 3) or speakers.shape != valid.shape:
            raise ValueError('Expected original [T,B,3] availability and [B,T] masks')
        if np.any(availability[~valid.T] != 0):
            raise ValueError('Padding must remain zero in the actual sampler')
        active = availability.transpose(1, 0, 2)[valid]
        if not np.isin(active, [0, 1]).all() or np.any(active.sum(1) == 0):
            raise ValueError('Invalid empty/nonbinary utterance mask')
        digest = self._digests.setdefault(id(row), hashlib.sha256())
        for array in (availability, valid, speakers):
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        digest.update(json.dumps(ids, ensure_ascii=False).encode())
        row['batches'] = row.get('batches', 0) + 1
        row['N'] = row.get('N', 0) + len(active)
        counts = np.bincount((active.astype(int) * np.asarray([1, 2, 4])).sum(1), minlength=8)
        row['pattern_counts'] = (np.asarray(row.get('pattern_counts', [0] * 8)) + counts).tolist()
        observed = active.sum(0).astype(int)
        row['observed_counts'] = (np.asarray(row.get('observed_counts', [0] * 3)) + observed).tolist()
        row['observed_fraction'] = (np.asarray(row['observed_counts']) / row['N']).tolist()
        row['missing_fraction'] = 1 - sum(row['observed_counts']) / (3 * row['N'])
        row['sha256'] = digest.hexdigest()

    def __enter__(self):
        if self._entered:
            raise RuntimeError('TrainingCapture is not reentrant')
        self._entered = True
        self._prepare = self.trainer._prepare_view
        self._evaluate = self.trainer.evaluate_rate
        prepare_signature = inspect.signature(self._prepare)
        evaluate_signature = inspect.signature(self._evaluate)

        @wraps(self._prepare)
        def prepare(*args, **kwargs):
            bound = prepare_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            view = self._prepare(*args, **kwargs)
            schedule = bound.arguments['schedule']
            if self._active_eval is not None:
                self._record_mask(view, self._active_eval['trace'])
                self._active_eval['pending'] = view
            elif schedule.split == 'train':
                key = str(int(bound.arguments['epoch']) + 1)
                row = self.trace['train'].setdefault(key, {})
                self._record_mask(view, row)
            return view

        @wraps(self._evaluate)
        def evaluate(*args, **kwargs):
            if self._active_eval is not None:
                raise RuntimeError('Nested evaluate_rate is unsupported')
            bound = evaluate_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            model, schedule = bound.arguments['model'], bound.arguments['schedule']
            rate = format(float(schedule.requested_missing_rate), '.1f')
            final = bool(bound.arguments['collect'])
            epoch = None if final else self._counts.get(rate, 0) + 1
            row = dict(rate=rate, epoch=epoch, split=schedule.split, final=final)
            active = {'trace': row, 'parts': [], 'pending': None}
            self._active_eval = active

            def forward_hook(module, inputs, output):
                view = active['pending']
                if view is None:
                    raise RuntimeError('Model forward without the original prepared view')
                logits = _array(output[0]).transpose(1, 0, 2)
                valid = _array(view['umask']).astype(bool)
                labels = _array(view['labels'])[valid].astype(np.int64)
                if logits.shape != (*valid.shape, 7) or not np.isfinite(logits).all():
                    raise ValueError('Require finite original seven-class MELD logits')
                ids = np.asarray([str(value) for value in view['conversation_ids']])
                active['parts'].append(dict(logits=logits[valid], labels=labels,
                    availability=_array(view['availability']).transpose(1, 0, 2)[valid],
                    conversation_ids=np.broadcast_to(ids[:, None], valid.shape)[valid].copy(),
                    utterance_indices=np.broadcast_to(np.arange(valid.shape[1]), valid.shape)[valid].copy(),
                    speakers=_array(view['qmask'])[valid]))
                active['pending'] = None

            hook = model.register_forward_hook(forward_hook)
            try:
                result = self._evaluate(*args, **kwargs)
            finally:
                hook.remove()
                self._active_eval = None
            if not active['parts']:
                raise ValueError('Evaluation produced no captured predictions')
            if final:
                self.trace['final'].append(row)
            else:
                artifacts = {name: np.concatenate([part[name] for part in active['parts']])
                             for name in active['parts'][0]}
                artifacts['predictions'] = artifacts['logits'].argmax(1)
                self._counts[rate] = epoch
                self.latest[rate] = dict(epoch=epoch, artifacts=artifacts, scores=_metrics(artifacts))
                self.trace['selection'].setdefault(str(epoch), {})[rate] = row
            return result

        self.trainer._prepare_view, self.trainer.evaluate_rate = prepare, evaluate
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.trainer._prepare_view, self.trainer.evaluate_rate = self._prepare, self._evaluate
        self._entered = False

    def selection_reference(self, epoch, rate):
        key = format(float(rate), '.1f')
        record = self.latest.get(key)
        if record is None or record['epoch'] != epoch:
            raise ValueError('Selected epoch must match the most recent selection evaluation')
        relative = Path('selection_versions') / f'epoch_{epoch:06d}_{key.replace(".", "p")}.npz'
        path = self.output / relative
        artifacts = record['artifacts']
        if path.exists():
            with np.load(path) as saved:
                if set(saved.files) != set(artifacts) | {'selected_epoch'}:
                    raise ValueError('Existing artifact schema changed')
                for name, value in artifacts.items():
                    np.testing.assert_array_equal(saved[name], value)
        else:
            _atomic(path, lambda stream: np.savez_compressed(stream, **artifacts, selected_epoch=np.asarray(epoch)))
        self.selected_scores[key] = dict(record['scores'], epoch=epoch)
        return dict(path=str(relative), sha256=_sha(path),
                    availability_sha256=hashlib.sha256(artifacts['availability'].tobytes()).hexdigest())

    def resume_trace(self, next_epoch):
        """Restore evidence only through the atomically committed epoch boundary.

        ``next_epoch`` is the trainer's zero-based next epoch, hence also the
        count of completed epochs. Trace saving precedes training-state commit,
        so a crash can leave extra rows that must not survive recovery.
        """
        if isinstance(next_epoch, bool) or not isinstance(next_epoch, int) or next_epoch < 0:
            raise ValueError('Expected nonnegative committed next_epoch')
        if self._active_eval is not None:
            raise ValueError('Cannot restore trace during evaluation')
        path = self.output / 'MASK_TRACE.json'
        if not path.exists():
            if next_epoch:
                raise ValueError('Committed training is missing its mask trace')
            payload = {'train': {}, 'selection': {}}
        else:
            payload = json.loads(path.read_text())
            for field, current in (('initial_state_sha256', self.initial_state_sha256),
                                   ('parameters', self.modelparameters)):
                if field in payload and payload[field] is not None and payload[field] != current:
                    raise ValueError('Resume initialization mismatch: ' + field)
        rates = {f'{i / 10:.1f}' for i in range(8)}
        train, selection = payload.get('train', {}), payload.get('selection', {})
        for epoch in range(1, next_epoch + 1):
            key = str(epoch)
            if key not in train or key not in selection or set(selection[key]) != rates:
                raise ValueError(f'Incomplete committed mask trace at epoch {epoch}')
        keys = [str(epoch) for epoch in range(1, next_epoch + 1)]
        self.trace = {'train': {key: train[key] for key in keys},
                      'selection': {key: selection[key] for key in keys}, 'final': []}
        self._counts = {rate: next_epoch for rate in rates}
        self._digests = {}
        self.latest = {}
        self.selected_scores = {}
        return self

    def write_trace(self):
        payload = dict(self.trace, initial_state_sha256=self.initial_state_sha256,
                       parameters=self.modelparameters,
                       note='Actual effective masks; padded rectangle, speaker and ID included in hash; no extra sampler calls')
        encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        _atomic(self.output / 'MASK_TRACE.json', lambda stream: stream.write(encoded))
