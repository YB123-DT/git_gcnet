"""One unchanged causal trajectory per fixed observed set; no training."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

PATTERNS = {'A': (1, 0, 0), 'T': (0, 1, 0), 'V': (0, 0, 1),
            'AT': (1, 1, 0), 'AV': (1, 0, 1), 'TV': (0, 1, 1), 'ATV': (1, 1, 1)}
CHECKPOINT_RATE = {name: {1: '0.7', 2: '0.3', 3: '0.0'}[sum(bits)] for name, bits in PATTERNS.items()}
FIELDS = ('labels', 'conversation_ids', 'utterance_indices', 'speakers')


def fixed_availability(umask, pattern):
    if pattern not in PATTERNS or umask.ndim != 2 or not torch.isin(umask, umask.new_tensor([0, 1])).all():
        raise ValueError('Require a nonempty observed-set name and binary [B,T] padding mask')
    bits = torch.tensor(PATTERNS[pattern], device=umask.device, dtype=torch.float32)
    return torch.where(umask.T.bool().unsqueeze(-1), bits, torch.zeros_like(bits))


def classification_metrics(labels, logits, classes):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    labels, logits = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    if (classes not in (4, 6, 7) or labels.ndim != 1 or not len(labels)
            or logits.shape != (len(labels), classes) or not np.isfinite(logits).all()
            or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= classes):
        raise ValueError('Expected aligned finite classification data with actual class count')
    predictions = logits.argmax(1)
    recall = recall_score(labels, predictions, labels=np.arange(classes), average=None, zero_division=0)
    shifted = logits - logits.max(1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(labels)), labels]
    return dict(ACC=100*float(accuracy_score(labels, predictions)), UA=100*float(recall.mean()),
        WF1=100*float(f1_score(labels, predictions, average='weighted', zero_division=0)),
        CE=float(ce.mean()), recall=(100*recall).tolist(), support=np.bincount(labels, minlength=classes).tolist(), N=len(labels))


def evaluate_condition(model, loader, dimensions, saved, trainer, pattern, classes):
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Provide frozen eval model')
    if saved['logits'].shape != (len(saved['labels']), classes):
        raise ValueError('Wrong selected checkpoint class count')
    np.testing.assert_array_equal(saved['predictions'], saved['logits'].argmax(1))
    parts = {k: [] for k in (*FIELDS, 'availability', 'logits')}
    cursor = scans = 0
    max_error = 0.
    with torch.no_grad():
        for raw in loader:
            data = trainer._move_batch(raw, next(model.parameters()).device)
            valid = data[7].bool()
            mask = valid.cpu().numpy()
            ids = np.asarray([str(v) for v in data[-1]])
            metadata = dict(labels=data[8].cpu().numpy()[mask].astype(np.int64),
                speakers=data[6].cpu().numpy()[mask],
                conversation_ids=np.broadcast_to(ids[:, None], mask.shape)[mask],
                utterance_indices=np.broadcast_to(np.arange(mask.shape[1]), mask.shape)[mask])
            count = len(metadata['labels'])
            for key in FIELDS:
                np.testing.assert_array_equal(metadata[key], saved[key][cursor:cursor+count])
            if pattern is None:
                batch_major = torch.zeros((*valid.shape, 3), device=valid.device)
                batch_major[valid] = torch.as_tensor(saved['availability'][cursor:cursor+count], device=valid.device, dtype=batch_major.dtype)
                availability = batch_major.transpose(0, 1)
            else:
                availability = fixed_availability(data[7], pattern)
            assert not bool(availability[~valid.T].any())
            view = trainer._prepare_view_from_primary_masks(data, availability, availability, dimensions)
            torch.testing.assert_close(view['availability'], availability, rtol=0, atol=0)
            logits, _, _, completion = model([view['incomplete']], view['availability'],
                view['qmask'], view['umask'], view['lengths'], predict_missing=False)
            if completion is not None:
                raise RuntimeError('Completion must remain off')
            current = logits.transpose(0, 1)[valid].cpu().numpy()
            if current.shape != (count, classes) or not np.isfinite(current).all():
                raise ValueError('Invalid task logits')
            if pattern is None:
                expected = saved['logits'][cursor:cursor+count]
                np.testing.assert_allclose(current, expected, rtol=2e-5, atol=2e-5)
                np.testing.assert_array_equal(current.argmax(1), saved['predictions'][cursor:cursor+count])
                max_error = max(max_error, float(np.abs(current-expected).max(initial=0)))
            elif not np.array_equal(availability.transpose(0, 1)[valid].cpu().numpy(), np.broadcast_to(PATTERNS[pattern], (count, 3))):
                raise ValueError('Persistent pattern changed across time or speaker')
            for key in FIELDS:
                parts[key].append(metadata[key])
            parts['logits'].append(current)
            parts['availability'].append(availability.transpose(0, 1)[valid].cpu().numpy())
            cursor += count
            scans += 1
    if cursor != len(saved['labels']):
        raise ValueError('Selected cohort not fully consumed')
    artifacts = {key: np.concatenate(value) for key, value in parts.items()}
    artifacts['predictions'] = artifacts['logits'].argmax(1)
    metrics = classification_metrics(artifacts['labels'], artifacts['logits'], classes)
    checks = dict(metadata_exact=True, scans=scans, selected_classes_exact=pattern is None,
        selected_logit_max_abs_error=max_error if pattern is None else None,
        availability_sha256=hashlib.sha256(artifacts['availability'].tobytes()).hexdigest())
    return dict(metrics=metrics, checks=checks), artifacts
