"""Pure/reversible helpers for a frozen Text-condition optimization audit.

No optimizer is created; temporary SGD is an explicitly reverted sensitivity
intervention, never a saved model or a test-label-selected prediction.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import math

import numpy as np
import torch
from sklearn.metrics import f1_score


GROUPS = ('ObservedEncoder', 'WriteKey', 'WriteValue', 'Query', 'Memory',
          'LocalFlatHead', 'Other')


def _valid(availability, umask):
    if not torch.is_tensor(availability) or not torch.is_tensor(umask):
        raise ValueError('availability and umask must be torch tensors')
    if availability.ndim != 3 or availability.shape[-1] != 3:
        raise ValueError('availability must be [T,B,3]')
    if umask.shape != (availability.shape[1], availability.shape[0]):
        raise ValueError('umask must be [B,T]')
    for name, value in (('availability', availability), ('umask', umask)):
        if not bool(torch.isfinite(value).all()) or not bool(((value == 0) | (value == 1)).all()):
            raise ValueError(f'{name} must contain finite binary values')
    return umask.to(device=availability.device, dtype=torch.bool).T


def history_text_seen(availability, umask):
    """Return bool [T,B], with current/padded observations excluded from history."""
    valid = _valid(availability, umask)
    text = availability[..., 1].bool() & valid
    count_before = text.long().cumsum(0) - text.long()
    return (count_before > 0) & valid


def paired_masks(availability, umask, target_t, target_b):
    """Clone two masks changing only target Text; never remove its last modality."""
    valid = _valid(availability, umask)
    if (not isinstance(target_t, (int, np.integer)) or not isinstance(target_b, (int, np.integer))
            or isinstance(target_t, bool) or isinstance(target_b, bool)):
        raise ValueError('target indices must be integers')
    if not (0 < target_t < availability.shape[0] and 0 <= target_b < availability.shape[1]):
        raise ValueError('target must be in bounds and strictly later than time zero')
    if not bool(valid[target_t, target_b]) or not bool(valid[:target_t, target_b].any()):
        raise ValueError('target must be valid and have earlier valid history')
    if not bool(availability[target_t, target_b, [0, 2]].bool().any()):
        raise ValueError('target must retain at least one observed A/V modality')
    visible, missing = availability.clone(), availability.clone()
    visible[target_t, target_b, 1] = 1
    missing[target_t, target_b, 1] = 0
    return visible, missing


def _parameters(named_parameters):
    params = list(named_parameters)
    names = [name for name, _ in params]
    if len(set(names)) != len(names) or len({id(p) for _, p in params}) != len(params):
        raise ValueError('named_parameters must be a disjoint unique parameter list')
    for name, p in params:
        if not isinstance(name, str) or not torch.is_tensor(p) or not p.is_floating_point():
            raise ValueError('expected (string, floating tensor) parameter entries')
        if not bool(torch.isfinite(p).all()):
            raise ValueError(f'non-finite parameter: {name}')
    return params


def _gradients(params, gradients):
    if isinstance(gradients, Mapping):
        unknown = set(gradients) - {name for name, _ in params}
        if unknown:
            raise ValueError(f'gradient contains unknown parameters: {sorted(unknown)}')
        values = [gradients.get(name) for name, _ in params]
    else:
        values = list(gradients)
        if len(values) != len(params):
            raise ValueError('gradient sequence must match named_parameters')
    for (name, p), grad in zip(params, values):
        if grad is not None and (not torch.is_tensor(grad) or grad.shape != p.shape
                                 or not grad.is_floating_point() or not bool(torch.isfinite(grad).all())):
            raise ValueError(f'invalid/non-finite gradient: {name}')
    return values


def _group(name):
    # Names verified in model.py: observed_set and smax_fc.  OSRAM owns the
    # projector ModuleDicts, query projector/type embedding and alpha/beta.
    parts = name.split('.')
    if 'observed_set' in parts:
        return 'ObservedEncoder'
    if 'key_projectors' in parts:
        return 'WriteKey'
    if 'value_projectors' in parts:
        return 'WriteValue'
    if any(x in parts for x in ('query_projector', 'query_type_embedding')):
        return 'Query'
    if any(x in parts for x in ('alpha_logits', 'beta_logits')):
        return 'Memory'
    if any(x in parts for x in ('local_path', 'local_skip', 'emotion_adapter', 'emotion_norm', 'smax_fc')):
        return 'LocalFlatHead'
    # node_norm/latent_norm/speaker_embedding/availability_embedding affect
    # multiple paths; do not misleadingly assign them exclusively to Query.
    return 'Other'


def _geometry(params, visible, missing, indices):
    vv = mm = dot = 0.
    visible_defined = missing_defined = 0
    for i in indices:
        _, p = params[i]
        gv, gm = visible[i], missing[i]
        if gv is not None:
            gv = gv.detach().to(device=p.device, dtype=torch.float64)
            vv += float(gv.square().sum())
            visible_defined += 1
        if gm is not None:
            gm = gm.detach().to(device=p.device, dtype=torch.float64)
            mm += float(gm.square().sum())
            missing_defined += 1
        if gv is not None and gm is not None:
            dot += float((gv * gm).sum())
    visible_norm, missing_norm = math.sqrt(vv), math.sqrt(mm)
    cosine = dot / (visible_norm * missing_norm) if visible_norm > 0 and missing_norm > 0 else None
    return {'dot': dot, 'visible_norm': visible_norm, 'missing_norm': missing_norm,
            'cosine': None if cosine is None else max(-1., min(1., cosine)),
            'parameter_count': len(indices), 'numel': sum(params[i][1].numel() for i in indices),
            'visible_defined': visible_defined, 'missing_defined': missing_defined,
            'visible_unused': len(indices)-visible_defined, 'missing_unused': len(indices)-missing_defined}


def gradient_geometry(named_parameters, grads_visible, grads_missing):
    """Disjoint parameter-group gradient geometry; undefined cosine is ``None``."""
    params = _parameters(named_parameters)
    visible = _gradients(params, grads_visible)
    missing = _gradients(params, grads_missing)
    indices = {name: [] for name in GROUPS}
    partition = {name: [] for name in GROUPS}
    for i, (name, _) in enumerate(params):
        group = _group(name)
        indices[group].append(i)
        partition[group].append(name)
    return {'overall': _geometry(params, visible, missing, range(len(params))),
            'groups': {group: _geometry(params, visible, missing, idx) for group, idx in indices.items()},
            'partition': partition,
            'partition_note': 'Shared OSRAM normalization, speaker and availability embeddings are Other, not exclusively Query/Writer.',
            'zero_norm_note': 'A zero norm produces null cosine, never evidence of negative gradient alignment.'}


@contextmanager
def temporary_sgd(named_parameters, gradients, learning_rate=1e-5):
    """Temporarily apply plain SGD, restoring all provided parameter values.

    Does not read/change ``parameter.grad`` or any optimizer state.  Snapshots
    stay on the original device and are restored even if the caller raises.
    Norms describe the actual floating-point update, not a saved training step.
    """
    if not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError('learning_rate must be finite and nonnegative')
    params = _parameters(named_parameters)
    grads = _gradients(params, gradients)
    snapshots = [p.detach().clone() for _, p in params]
    grad_sq = sum(float(g.detach().double().square().sum()) for g in grads if g is not None)
    parameter_sq = sum(float(old.double().square().sum()) for old in snapshots)
    try:
        with torch.no_grad():
            for (_, p), grad in zip(params, grads):
                if grad is not None:
                    p.add_(grad.to(device=p.device, dtype=p.dtype), alpha=-learning_rate)
            update_sq = sum(float((p.detach().double()-old.double()).square().sum())
                            for (_, p), old in zip(params, snapshots))
            if not all(bool(torch.isfinite(p).all()) for _, p in params):
                raise ValueError('temporary SGD produced non-finite parameters')
        parameter_norm, update_norm = math.sqrt(parameter_sq), math.sqrt(update_sq)
        yield {'learning_rate': float(learning_rate), 'gradient_norm': math.sqrt(grad_sq),
               'parameter_norm': parameter_norm, 'update_norm': update_norm,
               'relative_update': update_norm / parameter_norm if parameter_norm else None,
               'parameter_count': len(params), 'defined_gradient_count': sum(g is not None for g in grads)}
    finally:
        with torch.no_grad():
            for (_, p), old in zip(params, snapshots):
                p.copy_(old)


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _metrics(labels, prediction):
    labeled = labels != 0
    mse = float(np.mean((prediction-labels)**2)) if len(labels) else None
    if not labeled.any():
        return {'ACC': None, 'WF1': None, 'MSE': mse}
    truth, predicted = labels[labeled] > 0, prediction[labeled] > 0
    return {'ACC': float(100*np.mean(truth == predicted)),
            'WF1': float(100*f1_score(truth, predicted, average='weighted', zero_division=0)), 'MSE': mse}


def group_summary(labels, full, local, availability, history_seen):
    """MOSI scalar metrics and Local-to-Full corrections by current/history T.

    ACC/WF1 are percentages; MSE includes all valid labels including zero.
    ``local_minus_full_MSE>0`` means Full lowers MSE in that same group.
    """
    labels, full, local = [_numpy(x).reshape(-1).astype(np.float64) for x in (labels, full, local)]
    availability, history_seen = _numpy(availability), _numpy(history_seen).reshape(-1)
    n = len(labels)
    if len(full) != n or len(local) != n or availability.shape != (n, 3) or history_seen.shape != (n,):
        raise ValueError('prediction/label/availability/history sample alignment mismatch')
    if not all(np.isfinite(x).all() for x in (labels, full, local, availability, history_seen)):
        raise ValueError('summary arrays must be finite')
    if not np.isin(availability, [0, 1]).all() or not np.isin(history_seen, [0, 1]).all():
        raise ValueError('availability and history_seen must be binary')
    visible, seen = availability[:, 1].astype(bool), history_seen.astype(bool)
    groups = {'overall': np.ones(n, bool), 'Textvisible': visible,
              'Textmissing_seen': ~visible & seen, 'Textmissing_never': ~visible & ~seen}
    result = {}
    for name, mask in groups.items():
        y, fp, lp = labels[mask], full[mask], local[mask]
        polarity = y != 0
        fc, lc = (fp > 0) == (y > 0), (lp > 0) == (y > 0)
        fm, lm = _metrics(y, fp), _metrics(y, lp)
        result[name] = {'N': int(mask.sum()), 'polarity_N': int(polarity.sum()),
                        'full': fm, 'local': lm,
                        'corrections': int((polarity & ~lc & fc).sum()),
                        'harms': int((polarity & lc & ~fc).sum()),
                        'both_correct': int((polarity & lc & fc).sum()),
                        'both_wrong': int((polarity & ~lc & ~fc).sum()),
                        'local_minus_full_MSE': lm['MSE']-fm['MSE'] if len(y) else None}
    return result
