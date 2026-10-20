"""Paired, frozen gamma=1 interventions; never fit or select a model here."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


ROLES = ('Base', 'Gap-A', 'Gap-T', 'Gap-V')
SOURCES = ('A', 'T', 'V')


def metrics(labels, prediction):
    """Original MOSI: exclude label zero for polarity, include it in MSE."""
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    polarity = labels != 0
    result = {'N': int(labels.size), 'N_polarity': int(polarity.sum()),
              'MSE': float(np.mean((labels - prediction) ** 2)) if labels.size else None,
              'ACC': None, 'WF1': None}
    if polarity.any():
        y, p = labels[polarity] > 0, prediction[polarity] > 0
        result.update(ACC=100. * float(accuracy_score(y, p)),
                      WF1=100. * float(f1_score(y, p, average='weighted', zero_division=0)))
    return result


def paired(labels, learned, reset):
    """Positive gains mean learned calibration helps relative to gamma=1."""
    on, off = metrics(labels, learned), metrics(labels, reset)
    polarity = labels != 0
    correct_on = (learned[polarity] > 0) == (labels[polarity] > 0)
    correct_off = (reset[polarity] > 0) == (labels[polarity] > 0)
    return {'N': on['N'], 'N_polarity': on['N_polarity'], 'learned': on, 'reset': off,
            'gain_WF1_pp': on['WF1'] - off['WF1'] if polarity.any() else None,
            'gain_ACC_pp': on['ACC'] - off['ACC'] if polarity.any() else None,
            'delta_MSE': on['MSE'] - off['MSE'] if on['N'] else None,
            'corrections': int((correct_on & ~correct_off).sum()),
            'harms': int((~correct_on & correct_off).sum()),
            'both_correct': int((correct_on & correct_off).sum()),
            'both_wrong': int((~correct_on & ~correct_off).sum())}


def analyze_arrays(arrays):
    """Analyze aligned utterances from one seed/rate's fixed BEST checkpoint.

    Slots are neutralized, not removed: gamma_i=1 versus its learned value.
    Predictions must already share inputs, memory scan, masks and sample order.
    """
    names = ('labels', 'availability', 'active', 'gamma', 'prediction_learned',
             'prediction_neutral', 'prediction_reset', 'prediction_flat')
    data = {name: np.asarray(arrays[name]) for name in names}
    if data['labels'].ndim != 1 or not data['labels'].size:
        raise ValueError('Require nonempty 1-D aligned labels')
    n = len(data['labels'])
    expected = {'labels': (n,), 'availability': (n, 3), 'active': (n, 4, 3),
                'gamma': (n, 4, 3), 'prediction_reset': (n, 4, 3),
                'prediction_learned': (n,), 'prediction_neutral': (n,),
                'prediction_flat': (n,)}
    for name, shape in expected.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f'{name}: require finite shape {shape}')
    if not np.isin(data['availability'], [0, 1]).all() or not data['availability'].any(-1).all():
        raise ValueError('Every utterance requires a nonempty binary availability')
    if not np.isin(data['active'], [0, 1]).all():
        raise ValueError('Active must be binary')
    active = data['active'].astype(bool)
    if (active[:, 1:] & data['availability'][:, :, None].astype(bool)).any():
        raise ValueError('Observed target Gap cannot be active')
    if (data['gamma'][active] < 0).any() or (data['gamma'][active] > 2).any():
        raise ValueError('Direct 2*sigmoid gamma must be in [0,2]')
    if (data['gamma'][~active] != 0).any():
        raise ValueError('Inactive gamma must be exactly zero')
    learned = data['prediction_learned']
    inactive = ~active
    repeated = np.broadcast_to(learned[:, None, None], (n, 4, 3))
    if not np.array_equal(data['prediction_reset'][inactive], repeated[inactive]):
        raise ValueError('Inactive-slot neutralization changed predictions')
    labels = data['labels']
    models = {name: metrics(labels, data[f'prediction_{name}'])
              for name in ('learned', 'neutral', 'flat')}
    comparisons = {}
    for label, left, right in (('total', 'learned', 'flat'),
                               ('mainpath', 'neutral', 'flat'),
                               ('direct', 'learned', 'neutral')):
        comparisons[label] = {
            'WF1_pp': models[left]['WF1'] - models[right]['WF1'] if models[left]['WF1'] is not None else None,
            'ACC_pp': models[left]['ACC'] - models[right]['ACC'] if models[left]['ACC'] is not None else None,
            'MSE': models[left]['MSE'] - models[right]['MSE'],
            'corrections': paired(labels, data[f'prediction_{left}'], data[f'prediction_{right}'])['corrections'],
            'harms': paired(labels, data[f'prediction_{left}'], data[f'prediction_{right}'])['harms']}
    slots = {}
    for role_index, role in enumerate(ROLES):
        for source_index, source in enumerate(SOURCES):
            selection = active[:, role_index, source_index]
            reset = data['prediction_reset'][:, role_index, source_index]
            values = data['gamma'][selection, role_index, source_index]
            stats = {'N': int(values.size), 'mean': None, 'std': None,
                     'min': None, 'max': None, 'below_one_fraction': None,
                     'above_one_fraction': None}
            if values.size:
                stats.update(mean=float(values.mean()), std=float(values.std()),
                             min=float(values.min()), max=float(values.max()),
                             below_one_fraction=float((values < 1).mean()),
                             above_one_fraction=float((values > 1).mean()))
            slots[f'{role}/{source}'] = {
                'active_cohort': paired(labels[selection], learned[selection], reset[selection]),
                'full_test': paired(labels, learned, reset), 'gamma': stats}
    return {'label': 'INTERNAL DIAGNOSTIC ONLY', 'models': models,
            'comparisons': comparisons, 'slots': slots,
            'definitions': {'gain': 'learned minus gamma=1 reset; positive WF1/ACC is useful calibration',
                            'MSE': 'all valid utterances; lower is better',
                            'polarity': 'labels != 0, prediction > 0',
                            'inactive': 'active-cohort metrics N/A, full-test gain zero'}}
