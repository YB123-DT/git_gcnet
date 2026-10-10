"""Label-free four-condition masks and paired task contributions."""
import numpy as np


def select_plan(mask, target):
    a = np.asarray(mask)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isin(a, [0, 1]).all():
        raise ValueError('Expected binary [time,3]')
    if not 0 <= target < len(a) or (a.sum(-1) == 0).any():
        raise ValueError('Invalid target or empty utterance')
    candidates = np.flatnonzero((a[:target, 1] == 1) & (a[:target].sum(-1) > 1))
    if not len(candidates):
        return None
    i = int(candidates[-1])
    others = [(p, m) for p in range(target) if p != i and a[p].sum() > 1
              for m in (0, 2) if a[p, m] == 1]
    if not others:
        return None
    p, m = min(others, key=lambda x: (abs(x[0]-i), -x[0], x[1]))
    return dict(text_position=i, other_position=p, other_modality=m)


def four_masks(original, target, plan):
    from experiments.osram_frozen_memory_audit_20261009.intervention import delete_bit
    a = np.asarray(original)
    i, p, m = (plan[k] for k in ('text_position', 'other_position', 'other_modality'))
    if p == i or m not in (0, 2):
        raise ValueError('Background deletion must exclude focal whole utterance and Text')
    result = {'A+': a.copy(), 'A-': delete_bit(a, target, (i, 1))}
    result['B+'] = delete_bit(a, target, (p, m))
    result['B-'] = delete_bit(result['B+'], target, (i, 1))
    assert np.array_equal(result['A+'][i], result['B+'][i])
    assert np.array_equal(result['A-'][i], result['B-'][i])
    for value in result.values():
        assert np.array_equal(value[target:], a[target:])
        assert (value.sum(-1) > 0).all()
    assert np.array_equal(result['A+'][:, 1], result['B+'][:, 1])
    assert np.array_equal(result['A-'][:, 1], result['B-'][:, 1])
    return result


def contributions(y, predictions):
    if not np.isfinite([y, *predictions.values()]).all():
        raise ValueError('Nonfinite label/prediction')
    delta = {k: (y-predictions[k+'-'])**2-(y-predictions[k+'+'])**2 for k in ('A', 'B')}
    result = dict(delta_A=delta['A'], delta_B=delta['B'], interaction=delta['B']-delta['A'])
    for k in ('A', 'B'):
        on, off = predictions[k+'+'] > 0, predictions[k+'-'] > 0
        result[k+'_rescue'] = bool(y != 0 and on == (y > 0) and off != (y > 0))
        result[k+'_harm'] = bool(y != 0 and off == (y > 0) and on != (y > 0))
    return result


def summarize(rows):
    from experiments.osram_frozen_memory_audit_20261009.probe import metrics
    if not rows:
        return {'n': 0}
    y = np.array([r['label'] for r in rows])
    da, db, j = (np.array([r[k] for r in rows]) for k in ('delta_A', 'delta_B', 'interaction'))
    result = dict(n=len(rows), nonneutral_n=int(np.count_nonzero(y)),
                  delta_A_mean=float(da.mean()), delta_B_mean=float(db.mean()),
                  interaction_mean=float(j.mean()), interaction_abs_mean=float(np.abs(j).mean()),
                  interaction_quantiles={str(q): float(np.quantile(j, q)) for q in (.05, .25, .5, .75, .95)},
                  helpful_to_harmful=int(((da > 0) & (db < 0)).sum()),
                  harmful_to_helpful=int(((da < 0) & (db > 0)).sum()),
                  helpful_to_harmful_deadband=int(((da > 1e-6) & (db < -1e-6)).sum()),
                  harmful_to_helpful_deadband=int(((da < -1e-6) & (db > 1e-6)).sum()))
    for arm in ('A+', 'A-', 'B+', 'B-'):
        result[arm] = metrics(y, np.array([r['predictions'][arm] for r in rows]))
    for field in ('A_rescue', 'A_harm', 'B_rescue', 'B_harm'):
        result[field] = sum(r[field] for r in rows)
    return result
