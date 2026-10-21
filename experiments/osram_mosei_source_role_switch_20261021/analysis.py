"""Frozen same-weight comparison; group labels never enter inference."""
import numpy as np

ROLES = ('Base', 'Gap-A', 'Gap-T', 'Gap-V')
SOURCES = ('A', 'T', 'V')


def text_groups(availability, trajectory):
    visible = availability[:, 1].astype(bool)
    seen = trajectory[:, 1, 3] > 0
    return dict(T_visible=visible, T_missing_seen=~visible & seen,
                T_missing_unseen=~visible & ~seen)


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite distribution')
    result = dict(N=len(values), mean=None, std=None, variance=None, min=None, max=None,
                  p01=None, p05=None, median=None, p95=None, p99=None,
                  mean_abs_distance_from_one=None, within_0p01_of_one_fraction=None,
                  within_0p05_of_one_fraction=None, below_one_fraction=None, above_one_fraction=None)
    if len(values):
        result.update(mean=float(values.mean()), std=float(values.std()), variance=float(values.var()),
            min=float(values.min()), max=float(values.max()),
            **{k: float(v) for k, v in zip(('p01','p05','median','p95','p99'), np.quantile(values,[.01,.05,.5,.95,.99]))},
            mean_abs_distance_from_one=float(np.abs(values-1).mean()),
            within_0p01_of_one_fraction=float((np.abs(values-1)<=.01).mean()),
            within_0p05_of_one_fraction=float((np.abs(values-1)<=.05).mean()),
            below_one_fraction=float((values<1).mean()), above_one_fraction=float((values>1).mean()))
    return result


def gamma_statistics(gamma, active):
    return dict(all=distribution(gamma[active]),
        by_source={s:distribution(gamma[:,:,i][active[:,:,i]]) for i,s in enumerate(SOURCES)},
        by_slot={f'{r}/{s}':distribution(gamma[:,ri,si][active[:,ri,si]])
                 for ri,r in enumerate(ROLES) for si,s in enumerate(SOURCES)})


def analyze(arrays):
    from experiments.osram_mosi_source_role_neutral_audit_20261020.analysis import paired
    n = len(arrays['labels'])
    shapes = dict(labels=(n,), prediction_learned=(n,), prediction_neutral=(n,),
        availability=(n,3), trajectory=(n,3,4), active=(n,4,3), gamma=(n,4,3), read_relative_change=(n,4))
    for key, shape in shapes.items():
        if np.asarray(arrays[key]).shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError('Invalid shape or nonfinite '+key)
    av = arrays['availability'];active=arrays['active'].astype(bool);gamma=arrays['gamma']
    if (not np.isin(av,[0,1]).all() or not av.any(-1).all()
            or not np.isin(arrays['active'],[0,1]).all()
            or np.any(gamma[~active]!=0) or np.any((gamma[active]<0)|(gamma[active]>2))
            or np.any(active[:,1:] & av[:,:,None].astype(bool))):
        raise ValueError('Invalid availability/active gamma')
    groups = dict(all=np.ones(n,bool), **text_groups(av, arrays['trajectory']))
    groups.update({''.join(s for i,s in enumerate(SOURCES) if mask & (1<<i)):
                   (av.astype(int)*np.array([1,2,4])).sum(-1)==mask for mask in range(1,8)})
    records, grouped_gamma = {}, {}
    for name, selected in groups.items():
        y=arrays['labels'][selected];on=arrays['prediction_learned'][selected];off=arrays['prediction_neutral'][selected]
        result=paired(y,on,off)
        change=np.abs(on-off);polarity=y!=0
        result.update(mean_abs_prediction_change=float(change.mean()) if len(change) else None,
            max_abs_prediction_change=float(change.max()) if len(change) else None,
            sign_flip_N=int(((on[polarity]>0)!=(off[polarity]>0)).sum()),
            read_relative_change_mean=float(arrays['read_relative_change'][selected].mean()) if selected.any() else None)
        records[name]=result
        grouped_gamma[name]=gamma_statistics(gamma[selected],active[selected])
    return dict(groups=records, gamma=grouped_gamma['all'], grouped_gamma=grouped_gamma)
