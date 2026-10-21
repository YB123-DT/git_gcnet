"""Descriptive eight-rate/group effects and active-only gamma distributions."""
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from .analysis import gamma_statistics, text_groups


def summarize(output,records,arrays):
    output=Path(output)
    if set(records)!={f'{i/10:.1f}' for i in range(8)} or len(arrays)!=8:
        raise ValueError('Require all eight completed replay conditions')
    combined={k:np.concatenate([a[k] for a in arrays]) for k in ('gamma','active','availability','trajectory')}
    groups=dict(all=np.ones(len(combined['gamma']),bool),**text_groups(combined['availability'],combined['trajectory']))
    gamma={g:gamma_statistics(combined['gamma'][v],combined['active'][v].astype(bool)) for g,v in groups.items()}
    aggregate={};rows=[]
    for scope,rates in (('mean',tuple(records)),('high',('0.5','0.6','0.7'))):
        aggregate[scope]={}
        for group in records['0.0']['statistics']['groups']:
            values=[records[r]['statistics']['groups'][group] for r in rates]
            valid=[v for v in values if v['N_polarity']]
            aggregate[scope][group]=dict(rates_with_polarity=len(valid),
                N=sum(v['N'] for v in values),N_polarity=sum(v['N_polarity'] for v in values),
                corrections=sum(v['corrections'] for v in values),harms=sum(v['harms'] for v in values),
                learned={m:statistics.mean(v['learned'][m] for v in valid) if valid else None for m in ('ACC','WF1')},
                neutral={m:statistics.mean(v['reset'][m] for v in valid) if valid else None for m in ('ACC','WF1')},
                gain_WF1_pp=statistics.mean(v['gain_WF1_pp'] for v in valid) if valid else None,
                sign_flip_N=sum(v['sign_flip_N'] for v in values))
    for rate,record in records.items():
        for group,v in record['statistics']['groups'].items():
            rows.append(dict(rate=rate,group=group,N=v['N'],N_polarity=v['N_polarity'],
                learned_ACC=v['learned']['ACC'],neutral_ACC=v['reset']['ACC'],
                learned_WF1=v['learned']['WF1'],neutral_WF1=v['reset']['WF1'],
                gain_WF1_pp=v['gain_WF1_pp'],delta_MSE=v['delta_MSE'],corrections=v['corrections'],harms=v['harms'],
                sign_flip_N=v['sign_flip_N'],mean_abs_prediction_change=v['mean_abs_prediction_change'],
                max_abs_prediction_change=v['max_abs_prediction_change'],read_relative_change_mean=v['read_relative_change_mean']))
    result=dict(label='INTERNAL DIAGNOSTIC ONLY',aggregate=aggregate,gamma=gamma,per_rate=records,
        conventions='Same checkpoint within each rate. Existing per-rate BEST anchors;no re-selection. Metrics are equal rate means over nonempty cohorts;N and changes across rates are repeated exposures,not unique utterances. Gamma stats pool only active slots across rates;variance is population variance,not across seeds.')
    (output/'REPORT.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (output/'METRICS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    gamma_rows=[dict(group=g,slot=slot,**stats) for g,d in gamma.items() for slot,stats in {'all':d['all'],**d['by_source'],**d['by_slot']}.items()]
    with (output/'GAMMA.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(gamma_rows[0]));writer.writeheader();writer.writerows(gamma_rows)
    fmt=lambda v:'N/A' if v is None else f'{v:.6g}'
    lines=['# MOSEI seed66 Direct: frozen learned-gamma versus gamma=1','','INTERNAL DIAGNOSTIC ONLY','',
        'No training/new mask/weight update/threshold search. One causal Memory scan per batch.',
        'Each rate reuses its existing BEST checkpoint and exact saved mask;within-rate on/off weights and Local are identical.',
        'gamma=1 retains source reads;it is NOT source removal and NOT the separately trained original Flat.','',
        '| Scope | Learned ACC | Gamma1 ACC | Learned W-F1 | Gamma1 W-F1 | Delta W-F1(pp) | Corrections | Harms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ('mean','high'):
        v=aggregate[scope]['all'];lines.append('| '+scope+' | '+' | '.join(fmt(x) for x in
            (v['learned']['ACC'],v['neutral']['ACC'],v['learned']['WF1'],v['neutral']['WF1'],v['gain_WF1_pp'],v['corrections'],v['harms']))+' |')
    lines+=['','## Per-rate results','','| Rate | Learned W-F1 | Gamma1 W-F1 | Delta(pp) | Mean abs prediction change | Flips | Corrections | Harms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for rate,r in records.items():
        v=r['statistics']['groups']['all'];lines.append('| '+rate+' | '+' | '.join(fmt(x) for x in
            (v['learned']['WF1'],v['reset']['WF1'],v['gain_WF1_pp'],v['mean_abs_prediction_change'],v['sign_flip_N'],v['corrections'],v['harms']))+' |')
    lines+=['','## Text conditions (equal nonempty-rate means)','','| Group | Rates | N polarity(exposures) | Learned W-F1 | Gamma1 W-F1 | Delta(pp) | Corrections | Harms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in ('T_visible','T_missing_seen','T_missing_unseen'):
        v=aggregate['mean'][g];lines.append('| '+g+' | '+' | '.join(fmt(x) for x in
            (v['rates_with_polarity'],v['N_polarity'],v['learned']['WF1'],v['neutral']['WF1'],v['gain_WF1_pp'],v['corrections'],v['harms']))+' |')
    lines+=['','## Active gamma distributions (pooled rate exposures)','','| Slot | N | Mean | Variance | Min | Max | P05 | Median | P95 | Within0.01of1 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for slot,v in {'all':gamma['all']['all'],**gamma['all']['by_source'],**gamma['all']['by_slot']}.items():
        lines.append('| '+slot+' | '+' | '.join(fmt(v[k]) for k in ('N','mean','variance','min','max','p05','median','p95','within_0p01_of_one_fraction'))+' |')
    lines+=['','METRICS.csv includes all seven current availability patterns;GAMMA.csv/REPORT.json include per-source/per-role and Text-condition distributions.',
        'Groups are mutually exclusive:currentT visible;currentT missing and strict-pastT seen;currentT missing and strict-pastT unseen.',
        'Zero sentiment labels are excluded only from polarity metrics;MSE includes them. Corrections/harms use gamma1→learned,prediction>0.',
        'Counts across rates are repeated exposures;do not sum group W-F1 or interpret counts as independent unique utterances.',
        'Small effects cannot establish that Flat absorbed calibration,or identify a gradient/parameter-identifiability failure. No mechanism claim or independent-test performance claim.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
