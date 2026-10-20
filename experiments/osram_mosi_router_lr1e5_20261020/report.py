"""Complete-only, paired seed66 LR screen with exact selected prediction checks."""
import json
import math
from pathlib import Path

import numpy as np

from experiments.osram_mosi_gcnet_router_3seed_20261019.report import RATES, load_pair, report_outputs
from experiments.osram_mosi_gcnet_testbest_20261020.report import validate_policy


def write_report(output,queue):
    jobs=queue['jobs']
    if len(jobs)!=2 or {(j['seed'],j['mode']) for j in jobs}!={(66,'flat'),(66,'joint')} or any(j['status']!='complete' for j in jobs):
        raise ValueError('Require both completed seed66 runs')
    paths={j['mode']:j['output'] for j in jobs}
    pair,checks=load_pair(paths,dataset='CMUMOSI',seed=66)
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_text_path_router_joint_mosi_20261019.run import summarize_arrays
    rows=[]
    for mode,s in pair.items():
        validate_policy(s)
        if s['identity']['config']['learning_rate']!=1e-5 or s['identity']['backbone_learning_rates']!={'first_100':1e-5,'final_50':1e-5}:
            raise ValueError('Wrong backbone LR policy')
        history=json.loads((Path(paths[mode])/'history.json').read_text())
        for rate in RATES:
            scores=[epoch['test'][rate]['router']['WF1'] for epoch in history]
            if s['selected_epoch_by_rate'][rate]!=scores.index(max(scores))+1:
                raise ValueError('Not earliest per-rate TEST W-F1 best')
            artifact=Path(paths[mode])/'evaluation'/('random_'+rate.replace('.','p')+'.npz')
            with np.load(artifact,allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
            actual=summarize_arrays(arrays,trainer)['router']
            for m in ('ACC','WF1','MSE'):
                if not math.isclose(actual[m],s['evaluation'][rate]['router'][m],abs_tol=1e-8): raise ValueError('Selected metric mismatch')
            rows.append(dict(seed=66,mode=mode,rate=rate,selected_epoch=s['selected_epoch_by_rate'][rate],**actual))
    lines=['# MOSI backbone LR1e-5 seed66 paired screen','','INTERNAL TEST-ORACLE DIAGNOSTIC ONLY','',
        '150epochs from scratch. Backbone constant1e-5; Router3e-4 first100,3e-5 last50.',
        'Original GCNet missing; identical initialization/cohorts and training masks; no new loss or architecture.',
        'Per-rate Test-best W-F1; ACC uses same predictions. Eight-rate mean is not one deployment checkpoint.','',
        '| Model | ACC | W-F1 | High ACC | High W-F1 |','|---|---:|---:|---:|---:|']
    for mode,s in pair.items():
        values=[s['aggregate']['router'][scope][m] for scope in ('mean','high') for m in ('ACC','WF1')]
        lines.append('| '+mode+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines+=['','| Rate | Flat ACC | Router ACC | Flat W-F1 | Router W-F1 |','|---|---:|---:|---:|---:|']
    for rate in RATES:
        values=[pair[mode]['evaluation'][rate]['router'][m] for m in ('ACC','WF1') for mode in ('flat','joint')]
        lines.append('| '+rate+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    report_outputs(output,dict(runs=pair,checks=checks),rows,lines)
