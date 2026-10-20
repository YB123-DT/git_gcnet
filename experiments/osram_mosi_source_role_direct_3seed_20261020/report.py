"""Complete-only three-seed report, reusing existing seed66 without retraining."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from experiments.osram_coupled_source_role_20261020.report import write_report as per_seed_report
from experiments.osram_coupled_source_role_20261020.worker import RATES
from .run import UPSTREAM


def summarize_rows(rows):
    keys=[(r['seed'],r['rate']) for r in rows]
    if len(keys)!=24 or set(keys)!={(s,r) for s in (66,67,68) for r in RATES}:
        raise ValueError('Require exactly24 complete seed/rate rows')
    metrics=('ACC','WF1','flat_ACC','flat_WF1','delta_ACC_pp','delta_WF1_pp')
    values=[dict(r,delta_ACC_pp=r['ACC']-r['flat_ACC'],delta_WF1_pp=r['WF1']-r['flat_WF1']) for r in rows]
    def stats(group):
        return {k:dict(mean=statistics.mean(r[k] for r in group),std=statistics.stdev(r[k] for r in group)) for k in metrics}
    result=dict(by_rate={rate:stats([r for r in values if r['rate']==rate]) for rate in RATES})
    for name,rates in (('mean',RATES),('high',RATES[5:])):
        per_seed=[{k:statistics.mean(r[k] for r in values if r['seed']==seed and r['rate'] in rates) for k in metrics} for seed in (66,67,68)]
        result[name]=stats(per_seed)
    return result


def write_report(output,queue):
    from experiments.osram_mosi_gcnet_router_3seed_20261019.run import dependency_state
    if dependency_state(UPSTREAM)!='ready':raise ValueError('Seed66 is not complete')
    previous=json.loads(UPSTREAM.read_text())
    jobs=previous['jobs']+queue['jobs']
    if len(jobs)!=3 or sorted(j['seed'] for j in jobs)!=[66,67,68] or any(j['status']!='complete' for j in jobs):
        raise ValueError('Exactly three complete seeds required')
    output=Path(output);rows=[];runs={};reference=None
    for job in sorted(jobs,key=lambda j:j['seed']):
        per=output/'per_seed'/f"seed_{job['seed']}";per.mkdir(parents=True,exist_ok=True)
        per_seed_report(per,dict(jobs=[job]),source_role_mode='direct',method_label='Direct Source Weighting')
        s=json.loads((per/'SUMMARY.json').read_text())['run']
        numerical={k:v for k,v in s['identity']['source_hashes'].items() if k.startswith(('gcnet_missing_m3/','gcnet_modality_jepa/'))}
        cfg=dict(s['config']);cfg.pop('seed')
        current=(numerical,cfg,s['identity']['split'],s['identity']['mask_hashes'])
        if reference is not None and reference!=current:raise ValueError('Three-seed numerical code/config/split/masks differ')
        reference=current
        runs[str(job['seed'])]=dict(output=job['output'],source_commit=s['source_commit'],
            summary_sha256=hashlib.sha256((Path(job['output'])/'SUMMARY.json').read_bytes()).hexdigest(),
            aggregate=s['aggregate']['router'])
        with (per/'METRICS.csv').open() as f:
            for row in csv.DictReader(f):
                rows.append(dict(seed=int(row['seed']),rate=row['rate'],selected_epoch=int(row['selected_epoch']),
                    **{k:float(row[k]) for k in ('ACC','WF1','flat_ACC','flat_WF1')}))
    stats=summarize_rows(rows)
    (output/'SUMMARY.json').write_text(json.dumps(dict(status='complete',label='INTERNAL TEST-ORACLE SCREEN',
        seeds=[66,67,68],epochs=150,run_references=runs,statistics=stats,
        numerical_code_config_split_mask_banks_exact=True,std_definition='sample standard deviation across seeds, ddof1'),indent=2,sort_keys=True)+'\n')
    with (output/'METRICS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# MOSI Direct Source Weighting: three seeds','','INTERNAL TEST-ORACLE SCREEN','',
        'Seed66 is reused, not retrained. Seeds67/68 use the unchanged algorithm, original s0 runtime,GCNet missing and150epochs.',
        'Per-rate earliest Test-best W-F1; matching same-prediction ACC. Not one deployment checkpoint.','',
        '| Rate | Seed66 W-F1 | Seed67 W-F1 | Seed68 W-F1 | Mean ± sample std | Flat mean | Delta pp |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for rate in RATES:
        v=stats['by_rate'][rate];scores=[next(r['WF1'] for r in rows if r['seed']==seed and r['rate']==rate) for seed in (66,67,68)]
        lines.append('| '+rate+' | '+' | '.join(f'{s:.3f}' for s in scores)+f" | {v['WF1']['mean']:.3f} ± {v['WF1']['std']:.3f} | {v['flat_WF1']['mean']:.3f} | {v['delta_WF1_pp']['mean']:+.3f} |")
    lines+=['','| Scope | Flat W-F1 | Direct W-F1 | Delta pp | Flat ACC | Direct ACC |','|---|---:|---:|---:|---:|---:|']
    for scope in ('mean','high'):
        v=stats[scope];lines.append(f"| {scope} | {v['flat_WF1']['mean']:.3f} ± {v['flat_WF1']['std']:.3f} | {v['WF1']['mean']:.3f} ± {v['WF1']['std']:.3f} | {v['delta_WF1_pp']['mean']:+.3f} | {v['flat_ACC']['mean']:.3f} | {v['ACC']['mean']:.3f} |")
    lines+=['','High means.5/.6/.7 averaged within each seed before computing across-seed statistics.',
        'All selected prediction artifacts were recomputed and matched to same-seed Flat cohorts/masks.',
        'Three seeds are a finite exploratory repeat, not a new hyperparameter search or independent-test claim.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
