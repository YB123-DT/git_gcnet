"""Completed-only per-seed/per-rate V3 report, reusing existing seed66 runs."""
import csv
import statistics
from pathlib import Path

from experiments.osram_v3_classification_20261021.run import read
from experiments.osram_text_path_router_3seed_20261019.resource_gate import write

RATES = tuple(f'{i/10:.1f}' for i in range(8))


def summarize(root, queue, original_classification, original_scalar):
    jobs = list(original_classification['jobs']) + list(queue['jobs'])
    jobs += [dict(dataset='CMUMOSEI', fold=1, seed=66, output=j['output'])
        for j in original_scalar['jobs'] if j['dataset'] == 'CMUMOSEI']
    states, rows, complete = [], [], {}
    for job in jobs:
        summary = read(Path(job['output'])/'SUMMARY.json')
        dataset, fold, seed = job['dataset'], job.get('fold',1), job.get('seed',66)
        states.append(dict(dataset=dataset, fold=fold, seed=seed, output=job['output'],
            status=summary.get('status',job.get('status','pending')), epoch=summary.get('epoch')))
        if summary.get('status') != 'complete' or summary.get('phase') != 'train':
            continue
        cfg = summary.get('config', summary.get('identity',{}).get('config',{}))
        expected_epochs = {'MELD':50, 'CMUMOSEI':60, 'IEMOCAPFour':100, 'IEMOCAPSix':100}[dataset]
        if (summary.get('epoch') != expected_epochs or not cfg.get('osram_source_role_v3')
                or cfg.get('seed') != seed or cfg.get('dataset') != dataset):
            raise ValueError('Invalid completed V3 budget/config: '+job['output'])
        evaluation = summary.get('evaluation',{})
        if set(evaluation) != set(RATES):
            raise ValueError('Completed run missing all8 selected rates')
        complete[(dataset,fold,seed)] = evaluation
        for rate in RATES:
            metrics = evaluation[rate]['router']
            rows.append(dict(dataset=dataset, fold=fold, seed=seed, rate=rate,
                epoch=summary['best_by_rate'][rate]['epoch'], **{k: metrics[k] for k in
                    ('ACC','WF1','MSE') if k in metrics}, **{k: metrics[k] for k in ('UA','CE') if k in metrics}))
    per_seed, three_seed = [], []
    for dataset in ('MELD','IEMOCAPFour','IEMOCAPSix','CMUMOSEI'):
        folds = tuple(range(1,6)) if dataset.startswith('IEMOCAP') else (1,)
        metrics = ('ACC','WF1','MSE') if dataset == 'CMUMOSEI' else ('ACC','UA','WF1','CE')
        for seed in (66,67,68):
            if not all((dataset,f,seed) in complete for f in folds):
                continue
            for scope, rates in (('mean',RATES), ('high',RATES[5:])):
                per_seed.append(dict(dataset=dataset, seed=seed, scope=scope,
                    **{m:statistics.mean(complete[(dataset,f,seed)][r]['router'][m] for f in folds for r in rates)
                       for m in metrics}))
        for scope in ('mean','high'):
            selected = [r for r in per_seed if r['dataset']==dataset and r['scope']==scope]
            if len(selected) == 3:
                three_seed.append(dict(dataset=dataset, scope=scope, n=3,
                    **{m+suffix:fn(r[m] for r in selected) for m in metrics for suffix,fn in
                        (('_mean',statistics.mean),('_sample_std',statistics.stdev))}))
    result = dict(label='INTERNAL TEST-ORACLE SCREEN', states=states, per_rate=rows,
        per_seed=per_seed, three_seed=three_seed, fold_aggregation='unweighted-fold-metric-mean',
        queue_status=queue['status'], seed66_reused=True)
    write(root/'REPORT.json',result)
    if rows:
        columns = ['dataset','fold','seed','rate','epoch','ACC','UA','WF1','CE','MSE']
        with (root/'PER_RATE.csv').open('w') as stream:
            writer=csv.DictWriter(stream, fieldnames=columns);writer.writeheader();writer.writerows(rows)
    lines=['# V3 three-seed completion', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        '| Dataset | fold | seed | status | epoch |', '|---|---:|---:|---|---:|']
    for row in states:
        lines.append(f"| {row['dataset']} | {row['fold']} | {row['seed']} | {row['status']} | {row['epoch'] or '—'} |")
    lines += ['', 'Completed-only summaries:', '',
        '| Dataset | seed | scope | ACC | UA | W-F1 |', '|---|---:|---|---:|---:|---:|']
    for row in per_seed:
        ua=f"{row['UA']:.3f}" if 'UA' in row else 'N/A'
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['scope']} | {row['ACC']:.3f} | {ua} | {row['WF1']:.3f} |")
    lines += ['', 'Missing/unfinished runs never contribute to three-seed means.',
        'Per-rate Test-best is exploratory and is not a single deployed checkpoint.',
        'Seed66 is reused; the24 new runs are seeds67/68 only.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result
