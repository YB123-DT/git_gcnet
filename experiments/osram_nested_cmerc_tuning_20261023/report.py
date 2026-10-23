"""Report the entire finite screen; never select incomplete trials or old scores."""
import csv
import hashlib
import json
import math
from pathlib import Path

from .worker import DATASETS, METHODS, RATES


def metric_deltas(candidate, baseline):
    return {name: value - baseline[name] for name, value in candidate.items() if name in baseline}


def write_report(output, state):
    jobs = state['jobs']
    if (len(jobs) != 21 or any(j['status'] != 'complete' for j in jobs)
            or {(j['dataset'], j['method']) for j in jobs} != {(d, m) for d in DATASETS for m in METHODS}):
        raise ValueError('Require all21 approved jobs; no partial ranking')
    from .run import validate_summary
    from experiments.osram_iemocap6_gcnet_testbest_20261020.worker import validate_selected
    runs, rows, compact = {}, [], []
    for job in jobs:
        path = Path(job['output'])
        summary = validate_summary(job)
        validate_selected(path, summary['best_by_rate'])
        history = json.loads((path / 'history.json').read_text())
        if len(history) != job['epochs']:
            raise ValueError('History does not cover full budget')
        for scope, rates in (('mean', RATES), ('high', RATES[5:])):
            for metric, reported in summary['aggregate']['router'][scope].items():
                actual = sum(summary['evaluation'][r]['router'][metric] for r in rates) / len(rates)
                if not math.isfinite(actual) or not math.isclose(actual, reported, abs_tol=1e-9):
                    raise ValueError('Aggregate does not match selected predictions')
        runs[job['dataset'], job['method']] = (summary, history)
        for rate in RATES:
            score = summary['evaluation'][rate]['router']
            best = summary['best_by_rate'][rate]
            rows.append(dict(dataset=job['dataset'], fold=job['fold'], seed=66, method=job['method'],
                rate=rate, selected_epoch=best['epoch'], **{k: score.get(k, '') for k in ('ACC', 'UA', 'WF1', 'MSE', 'CE')}))
        compact.append(dict(dataset=job['dataset'], method=job['method'], fold=job['fold'], seed=66,
            path=str(path), summary_sha256=hashlib.sha256((path / 'SUMMARY.json').read_bytes()).hexdigest(),
            config=summary['config'], aggregate=summary['aggregate']['router'],
            total_parameters=summary['total_parameters'], nested_parameters=summary['nested_parameters'],
            seconds=summary['seconds'], peak_memory_mib=summary['peak_memory_mib'],
            selected=summary['best_by_rate']))
    lines = ['# Existing Nested cmerc finite screen', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        'Seed66 only. IEMOCAP4/6 cover fold5 ONLY, not fivefold means.',
        'Original tasks, CHM/Flat1600/LocalSkip and gcnet-random-mask-v1 unchanged.',
        'Evaluation replays fixed accepted masks. Historical IEMOCAP resampled per-epoch test masks;',
        'only the newly trained same-source Flat is the paired score comparator.',
        'All arms use the existing Torch1.13 UUID enumeration/RNG repair; historical unrepaired MOSEI is reference only.',
        'No old trained weights loaded; no automatic extra seeds/HPO. Metric key router denotes plain task predictions, not a router.', '']
    ranking = {}
    for dataset in DATASETS:
        baseline, baseline_history = runs[dataset, 'none']
        ranking[dataset] = []
        lines += [f'## {dataset}' + (' fold5' if dataset != 'CMUMOSEI' else ''), '',
            '| Method | Params | ACC | ΔACC pp | UA | ΔUA pp | W-F1 | ΔWF1 pp | High ACC | High UA | High WF1 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for method in METHODS:
            s, history = runs[dataset, method]
            for name in ('initial_backbone_sha256',):
                if s[name] != baseline[name]:
                    raise ValueError('Original initialization differs between paired arms')
            if s['identity']['mask_reference'] != baseline['identity']['mask_reference']:
                raise ValueError('Pair did not replay the same accepted masks')
            if s['identity']['split'] != baseline['identity']['split']:
                raise ValueError('Pair used different training/test splits')
            for old, new in zip(baseline_history, history):
                for name in ('epoch', 'mask_trace_sha256', 'optimizer_steps', 'learning_rates'):
                    if old[name] != new[name]:
                        raise ValueError('Pair training protocol differs: ' + name)
            mean, high = s['aggregate']['router']['mean'], s['aggregate']['router']['high']
            delta = metric_deltas(mean, baseline['aggregate']['router']['mean'])
            cell = lambda x: f'{x:.3f}' if x is not None else 'N/A'
            values = [cell(mean['ACC']), cell(delta['ACC']), cell(mean.get('UA')), cell(delta.get('UA')),
                cell(mean['WF1']), cell(delta['WF1']), cell(high['ACC']), cell(high.get('UA')), cell(high['WF1'])]
            lines.append('| ' + method + f" | {s['total_parameters']} | " + ' | '.join(values) + ' |')
            if method != 'none':
                ranking[dataset].append(dict(method=method, delta=delta,
                    primary=mean['WF1'] if dataset == 'CMUMOSEI' else mean['ACC']))
        ranking[dataset].sort(key=lambda x: x['primary'], reverse=True)
        lines += ['', 'All per-rate predictions and selected epochs are in PER_RATE.csv.',
            'Best primary-metric candidate (descriptive, no automatic promotion): ' + ranking[dataset][0]['method'], '']
    output = Path(output)
    payload = dict(status='complete', source_commit=state['source_commit'], label='INTERNAL TEST-ORACLE SCREEN',
        scope='MOSEI seed66 and IEMOCAP4/6 seed66 fold5 only', runs=compact, ranking=ranking)
    (output / 'RESULT.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    with (output / 'PER_RATE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
