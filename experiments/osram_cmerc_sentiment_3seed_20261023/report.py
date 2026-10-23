"""Complete-only three-seed reports from saved predictions, without new inference."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.osram_mosi_gcnet_router_3seed_20261019.report import RATES, write
from .run import REFERENCE, SOURCE


def load_run(path, dataset, seed):
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
    path = Path(path)
    summary = json.loads((path / 'SUMMARY.json').read_text())
    epochs = 150 if dataset == 'CMUMOSI' else 60
    expected = dict(status='complete', mode='flat', dataset=dataset, seed=seed, epoch=epochs)
    reused = path.resolve() == Path(REFERENCE).resolve() and dataset == 'CMUMOSI' and seed == 66
    if summary.get('phase') != 'train' and not (reused and 'phase' not in summary):
        raise ValueError('Run phase differs: ' + str(path))
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError('Incomplete or mismatched run: ' + str(path))
    cfg = summary['identity']['config']
    if cfg['epochs'] != epochs or cfg['seed'] != seed or cfg['batch_size'] != 32:
        raise ValueError('Run configuration differs')
    if (cfg.get('weight_decay') != 1e-5
            or cfg.get('learning_rate') != (3e-4 if dataset == 'CMUMOSI' else 1e-4)
            or cfg.get('missing_mask_protocol') != 'gcnet-random-mask-v1'
            or cfg.get('checkpoint_selection') != 'test-oracle-per-rate'
            or cfg.get('checkpoint_selection_metric') != 'weighted_f1'):
        raise ValueError('Run protocol differs')
    if len(json.loads((path / 'history.json').read_text())) != epochs:
        raise ValueError('Training history does not cover the approved budget')
    if set(summary['evaluation']) != set(RATES):
        raise ValueError('All eight rates required')
    validate_selected(path, summary['best_by_rate'])
    for rate in RATES:
        row = summary['evaluation'][rate]
        artifact = path / 'evaluation' / ('random_' + rate.replace('.', 'p') + '.npz')
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        best = summary['best_by_rate'][rate]
        if digest != row['output_sha256'] or digest != best['prediction_sha256']:
            raise ValueError('Published prediction integrity failure')
        if row['router'] != best['metrics']['router'] or row['selected_epoch'] != best['epoch']:
            raise ValueError('Published metrics differ from selected ledger')
        if not all(math.isfinite(row['router'][metric]) for metric in ('ACC', 'WF1', 'MSE')):
            raise ValueError('Nonfinite metrics')
    for scope, rates in (('mean', RATES), ('high', RATES[5:])):
        for metric in ('ACC', 'WF1', 'MSE'):
            mean = statistics.mean(summary['evaluation'][rate]['router'][metric] for rate in rates)
            if not math.isclose(summary['aggregate']['router'][scope][metric], mean, abs_tol=1e-9):
                raise ValueError('Aggregate differs from saved per-rate metrics')
    return summary


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values), n=3)


def write_report(output, queue):
    if len(queue['jobs']) != 5 or any(j['status'] != 'complete' for j in queue['jobs']):
        raise ValueError('Require exactly five completed new jobs')
    expected = {('CMUMOSI', 67), ('CMUMOSI', 68)} | {('CMUMOSEI', s) for s in (66, 67, 68)}
    if {(j['dataset'], j['seed']) for j in queue['jobs']} != expected:
        raise ValueError('Five-job identities differ')
    paths = {(j['dataset'], j['seed']): j['output'] for j in queue['jobs']}
    paths['CMUMOSI', 66] = REFERENCE
    runs, aggregate, per_rate, rows = {}, {}, {}, []
    lines = ['# CMERC sentiment three-seed results', '', 'INTERNAL TEST-ORACLE SCREEN', '',
             'Original Flat / newGCNet missing. Per-rate test W-F1 selection; different rates may select different epochs.',
             'MOSI seed66 reuses the complete execution_v2 control. No new inference.', '',
             'Every score below is the Flat prediction stored under the legacy `router` metric key; no router model is trained.',
             '`mean` averages rates0.0–0.7; `high` averages rates0.5–0.7. Sample SD is across the three seeds.', '',
             '| Dataset | Scope | Metric | Mean | Sample SD |', '|---|---|---|---:|---:|']
    for dataset in ('CMUMOSI', 'CMUMOSEI'):
        runs[dataset] = {str(seed): load_run(paths[dataset, seed], dataset, seed) for seed in (66, 67, 68)}
        for seed, run in runs[dataset].items():
            if (dataset, int(seed)) != ('CMUMOSI', 66) and run['source_commit'] != queue['source_commit']:
                raise ValueError('Numerical source commit differs')
        aggregate[dataset] = {}
        for scope in ('mean', 'high'):
            aggregate[dataset][scope] = {}
            for metric in ('ACC', 'WF1', 'MSE'):
                value = stats([run['aggregate']['router'][scope][metric] for run in runs[dataset].values()])
                aggregate[dataset][scope][metric] = value
                lines.append(f'| {dataset} | {scope} | {metric} | {value["mean"]:.6f} | {value["std"]:.6f} |')
        per_rate[dataset] = {}
        for rate in RATES:
            per_rate[dataset][rate] = {metric: stats([run['evaluation'][rate]['router'][metric]
                for run in runs[dataset].values()]) for metric in ('ACC', 'WF1', 'MSE')}
            for seed, run in runs[dataset].items():
                row = run['evaluation'][rate]
                rows.append(dict(dataset=dataset, seed=seed, rate=rate, selected_epoch=row['selected_epoch'],
                                 **{metric: row['router'][metric] for metric in ('ACC', 'WF1', 'MSE')}))
        lines += ['', f'## {dataset} per-rate mean ± sample SD', '',
                  '| Rate | ACC | W-F1 | MSE |', '|---|---:|---:|---:|']
        for rate, values in per_rate[dataset].items():
            lines.append('| ' + rate + ' | ' + ' | '.join(f'{values[m]["mean"]:.6f} ± {values[m]["std"]:.6f}'
                for m in ('ACC', 'WF1', 'MSE')) + ' |')
        lines += ['', f'## {dataset} seed-level scores', '',
                  '| Seed | Eight-rate ACC | Eight-rate W-F1 | Eight-rate MSE | High ACC | High W-F1 | High MSE |',
                  '|---:|---:|---:|---:|---:|---:|---:|']
        for seed, run in runs[dataset].items():
            values = [run['aggregate']['router'][scope][metric] for scope in ('mean', 'high')
                      for metric in ('ACC', 'WF1', 'MSE')]
            lines.append('| ' + seed + ' | ' + ' | '.join(f'{value:.6f}' for value in values) + ' |')
    output = Path(output)
    compact = {}
    for dataset, seeds in runs.items():
        compact[dataset] = {}
        for seed, run in seeds.items():
            path = Path(paths[dataset, int(seed)])
            compact[dataset][seed] = dict(path=str(path),
                summary_sha256=hashlib.sha256((path / 'SUMMARY.json').read_bytes()).hexdigest(),
                source_commit=run.get('source_commit'), environment=run.get('environment'),
                config=run['identity']['config'], aggregate=run['aggregate']['router'],
                selected={rate: {key: row[key] for key in
                    ('epoch', 'checkpoint_sha256', 'prediction_sha256')}
                    for rate, row in run['best_by_rate'].items()})
    write(output / 'RESULT.json', dict(status='complete', label='INTERNAL TEST-ORACLE SCREEN',
        source_commit=queue['source_commit'], numerical_commit=SOURCE, reused_reference=REFERENCE,
        runs=compact, aggregate=aggregate, per_rate=per_rate))
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    with (output / 'PER_RATE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
