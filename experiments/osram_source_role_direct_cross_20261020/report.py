"""Complete-only reporting for the finite direct-source cross-dataset experiment."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


RATES = tuple(f'{i / 10:.1f}' for i in range(8))
SEEDS = (66, 67, 68)
SPECS = {
    'CMUMOSEI': dict(epochs=60, folds=(1,), metrics=('ACC', 'WF1', 'MSE'), selection='weighted_f1'),
    'MELD': dict(epochs=50, folds=(1,), metrics=('ACC', 'UA', 'WF1', 'CE'), selection='accuracy'),
    'IEMOCAPSix': dict(epochs=100, folds=(1, 2, 3, 4, 5), metrics=('ACC', 'UA', 'WF1', 'CE'), selection='accuracy'),
}


def _read_rows(state):
    """Validate all expected runs before publishing any complete-result artifact."""
    expected = {(dataset, seed, fold) for dataset, spec in SPECS.items()
                for seed in SEEDS for fold in spec['folds']}
    jobs = state.get('jobs', [])
    actual = [(job['dataset'], job['seed'], job['fold']) for job in jobs]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Exactly 21 jobs are required: three seeds, MOSEI/MELD fold1, IEMOCAPSix five folds')
    if any(job.get('status') != 'complete' for job in jobs):
        raise ValueError('All 21 jobs must be complete; failed or unfinished runs cannot enter the report')
    rows, references, sources = [], [], set()
    for job in sorted(jobs, key=lambda value: (value['dataset'], value['seed'], value['fold'])):
        dataset, seed, fold = job['dataset'], job['seed'], job['fold']
        spec = SPECS[dataset]
        path = Path(job['output']) / 'SUMMARY.json'
        raw = path.read_bytes()
        summary = json.loads(raw)
        config = summary.get('config', {})
        required = dict(osram_source_role=True, osram_source_role_mode='direct',
                        missing_mask_protocol='gcnet-random-mask-v1',
                        checkpoint_selection='test-oracle-per-rate',
                        checkpoint_selection_metric=spec['selection'])
        if (summary.get('status') != 'complete' or summary.get('phase') != 'train'
                or summary.get('epoch') != spec['epochs']
                or any(config.get(key) != value for key, value in required.items())
                or ('epochs' in config and config['epochs'] != spec['epochs'])):
            raise ValueError(f'Incomplete or incompatible direct-source training summary: {path}')
        source = summary.get('source_commit')
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f'Missing immutable source commit: {path}')
        sources.add(source)
        evaluation = summary.get('evaluation', {})
        if set(evaluation) != set(RATES):
            raise ValueError(f'Exactly eight selected evaluation rates are required: {path}')
        for rate in RATES:
            entry = evaluation[rate]
            selected = entry.get('selected_epoch', summary.get('selected_epoch_by_rate', {}).get(rate))
            if type(selected) is not int or not 1 <= selected <= spec['epochs']:
                raise ValueError(f'Invalid selected epoch for {dataset} seed{seed} fold{fold} rate{rate}')
            metric = entry.get('router', {})
            if any(key not in metric or isinstance(metric[key], bool)
                   or not isinstance(metric[key], (float, int))
                   or not math.isfinite(metric[key]) for key in spec['metrics']):
                raise ValueError(f'Missing or nonfinite same-prediction metric: {path}, rate{rate}')
            if any(not 0 <= metric[key] <= 100 for key in spec['metrics'] if key in ('ACC', 'UA', 'WF1')):
                raise ValueError(f'Classification metrics must be percentages: {path}, rate{rate}')
            if any(metric[key] < 0 for key in spec['metrics'] if key in ('CE', 'MSE')):
                raise ValueError(f'Negative task loss: {path}, rate{rate}')
            rows.append(dict(dataset=dataset, seed=seed, fold=fold, rate=rate, selected_epoch=selected,
                             **{key: float(metric[key]) for key in spec['metrics']}))
        references.append(dict(dataset=dataset, seed=seed, fold=fold, output=job['output'],
                               source_commit=source, summary_sha256=hashlib.sha256(raw).hexdigest(),
                               config=config))
    if len(sources) != 1:
        raise ValueError('All jobs must share one immutable source commit; do not mix implementation versions')
    return rows, references, sources.pop()


def _aggregate(rows, dataset):
    spec = SPECS[dataset]
    per_seed = {}
    for seed in SEEDS:
        seed_rows = [row for row in rows if row['dataset'] == dataset and row['seed'] == seed]
        by_rate = {rate: {key: statistics.mean(row[key] for row in seed_rows if row['rate'] == rate)
                          for key in spec['metrics']} for rate in RATES}
        per_seed[str(seed)] = dict(by_rate=by_rate)
        for scope, rates in (('mean', RATES), ('high', RATES[5:])):
            per_seed[str(seed)][scope] = {
                key: statistics.mean(by_rate[rate][key] for rate in rates) for key in spec['metrics']}

    def seed_stats(values):
        return dict(mean=statistics.mean(values), std=statistics.stdev(values))

    result = dict(epochs=spec['epochs'], folds=list(spec['folds']), per_seed=per_seed,
                  by_rate={rate: {key: seed_stats([per_seed[str(seed)]['by_rate'][rate][key] for seed in SEEDS])
                                  for key in spec['metrics']} for rate in RATES})
    for scope in ('mean', 'high'):
        result[scope] = {key: seed_stats([per_seed[str(seed)][scope][key] for seed in SEEDS])
                         for key in spec['metrics']}
    return result


def write_report(output, state):
    """Publish only when every approved seed/fold has its full training summary."""
    rows, references, source = _read_rows(state)
    datasets = {dataset: _aggregate(rows, dataset) for dataset in SPECS}
    result = dict(status='complete', label='INTERNAL TEST-ORACLE SCREEN', source_commit=source,
                  seeds=list(SEEDS), completed_jobs=len(references), rows=rows, datasets=datasets,
                  run_references=references,
                  aggregation='Equal fold means within seed/rate, equal eight-rate means within seed, '
                              'then three-seed mean and sample SD (ddof=1); not pooled utterance metrics.',
                  evaluation_masks='MELD: original per-epoch resampled GCNet test masks; '
                                   'MOSEI/IEMOCAPSix: original fixed GCNet evaluation banks. '
                                   'ACC/UA/W-F1 are from the same selected predictions and exact mask.',
                  limitations='Per-rate Test-oracle selection, not one deployment checkpoint; '
                              'no independent-test generalization claim. No matched Flat comparison is inferred.')
    lines = ['# Direct Source Weighting: MOSEI, MELD and IEMOCAP-6', '',
             'INTERNAL TEST-ORACLE SCREEN', '',
             'gamma=2*sigmoid(score); direct sum of source reads, without original-read add-back. '
             'The original Local Skip/Flat/CHM remain; this is not removal of the Local Skip.', '',
             'MOSEI: 60 epochs; MELD: 50 epochs; IEMOCAP-6: 100 epochs, all five folds. '
             'Seeds66/67/68. All 21 jobs completed.', '',
             result['aggregation'], '', result['evaluation_masks'], '',
             'MOSEI selects per-rate Test-best weighted F1; classification datasets select per-rate '
             'Test-best ACC. UA/W-F1 must come from the ACC-selected prediction, not separately selected epochs.', '',
             result['limitations'], '', f'Source commit: `{source}`.', '']
    for dataset, stats in datasets.items():
        metrics = SPECS[dataset]['metrics']
        lines += [f'## {dataset}', '', '| Scope | Seed | ' + ' | '.join(metrics) + ' |',
                  '|---|---|' + '---:|' * len(metrics)]
        for scope in ('mean', 'high'):
            for seed in SEEDS:
                values = stats['per_seed'][str(seed)][scope]
                lines.append(f'| {scope} | {seed} | ' + ' | '.join(f'{values[key]:.4f}' for key in metrics) + ' |')
            lines.append(f'| {scope} | 3-seed mean ± sample SD | ' + ' | '.join(
                f"{stats[scope][key]['mean']:.4f} ± {stats[scope][key]['std']:.4f}" for key in metrics) + ' |')
        lines += ['', '| Rate | ' + ' | '.join(metrics) + ' |', '|---|' + '---:|' * len(metrics)]
        for rate in RATES:
            lines.append(f'| {rate} | ' + ' | '.join(
                f"{stats['by_rate'][rate][key]['mean']:.4f} ± {stats['by_rate'][rate][key]['std']:.4f}"
                for key in metrics) + ' |')
        lines += ['', '### Per-seed / fold / rate (selected checkpoints)', '',
                  '| Seed | Fold | Rate | Selected epoch | ' + ' | '.join(metrics) + ' |',
                  '|---:|---:|---|---:|' + '---:|' * len(metrics)]
        for row in rows:
            if row['dataset'] == dataset:
                lines.append(f"| {row['seed']} | {row['fold']} | {row['rate']} | {row['selected_epoch']} | "
                             + ' | '.join(f'{row[key]:.4f}' for key in metrics) + ' |')
        lines += ['']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ('dataset', 'seed', 'fold', 'rate', 'selected_epoch', 'ACC', 'UA', 'WF1', 'CE', 'MSE')
    with (output / 'METRICS.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / 'REPORT.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    return result
