"""Five-fold three-seed completion, explicitly reusing six existing fold5 jobs."""
from pathlib import Path
import statistics

from experiments.osram_mosi_gcnet_router_3seed_20261019.report import (
    CORE, RATES, load_pair, report_outputs, stats,
)


def write_report(output, queue):
    output = Path(output)
    if len(queue['jobs']) != 24 or any(j['status'] != 'complete' for j in queue['jobs']):
        raise ValueError('Require exactly 24 completed new fold1..4 jobs')
    refs = queue.get('references', [])
    if len(refs) != 6 or any(j.get('fold') != 5 or not j.get('reused') for j in refs):
        raise ValueError('Require six explicitly reused completed fold5 results')
    plans = queue['jobs'] + refs
    if {(j['fold'], j['seed'], j['mode']) for j in plans} != {
            (f, s, m) for f in range(1, 6) for s in (66, 67, 68) for m in ('flat', 'joint')}:
        raise ValueError('Five-fold/three-seed plan has omissions or duplicates')
    runs, checks, rows = {}, {}, []
    core = None
    lines = ['# IEMOCAP-6: five-fold / three-seed Flat and one-stage Router', '',
             'INTERNAL EXPLORATORY EXPERIMENT', '',
             '24 new fold1..4 runs; six completed fold5 runs reused, never retrained.',
             '100 epochs per fold. GCNet random-mask-v1. One validation-selected checkpoint per run.',
             'Fold-specific validation/test masks are paired across models and fixed across seeds.',
             'Reported five-fold scores are unweighted fold-metric means (not pooled utterance metrics).', '',
             '| Fold | Seed | Model | Selected epoch | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |',
             '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for fold in range(1, 6):
        runs[str(fold)], checks[str(fold)] = {}, {}
        reference_masks = None
        for seed in (66, 67, 68):
            selected = [j for j in plans if j['fold'] == fold and j['seed'] == seed]
            if len(selected) != 2 or {j['mode'] for j in selected} != {'flat', 'joint'}:
                raise ValueError('Invalid fold seed pair')
            pair, check = load_pair({j['mode']: j['output'] for j in selected},
                                    dataset='IEMOCAPSix', seed=seed, fold=fold, epochs=100)
            new_core = {name: pair['flat']['identity']['source_hashes'][name] for name in CORE}
            if core is not None and core != new_core:
                raise ValueError('Numerical core changed across IEMOCAP folds/seeds')
            core = new_core
            identity = pair['flat']['identity']
            mask = dict(validation=identity['validation_mask_hashes'], test=identity['test_masks'], split=identity['split'])
            if reference_masks is not None and mask != reference_masks:
                raise ValueError('Same-fold evaluation masks/split changed across seeds')
            reference_masks = mask
            runs[str(fold)][str(seed)], checks[str(fold)][str(seed)] = pair, check
            for mode, summary in pair.items():
                values = [summary['aggregate']['router'][scope][metric] for scope in ('mean', 'high')
                          for metric in ('ACC', 'UA', 'WF1')]
                lines.append(f'| {fold} | {seed} | {mode} | {summary["selected_epoch"]} | ' +
                             ' | '.join(f'{v:.4f}' for v in values) + ' |')
                for rate in RATES:
                    rows.append(dict(dataset='IEMOCAPSix', fold=fold, seed=seed, mode=mode, rate=rate,
                                     reused_fold5=fold == 5, selected_epoch=summary['selected_epoch'],
                                     **summary['evaluation'][rate]['router']))
    # Unit of variation is seed. Average each seed's five fold scores first.
    per_seed = {str(seed): {mode: {scope: {metric: statistics.mean([
        runs[str(fold)][str(seed)][mode]['aggregate']['router'][scope][metric] for fold in range(1, 6)])
        for metric in ('ACC', 'UA', 'WF1', 'CE')} for scope in ('mean', 'high')}
        for mode in ('flat', 'joint')} for seed in (66, 67, 68)}
    aggregate = {mode: {scope: {metric: stats([per_seed[str(seed)][mode][scope][metric]
                                            for seed in (66, 67, 68)])
        for metric in ('ACC', 'UA', 'WF1', 'CE')} for scope in ('mean', 'high')}
        for mode in ('flat', 'joint')}
    per_rate = {rate: {mode: {metric: stats([statistics.mean([
        runs[str(fold)][str(seed)][mode]['evaluation'][rate]['router'][metric] for fold in range(1, 6)])
        for seed in (66, 67, 68)]) for metric in ('ACC', 'UA', 'WF1', 'CE')}
        for mode in ('flat', 'joint')} for rate in RATES}
    lines += ['', '## Per-seed five-fold means', '',
              '| Seed | Model | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    for seed, models in per_seed.items():
        for mode, scopes in models.items():
            values = [scopes[scope][metric] for scope in ('mean', 'high') for metric in ('ACC', 'UA', 'WF1')]
            lines.append(f'| {seed} | {mode} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', '## Three-seed mean ± sample standard deviation of five-fold means', '',
              '| Model | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode, scopes in aggregate.items():
        values = [f'{scopes[scope][metric]["mean"]:.4f} ± {scopes[scope][metric]["std"]:.4f}'
                  for scope in ('mean', 'high') for metric in ('ACC', 'UA', 'WF1')]
        lines.append('| ' + mode + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## Per-rate three-seed means of five-fold metrics', '',
              '| Rate | Flat ACC | Joint ACC | Flat UA | Joint UA | Flat W-F1 | Joint W-F1 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for rate, models in per_rate.items():
        values = [models[mode][metric]['mean'] for metric in ('ACC', 'UA', 'WF1') for mode in ('flat', 'joint')]
        lines.append('| ' + rate + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', 'Fold5 provenance remains its original source SHA; unchanged core hashes are checked across all 30 runs.',
              'Historical test-guided research and protected-mask provenance remain internal exploratory limitations.',
              'No automatic additional seeds, folds, methods or HPO.']
    report_outputs(output, dict(runs=runs, checks=checks, per_seed=per_seed, aggregate=aggregate,
                               per_rate=per_rate, reused_fold5=refs, fold_aggregation='unweighted-fold-metric-mean'), rows, lines)
