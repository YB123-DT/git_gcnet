"""Internal per-rate oracle scores, not performance of a single checkpoint."""
from pathlib import Path

from experiments.osram_mosi_gcnet_router_3seed_20261019.report import (
    CORE, RATES, load_pair, masks, report_outputs, stats,
)


def validate_policy(summary):
    cfg = summary['identity']['config']
    if (cfg.get('checkpoint_selection') != 'test-oracle-per-rate'
            or cfg.get('checkpoint_selection_metric') != 'weighted_f1'
            or cfg.get('missing_mask_protocol') != 'gcnet-random-mask-v1'):
        raise ValueError('Require GCNet masks and explicit per-rate Test-best WF1')
    epochs = summary.get('selected_epoch_by_rate', {})
    if set(epochs) != set(RATES):
        raise ValueError('Require eight actual selected epochs')
    if any(type(epoch) is not int or not 1 <= epoch <= 150 for epoch in epochs.values()):
        raise ValueError('Invalid selected epoch')
    return epochs


def write_report(output, queue):
    output = Path(output)
    if len(queue['jobs']) != 6 or any(j['status'] != 'complete' for j in queue['jobs']):
        raise ValueError('Require exactly six completed MOSI Test-best jobs')
    runs, checks, rows, core, shared_cohort = {}, {}, [], None, None
    lines = ['# MOSI GCNet missing: per-rate Test-best', '',
        'INTERNAL TEST-ORACLE DIAGNOSTIC ONLY', '',
        'Seeds66/67/68 x Flat/Router,150epochs. Original gcnet-random-mask-v1.',
        'Each rate selects its own epoch by TEST W-F1; ties retain the earliest epoch.',
        'These means combine eight checkpoints and are NOT single-checkpoint deployment scores.',
        'The fixed GCNet evaluation banks and training settings match the preceding validation-selected runs.',
        'Only checkpoint selection changes; old results/checkpoints remain untouched.',
        'Old80.747 still used a different missing sampler, even though its Test-best policy is restored.', '',
        '| Seed | Model | ACC | W-F1 | High ACC | High W-F1 |',
        '|---:|---|---:|---:|---:|---:|']
    for seed in (66, 67, 68):
        jobs = [j for j in queue['jobs'] if j['seed'] == seed]
        if len(jobs) != 2 or {j['mode'] for j in jobs} != {'flat', 'joint'}:
            raise ValueError('Each seed requires one matched Flat/Router pair')
        pair, checked = load_pair({j['mode']: j['output'] for j in jobs}, dataset='CMUMOSI', seed=seed)
        actual_core = {name: pair['flat']['identity']['source_hashes'][name] for name in CORE}
        cohort = dict(masks=masks(pair['flat']), split=pair['flat']['identity']['split'])
        if core is not None and core != actual_core:
            raise ValueError('Numerical core differs between seeds')
        if shared_cohort is not None and shared_cohort != cohort:
            raise ValueError('Evaluation masks/split differ between seeds')
        core, shared_cohort = actual_core, cohort
        runs[str(seed)], checks[str(seed)] = pair, checked
        for mode, summary in pair.items():
            epochs = validate_policy(summary)
            values = [summary['aggregate']['router'][scope][metric]
                for scope in ('mean', 'high') for metric in ('ACC', 'WF1')]
            lines.append(f'| {seed} | {mode} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
            for rate in RATES:
                rows.append(dict(dataset='CMUMOSI', seed=seed, mode=mode, rate=rate,
                    selected_epoch=epochs[rate], **summary['evaluation'][rate]['router']))
    aggregate = {mode: {scope: {metric: stats([
        runs[str(seed)][mode]['aggregate']['router'][scope][metric] for seed in (66, 67, 68)])
        for metric in ('ACC', 'WF1', 'MSE')} for scope in ('mean', 'high')}
        for mode in ('flat', 'joint')}
    per_rate = {rate: {mode: {metric: stats([
        runs[str(seed)][mode]['evaluation'][rate]['router'][metric] for seed in (66, 67, 68)])
        for metric in ('ACC', 'WF1', 'MSE')} for mode in ('flat', 'joint')} for rate in RATES}
    lines += ['', '## Three-seed mean ± sample standard deviation', '',
        '| Model | ACC | W-F1 | High ACC | High W-F1 |', '|---|---:|---:|---:|---:|']
    for mode, scopes in aggregate.items():
        values = [f'{scopes[s][m]["mean"]:.4f} ± {scopes[s][m]["std"]:.4f}'
            for s in ('mean', 'high') for m in ('ACC', 'WF1')]
        lines.append('| ' + mode + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## Per-rate three-seed means', '',
        '| Rate | Flat ACC | Router ACC | Flat W-F1 | Router W-F1 |', '|---|---:|---:|---:|---:|']
    for rate, models in per_rate.items():
        values = [models[mode][metric]['mean'] for metric in ('ACC', 'WF1') for mode in ('flat', 'joint')]
        lines.append('| ' + rate + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', 'All selected epochs are listed in METRICS.csv and each run SUMMARY.json.',
        'Selection uses test labels: exploratory only, no independent generalization claim.',
        'No automatic extra seeds, datasets, HPO or model changes.']
    report_outputs(output, dict(runs=runs, checks=checks, aggregate=aggregate, per_rate=per_rate,
        checkpoint_selection='test-oracle-per-rate', mask_protocol='gcnet-random-mask-v1'), rows, lines)
