"""Complete-only paired seed66 report for the four approved MOSEI variants."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL_ROOT = Path('/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66')
OLD_SOURCE = Path('/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/source_v2')
REGISTRY = 'gcnet_missing_m3/meaningful_new40_registry.py'
REGISTRY_EXTENSIONS = ('mosei_nested_d32_l3_g8_m128', 'mosei_nested_d32_l3_g8_m64',
    'mosei_nested_d32_l2_g8_m128', 'mosei_nested_d32_l3_g4_m128',
    'mosei_nested_d32_l3_g4_m32')
VARIANTS = {
    'nested32_lr5e5': ('nested_dim32', 5e-5),
    'nested32_lr2e4': ('nested_dim32', 2e-4),
    'nested32_depth2': ('mosei_nested_dim32_depth2', None),
    'nested32_groups4': ('mosei_nested_d32_l3_g4_m32', None),
}
RATES = tuple(f'{i/10:.1f}' for i in range(8))
METRICS = ('ACC', 'WF1', 'MSE')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_source_extension(root, old_root=OLD_SOURCE):
    root, old_root = Path(root), Path(old_root)
    lines, removed = [], []
    for line in (root/REGISTRY).read_text().splitlines(keepends=True):
        matches = [name for name in REGISTRY_EXTENSIONS if "'"+name+"':" in line]
        if matches:
            removed.extend(matches)
        else:
            lines.append(line)
    if sorted(removed) != sorted(REGISTRY_EXTENSIONS) or ''.join(lines) != (old_root/REGISTRY).read_text():
        raise ValueError('Registry differs beyond the five authorized configuration entries')
    checked = 0
    for package in ('gcnet_missing_m3', 'gcnet_modality_jepa'):
        files = {p.name for p in (old_root/package).glob('*.py')}
        if files != {p.name for p in (root/package).glob('*.py')}:
            raise ValueError('Numerical source file addition/removal')
        for name in files:
            relative = f'{package}/{name}'
            if relative == REGISTRY:
                continue
            if (root/relative).read_bytes() != (old_root/relative).read_bytes():
                raise ValueError('Numerical source changed: '+relative)
            checked += 1
    return dict(unchanged_numerical_files=checked,
        registry_change='five named configuration entries only',
        old_registry_sha256=sha256(old_root/REGISTRY), new_registry_sha256=sha256(root/REGISTRY))


def validate_history(history, *, nested_learning_rate=None):
    expected = [1e-4] if nested_learning_rate is None else [1e-4, nested_learning_rate]
    if len(history) != 60:
        raise ValueError('Require full60epoch history')
    for row in history:
        if row.get('learning_rates') != expected:
            raise ValueError('Exact learning rate group vector or constant schedule differs')


def validate_optimizer(summary, nested_learning_rate):
    if nested_learning_rate is None:
        if summary.get('nested_learning_rate') is not None or summary['identity'].get('nested_learning_rate') is not None:
            raise ValueError('Unexpected local Nested learning rate')
        return
    if nested_learning_rate not in (5e-5, 2e-4):
        raise ValueError('Unapproved Nested learning rate')
    if any(container.get('nested_learning_rate') != nested_learning_rate
           for container in (summary, summary['identity'])):
        raise ValueError('Summary/identity Nested learning rate differs')
    groups = summary.get('optimizer_groups')
    expected = [dict(name='backbone', lr=1e-4,
                     parameters=summary['total_parameters']-summary['nested_parameters']),
                dict(name='nested', lr=nested_learning_rate, parameters=summary['nested_parameters'])]
    if groups != expected or expected[1]['parameters'] <= 0:
        raise ValueError('Optimizer group order, learning rate or parameter count differs')
    if summary['identity'].get('optimizer_groups', groups) != groups:
        raise ValueError('Optimizer group identity differs')


def validate_pair(candidate, history, baseline, base_history, *, nested_learning_rate=None):
    validate_history(history, nested_learning_rate=nested_learning_rate)
    validate_history(base_history)
    validate_optimizer(candidate, nested_learning_rate)
    normalized = copy.deepcopy(history)
    for row in normalized:
        row['learning_rates'] = [1e-4]
    from experiments.osram_mosei_nested_combinations_20261024.report import validate_pair as original_pair
    original_pair(candidate, normalized, baseline, base_history)


def validate_selected_metrics(summary, history):
    if set(summary['best_by_rate']) != set(RATES) or set(summary['evaluation']) != set(RATES):
        raise ValueError('Require all eight selected rates')
    for rate in RATES:
        best, evaluation = summary['best_by_rate'][rate], summary['evaluation'][rate]
        epoch = best['epoch']
        if type(epoch) is not int or not 1 <= epoch <= 60 or not best['same_evaluation_call']:
            raise ValueError('Invalid selected epoch or evaluation call')
        if evaluation.get('selected_epoch') != epoch or any(
                evaluation.get(key) != value for key, value in best['metrics'].items()):
            raise ValueError('Selected metrics differ from published evaluation')
        for metric in METRICS:
            if not math.isfinite(evaluation['router'][metric]):
                raise ValueError('Nonfinite selected metric')
        scores = [row['test'][rate]['router']['WF1'] for row in history]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError('Nonfinite selection history')
        selected = scores.index(max(scores))+1
        if epoch != selected or history[epoch-1]['test'][rate] != best['metrics']:
            raise ValueError('Selection is not earliest per-rate TEST W-F1 maximum')
    for scope, rates in (('mean', RATES), ('high', RATES[5:])):
        for metric in METRICS:
            actual = sum(summary['evaluation'][rate]['router'][metric] for rate in rates)/len(rates)
            reported = summary['aggregate']['router'][scope][metric]
            if not math.isfinite(reported) or not math.isclose(actual, reported, rel_tol=0., abs_tol=1e-9):
                raise ValueError('Aggregate differs from selected predictions')


def validate_jobs(state):
    jobs = state['jobs']
    if len(jobs) != 4 or {job.get('variant') for job in jobs} != set(VARIANTS):
        raise ValueError('Require all four approved variants; no partial ranking')
    by_variant = {job['variant']: job for job in jobs}
    for variant, (method, nested_lr) in VARIANTS.items():
        job = by_variant[variant]
        required = dict(status='complete', method=method, dataset='CMUMOSEI', seed=66,
            fold=1, epochs=60, learning_rate=1e-4, nested_learning_rate=nested_lr,
            source_commit=state['source_commit'])
        if any(job.get(key) != value for key, value in required.items()):
            raise ValueError('Require four complete same-protocol approved jobs: '+variant)
    if len({str(Path(job['output']).resolve()) for job in jobs}) != 4:
        raise ValueError('Each candidate needs an independent output directory')
    return [by_variant[variant] for variant in VARIANTS]


def write_report(output, state):
    jobs = validate_jobs(state)
    from experiments.osram_nested_cmerc_tuning_20261023.run import validate_summary
    from experiments.osram_iemocap6_gcnet_testbest_20261020.worker import validate_selected
    proof = validate_source_extension(ROOT)
    rows, runs = [], []
    baseline = base_history = None
    arms = [('flat', 'none', None, CONTROL_ROOT/'none', None),
            ('nested32', 'nested_dim32', None, CONTROL_ROOT/'nested_dim32', None)]
    arms += [(job['variant'], job['method'], job['nested_learning_rate'], Path(job['output']), job)
             for job in jobs]
    for variant, method, nested_lr, path, job in arms:
        summary = validate_summary(job) if job else json.loads((path/'SUMMARY.json').read_text())
        required = dict(status='complete', phase='train', epoch=60, seed=66,
                        dataset='CMUMOSEI', fold=1, method=method)
        if any(summary.get(key) != value for key, value in required.items()):
            raise ValueError('Require complete same-protocol controls/candidates')
        validate_selected(path, summary['best_by_rate'])
        history = json.loads((path/'history.json').read_text())
        if baseline is None:
            baseline, base_history = summary, history
        validate_pair(summary, history, baseline, base_history, nested_learning_rate=nested_lr)
        validate_selected_metrics(summary, history)
        aggregate = summary['aggregate']['router']
        deltas = {scope: {metric: aggregate[scope][metric]-baseline['aggregate']['router'][scope][metric]
                          for metric in METRICS} for scope in ('mean', 'high')}
        for rate in RATES:
            rows.append(dict(variant=variant, method=method, nested_learning_rate=nested_lr,
                rate=rate, selected_epoch=summary['best_by_rate'][rate]['epoch'],
                **{metric: summary['evaluation'][rate]['router'][metric] for metric in METRICS}))
        runs.append(dict(variant=variant, method=method, nested_learning_rate=nested_lr,
            path=str(path), config=summary['config'], aggregate=aggregate, delta_vs_flat=deltas,
            optimizer_groups=summary.get('optimizer_groups'),
            selected_epoch_by_rate={rate: summary['best_by_rate'][rate]['epoch'] for rate in RATES},
            total_parameters=summary['total_parameters'], nested_parameters=summary['nested_parameters'],
            seconds=summary['seconds'], peak_memory_mib=summary['peak_memory_mib'],
            source_commit=summary['source_commit'], source_hashes=summary['identity']['source_hashes'],
            summary_sha256=sha256(path/'SUMMARY.json'), history_sha256=sha256(path/'history.json')))
    nested_control = runs[1]['aggregate']
    for run in runs:
        run['delta_vs_nested32'] = {scope: {
            metric: run['aggregate'][scope][metric]-nested_control[scope][metric]
            for metric in METRICS} for scope in ('mean', 'high')}
    lines = ['# MOSEI finite Nested local screen', '', 'INTERNAL TEST-ORACLE SCREEN; seed66 only.',
        'Original cmerc-repro, exact accepted test masks,60epochs,batch32,WD1e-5,MSE; global gradient clipping.',
        'Backbone LR remains constant1e-4; only the two local-LR arms use their stated Nested LR.',
        'Flat and Nested32 controls are reused read-only. All four approved candidates must complete.',
        'Per-rate TEST-best W-F1 selects eight potentially different epochs; this is not a single-checkpoint deployment score.',
        'No statistical-significance or independent-generalization claim; no automatic promotion.', '',
        '| Variant | Scope | ACC | ΔACC pp | W-F1 | ΔWF1 pp | MSE | ΔMSE |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        for scope in ('mean', 'high'):
            values, delta = run['aggregate'][scope], run['delta_vs_flat'][scope]
            lines.append(f"| {run['variant']} | {scope} | {values['ACC']:.4f} | {delta['ACC']:+.4f} | {values['WF1']:.4f} | {delta['WF1']:+.4f} | {values['MSE']:.6f} | {delta['MSE']:+.6f} |")
    lines += ['', 'Deltas below compare the same selected aggregates with the original Nested32 control.', '',
        '| Variant | Scope | ΔACC pp | ΔWF1 pp | ΔMSE |', '|---|---|---:|---:|---:|']
    for run in runs:
        for scope in ('mean', 'high'):
            delta = run['delta_vs_nested32'][scope]
            lines.append(f"| {run['variant']} | {scope} | {delta['ACC']:+.4f} | {delta['WF1']:+.4f} | {delta['MSE']:+.6f} |")
    lines += ['', '| Variant | Total params | Nested params | Seconds | Peak MiB | Selected epochs (rates0.0–0.7) | Source SHA |',
        '|---|---:|---:|---:|---:|---|---|']
    for run in runs:
        epochs = ','.join(str(run['selected_epoch_by_rate'][rate]) for rate in RATES)
        lines.append(f"| {run['variant']} | {run['total_parameters']} | {run['nested_parameters']} | {run['seconds']:.1f} | {run['peak_memory_mib']:.1f} | {epochs} | {run['source_commit']} |")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    (output/'RESULT.json').write_text(json.dumps(dict(status='complete', label='INTERNAL TEST-ORACLE SCREEN',
        seed=66, source_commit=state['source_commit'], source_extension_proof=proof,
        runs=runs), indent=2, sort_keys=True, allow_nan=False)+'\n')
    with (output/'PER_RATE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
