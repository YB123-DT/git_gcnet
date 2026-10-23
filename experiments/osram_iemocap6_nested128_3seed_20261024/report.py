"""Strict fifteen-run reporting; no Torch import or new inference/selection.

Metric key ``router`` is the historical evaluator's name for ordinary Nested
predictions. It does not mean this experiment adds a prediction router.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
METRICS = ('ACC', 'UA', 'WF1', 'CE')
REUSED = Path('/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/IEMOCAPSix/fold_5/seed_66/nested_dim128')
REQUIRED_ENVIRONMENT = dict(executable='/data2/yb/conda_envs/cmerc-repro/bin/python',
    torch='1.13.1+cu117', cuda='11.7', numpy='1.26.4', sklearn='1.3.2')
REQUIRED_CONFIG = dict(dataset='IEMOCAPSix', epochs=100, learning_rate=.001,
    weight_decay=1e-5, optimizer='adam', lr_schedule='constant', batch_size=32,
    missing_mask_protocol='gcnet-random-mask-v1', train_rate_mode='cyclic',
    training_objective='emotion-only', checkpoint_selection='test-oracle-per-rate',
    checkpoint_selection_metric='accuracy', evaluate_test=True,
    osram_meaningful_block='nested_dim128', osram_output_dim=1600,
    osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
    osram_write_step=.6, osram_bidirectional=False, osram_readout_fusion='flat',
    osram_disable_local_skip=False)
DISABLED = ('paired_history_views', 'osram_tasr', 'osram_source_role',
    'osram_source_role_v3', 'osram_source_relation', 'osram_source_tracking',
    'osram_conditional_reread', 'osram_base_first_gap', 'osram_history_recency',
    'osram_causal_gru_history', 'osram_relation_block', 'osram_relation_dual_readout',
    'osram_gap_increment_filter', 'osram_decision_correction',
    'osram_history_query_adapter', 'osram_post_grn', 'osram_local_skip_gate',
    'osram_memory_only_adapter', 'osram_history_input_gate', 'osram_local_evidence_gate',
    'osram_hierarchical_evidence_gate', 'osram_ced_block')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _same(actual, expected, context):
    if actual != expected:
        raise ValueError(context + ' mismatch')


def _close(actual, expected, context):
    if (isinstance(actual, bool) or not isinstance(actual, (float, int))
            or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9)):
        raise ValueError(context + ' nonfinite or mismatch')


def _artifact(root, filename, expected_hash):
    filename = Path(filename)
    if filename.is_absolute() or '..' in filename.parts:
        raise ValueError('Selected artifact escapes output')
    path = root / filename
    if not path.is_file() or sha256(path) != expected_hash:
        raise ValueError('Selected artifact integrity failure: ' + str(path))


def load_run(path, seed, fold):
    """Validate full budget, immutable artifacts and exact same-call metrics.

    Hashes bind retained predictions/checkpoints, not a fresh random re-evaluation.
    No model is loaded. The training evaluator is responsible for computing the
    selected metrics from those predictions; this checks its publication ledger,
    fixed mask identity and earliest strict-ACC selection against all 100 epochs.
    """
    path = Path(path)
    summary = _json(path / 'SUMMARY.json')
    expected = dict(status='complete', phase='train', dataset='IEMOCAPSix',
        method='nested_dim128', seed=seed, fold=fold, epoch=100,
        total_parameters=13950441, nested_parameters=432387)
    for name, value in expected.items():
        _same(summary.get(name), value, 'Completed run ' + name)
    if seed not in (66, 67, 68) or fold not in range(1, 6):
        raise ValueError('Unapproved seed/fold')
    config = summary['config']
    for key, value in dict(REQUIRED_CONFIG, seed=seed, fold=fold).items():
        _same(config.get(key), value, 'Protected configuration ' + key)
    for key in DISABLED:
        if config.get(key, False):
            raise ValueError('Additional module enabled: ' + key)
    for key in ('completion_path', 'osram_readout_candidate', 'osram_counterfactual_readout'):
        _same(config.get(key, 'none'), 'none', key)
    if 'train_missing_rates' in config:
        _same(config['train_missing_rates'], [float(r) for r in RATES], 'Cyclic rates')
    _same(_json(path / 'config.json'), config, 'Retained config')
    identity = summary['identity']
    _same(identity['config'], config, 'Identity config')
    environment = summary['environment']
    for key, value in REQUIRED_ENVIRONMENT.items():
        _same(environment.get(key), value, 'cmerc environment ' + key)
        _same(identity['environment'].get(key), value, 'Identity environment ' + key)
    source = identity['source_hashes']
    if not all(k in source for k in ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py')):
        raise ValueError('Numerical model source hashes missing')
    history = _json(path / 'history.json')
    if len(history) != 100 or [row['epoch'] for row in history] != list(range(1, 101)):
        raise ValueError('Full 100 epoch history required')
    for artifact in ('last.pt', 'test_masks.pt'):
        if not (path / artifact).is_file():
            raise ValueError('Required retained artifact missing: ' + artifact)
    best, evaluation = summary['best_by_rate'], summary['evaluation']
    aggregate = summary['aggregate']['router']
    for mapping in (best, evaluation, aggregate['rates'], identity['mask_reference']):
        _same(set(mapping), set(RATES), 'All eight missing rates')
    for rate in RATES:
        row, evaluated = best[rate], evaluation[rate]
        if (row.get('same_evaluation_call') is not True
                or row.get('criterion') != 'TEST accuracy; strict improvement; earliest tie'):
            raise ValueError('Same-call TEST ACC selection required')
        _same(evaluated['selected_epoch'], row['epoch'], 'Selected epoch')
        mask_hash = row['metrics']['mask_sha256']
        _same(evaluated['mask_sha256'], mask_hash, 'Selected evaluation mask')
        _same(identity['mask_reference'][rate]['mask_sha256'], mask_hash, 'Fixed accepted test mask')
        for field, digest in (('checkpoint', 'checkpoint_sha256'), ('predictions', 'prediction_sha256')):
            _artifact(path, row[field], row[digest])
        _same(evaluated['output_sha256'], row['prediction_sha256'], 'Published prediction hash')
        _artifact(path, 'evaluation/random_' + rate.replace('.', 'p') + '.npz', evaluated['output_sha256'])
        scores = [entry['test'][rate]['router']['ACC'] for entry in history]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in scores):
            raise ValueError('Nonfinite epoch accuracy')
        epoch = max(range(100), key=scores.__getitem__) + 1
        _same(row['epoch'], epoch, 'Earliest strict TEST ACC best epoch')
        for metric in METRICS:
            selected = row['metrics']['router'][metric]
            for value in (evaluated['router'][metric], aggregate['rates'][rate][metric],
                          history[epoch - 1]['test'][rate]['router'][metric]):
                _close(value, selected, 'Same-call ' + metric)
    for scope, rates in (('mean', RATES), ('high', RATES[5:])):
        for metric in METRICS:
            average = statistics.mean(evaluation[r]['router'][metric] for r in rates)
            _close(aggregate[scope][metric], average, 'Aggregate ' + scope + ' ' + metric)
    return summary


def validate_reused(path=REUSED):
    """The sole reusable trajectory is the already completed seed66 fold5."""
    return load_run(path, 66, 5)


def _numerical_sources(summary):
    return {k: v for k, v in summary['identity']['source_hashes'].items()
            if k.startswith(('gcnet_missing_m3/', 'gcnet_modality_jepa/'))}


def write_report(output, state):
    """Publish only when all fifteen approved seed/fold trajectories are valid."""
    jobs = state['jobs']
    reused = state['reused']
    expected = {(seed, fold) for seed in (66, 67, 68) for fold in range(1, 6)}
    records = jobs + [reused]
    if (len(jobs) != 14 or len(records) != 15
            or {(j['seed'], j['fold']) for j in records} != expected
            or any(j['status'] != 'complete' for j in jobs)
            or reused['status'] != 'reused_complete'
            or (reused['seed'], reused['fold']) != (66, 5)):
        raise ValueError('Require 14 complete new jobs plus reused seed66 fold5: all 15, no partial means')
    runs, rows, compact = {}, [], []
    for job in records:
        seed, fold, path = job['seed'], job['fold'], Path(job['output'])
        summary = load_run(path, seed, fold)
        if job is not reused:
            _same(summary['source_commit'], state['source_commit'], 'New run source commit')
        runs[seed, fold] = summary
        compact.append(dict(seed=seed, fold=fold, output=str(path), reused=job is reused,
            source_commit=summary['source_commit'], summary_sha256=sha256(path / 'SUMMARY.json'),
            aggregate=summary['aggregate']['router'], total_parameters=summary['total_parameters'],
            nested_parameters=summary['nested_parameters'], seconds=summary['seconds'],
            peak_memory_mib=summary['peak_memory_mib'],
            selected={r: {k: v for k, v in summary['best_by_rate'][r].items() if k != 'metrics'}
                      for r in RATES}))
        for rate in RATES:
            selected = summary['best_by_rate'][rate]
            rows.append(dict(seed=seed, fold=fold, rate=rate, selected_epoch=selected['epoch'],
                **{m: summary['evaluation'][rate]['router'][m] for m in METRICS}))
    canonical = _numerical_sources(runs[66, 5])
    for key, summary in runs.items():
        _same(_numerical_sources(summary), canonical, 'Numerical model source')
        _same(summary['identity']['mask_reference'], runs[66, key[1]]['identity']['mask_reference'],
              'Same-fold test mask across seeds')
    seeds = {}
    for seed in (66, 67, 68):
        by_rate = {r: {m: statistics.mean(runs[seed, fold]['evaluation'][r]['router'][m]
                                        for fold in range(1, 6)) for m in METRICS} for r in RATES}
        seeds[str(seed)] = dict(rates=by_rate)
        for scope, rates in (('mean', RATES), ('high', RATES[5:])):
            seeds[str(seed)][scope] = {m: statistics.mean(by_rate[r][m] for r in rates) for m in METRICS}
    three_seed = {'rates': {}}
    for scope in ('mean', 'high'):
        three_seed[scope] = {m: dict(mean=statistics.mean(seeds[str(s)][scope][m] for s in (66, 67, 68)),
            sample_sd=statistics.stdev(seeds[str(s)][scope][m] for s in (66, 67, 68))) for m in METRICS}
    for rate in RATES:
        three_seed['rates'][rate] = {m: dict(mean=statistics.mean(seeds[str(s)]['rates'][rate][m]
             for s in (66, 67, 68)), sample_sd=statistics.stdev(seeds[str(s)]['rates'][rate][m]
             for s in (66, 67, 68))) for m in METRICS}
    lines = ['# IEMOCAP-6 Nested128: five folds, three seeds', '',
        'INTERNAL TEST-ORACLE DIAGNOSTIC ONLY', '',
        'Fixed nested_dim128, 100 epochs, Adam LR=1e-3, WD=1e-5, batch32; cmerc-repro.',
        'Original gcnet-random-mask-v1 cyclic training; fixed accepted per-fold test masks.',
        'Each rate selects TEST ACC with earliest strict improvement; UA/W-F1/CE use the SAME call.',
        'These eight-rate selections are not a single deployable checkpoint.',
        'Only completed trajectories enter this report: 14 fresh runs plus existing seed66 fold5.',
        'Nested128: 13,950,441 total parameters; 432,387 additional Nested parameters.',
        'Five-fold metrics are equally weighted fold means, not pooled utterance metrics.',
        'Across-seed SD uses n-1 on the three five-fold means.',
        'No strict comparison to historical Flat: old IEMOCAP runs resampled evaluation masks and',
        'used a different fold-initialization treatment. No matched fifteen-run Flat was trained here.',
        'The selection of Nested128 itself used seed66 fold5; this is exploratory, not independent generalization.', '',
        '## Five-fold results by seed', '',
        '| Seed | ACC | UA | W-F1 | CE | High ACC | High UA | High W-F1 | High CE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed, values in seeds.items():
        cells = [values[scope][m] for scope in ('mean', 'high') for m in METRICS]
        lines.append('| ' + seed + ' | ' + ' | '.join(f'{v:.3f}' for v in cells) + ' |')
    cells = [three_seed[scope][m] for scope in ('mean', 'high') for m in METRICS]
    lines.append('| Mean ± sample SD | ' + ' | '.join(f"{v['mean']:.3f} ± {v['sample_sd']:.3f}"
                                                  for v in cells) + ' |')
    lines += ['', '## Per-rate three-seed five-fold mean', '',
        '| Missing | ACC | UA | W-F1 | CE |', '|---|---:|---:|---:|---:|']
    for rate in RATES:
        values = three_seed['rates'][rate]
        lines.append('| ' + rate + ' | ' + ' | '.join(
            f"{values[m]['mean']:.3f} ± {values[m]['sample_sd']:.3f}" for m in METRICS) + ' |')
    lines += ['', 'PER_FOLD_RATE.csv preserves all 120 fold/seed/rate selections and epochs.',
        'PER_SEED_RATE.csv preserves the 24 equally weighted five-fold rate summaries.',
        'Best checkpoints, last.pt, masks and exact selected predictions remain in the remote run directories.',
        'No new model, loss, HPO or extra dataset was introduced. This report makes no mechanism claim.', '']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(status='complete', source_commit=state['source_commit'],
        label='INTERNAL TEST-ORACLE DIAGNOSTIC ONLY', fold_aggregation='equal fold means',
        runs=compact, seeds=seeds, three_seed=three_seed)
    (output / 'RESULT.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    (output / 'RESULT.md').write_text('\n'.join(lines))
    rows.sort(key=lambda r: (r['seed'], r['fold'], r['rate']))
    seed_rows = [dict(seed=seed, rate=rate, **value['rates'][rate])
                 for seed, value in seeds.items() for rate in RATES]
    for filename, values in (('PER_FOLD_RATE.csv', rows), ('PER_SEED_RATE.csv', seed_rows)):
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    return payload
