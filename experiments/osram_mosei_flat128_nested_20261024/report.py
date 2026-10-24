"""Complete-only Flat128 seed controls and the finite, paired Nested screen."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics

from .config import (BASE_SOURCE, REUSED_CONTROL, VARIANTS, method_name, run_spec,
                     verify_registry_extension)
from experiments.osram_mosei_nested_local_20261024.report import validate_selected_metrics

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(f'{i/10:.1f}' for i in range(8))
METRICS = ('ACC', 'WF1', 'MSE')
REGISTRY = 'gcnet_missing_m3/meaningful_new40_registry.py'
FRESH = ('F67', 'F68', *VARIANTS)
_CACHE = {}
REQUIRED_CONFIG = dict(dataset='CMUMOSEI', fold=1, epochs=60,
    learning_rate=1e-4, batch_size=32, weight_decay=1e-5, optimizer='adam',
    lr_schedule='constant', missing_mask_protocol='gcnet-random-mask-v1',
    training_objective='emotion-only', train_rate_mode='cyclic',
    train_missing_rates=[i/10 for i in range(8)], checkpoint_selection='test-oracle-per-rate',
    checkpoint_selection_metric='weighted_f1', evaluate_test=True,
    osram_adapter_hidden_dim=128, osram_output_dim=1600,
    osram_num_heads=8, osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
    osram_disable_local_skip=False, osram_bidirectional=False, osram_readout_fusion='flat',
    paired_history_views=False, osram_source_role_v3=False, osram_tasr=False,
    osram_source_role=False, osram_source_relation=False, osram_source_tracking=False,
    completion_path='none', initial_backbone_checkpoint=None,
    mosi_task_mode='regression', task_regression_loss='mse')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pair(candidate, history, baseline, base_history, *, variant):
    """Seed66 is paired; the two explicit Flat seeds vary training randomness."""
    spec = run_spec(variant)
    expected = dict(REQUIRED_CONFIG, seed=spec['seed'], osram_meaningful_block=spec['method'])
    if any(candidate['config'].get(key) != value for key, value in expected.items()):
        raise ValueError('Protected Flat128 training configuration differs: '+variant)
    if candidate['config'] != candidate['identity']['config']:
        raise ValueError('Configuration identity differs')
    if candidate['seed'] != spec['seed'] or candidate['method'] != spec['method']:
        raise ValueError('Run seed/method differs from approved variant')
    seed_protocol = candidate['identity']['seed_protocol']
    if any(seed_protocol.get(k) != v for k, v in dict(training_seed=spec['seed'],
            split_seed=66, evaluation_seed=66).items()):
        raise ValueError('Training seed protocol or accepted split/evaluation seed differs')
    if spec['seed'] == 66:
        if candidate['initial_backbone_sha256'] != baseline['initial_backbone_sha256']:
            raise ValueError('Same-width non-Nested initialization differs')
        if seed_protocol != baseline['identity']['seed_protocol']:
            raise ValueError('Seed66 loader protocol differs')
    else:
        if candidate['initial_backbone_sha256'] == baseline['initial_backbone_sha256']:
            raise ValueError('Fresh Flat seed reused seed66 initialization')
        if seed_protocol['train_loader'] == baseline['identity']['seed_protocol']['train_loader']:
            raise ValueError('Fresh Flat seed reused seed66 training loader')
        if ([row['mask_trace_sha256'] for row in history]
                == [row['mask_trace_sha256'] for row in base_history]):
            raise ValueError('Fresh Flat seed reused seed66 training mask trace')
    if candidate['config']['osram_adapter_hidden_dim'] != baseline['config']['osram_adapter_hidden_dim']:
        raise ValueError('Same-width Flat128 control required')
    numerical = lambda item: {k: v for k, v in item['identity']['source_hashes'].items()
        if k.startswith(('gcnet_missing_m3/', 'gcnet_modality_jepa/')) and k != REGISTRY}
    if numerical(candidate) != numerical(baseline):
        raise ValueError('Numerical model/sampler source differs')
    if [row['epoch'] for row in history] != list(range(1, 61)):
        raise ValueError('Require ordered complete60epoch history')
    nested = candidate['nested_parameters']
    if (candidate['total_parameters'] != 4746977+nested or
            (nested != 0 if variant.startswith('F') else nested <= 0)):
        raise ValueError('Flat128 or Nested parameter budget differs')
    normalized, normalized_history = candidate, history
    if spec['seed'] != 66:
        normalized, normalized_history = copy.deepcopy(candidate), copy.deepcopy(history)
        normalized['config']['seed'] = 66
        if len(history) != len(base_history):
            raise ValueError('Training budget differs')
        for row, old in zip(normalized_history, base_history):
            row['mask_trace_sha256'] = old['mask_trace_sha256']
    from experiments.osram_mosei_capacity_cmerc_20261024.report import validate_pair as capacity_pair
    capacity_pair(normalized, normalized_history, baseline, base_history)


def _validate_artifacts(path, summary, history):
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
    validate_selected(path, summary['best_by_rate'])
    validate_selected_metrics(summary, history)
    for rate in RATES:
        published = path/'evaluation'/f"random_{rate.replace('.', 'p')}.npz"
        selected = summary['best_by_rate'][rate]['prediction_sha256']
        if (summary['evaluation'][rate].get('output_sha256') != selected
                or sha256(published) != selected):
            raise ValueError('Published selected prediction integrity differs: '+rate)


def _load(path, variant, baseline=None, base_history=None, *, final=False, base_signature=None):
    path = Path(path)
    file = path/'SUMMARY.json'
    stamp = file.stat()
    signature = (stamp.st_mtime_ns, stamp.st_size, base_signature)
    cached = _CACHE.get(str(path.resolve()))
    if not final and cached is not None and cached['signature'] == signature:
        return cached['summary'], cached['history'], cached['item']
    summary = json.loads(file.read_text())
    required = dict(status='complete', phase='train', dataset='CMUMOSEI', fold=1,
                    epoch=60, **run_spec(variant))
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError('Require complete60epoch approved result: '+variant)
    history = json.loads((path/'history.json').read_text())
    validate_pair(summary, history, summary if baseline is None else baseline,
                  history if base_history is None else base_history, variant=variant)
    _validate_artifacts(path, summary, history)
    item = dict(variant=variant, path=str(path), reused=variant=='F66', **run_spec(variant),
        architecture=dict(flat_hidden=128, output=1600, chm_heads=8, chm_head_dim=64,
                          nested=VARIANTS.get(variant)),
        aggregate=summary['aggregate']['router'], config=summary['config'], identity=summary['identity'],
        environment=summary['environment'], source_commit=summary['source_commit'],
        source_hashes=summary['identity']['source_hashes'],
        initial_backbone_sha256=summary['initial_backbone_sha256'],
        total_parameters=summary['total_parameters'], nested_parameters=summary['nested_parameters'],
        seconds=summary['seconds'], timing=dict(wall_seconds_per_epoch=summary['seconds']/60,
            training_seconds_by_epoch=None,
            epoch_timing_note='Wall average includes setup/train/eval/save; training-only epoch times unavailable.'),
        peak_memory_mib=summary['peak_memory_mib'], gpu=summary.get('gpu'),
        selected_epoch_by_rate={r: summary['best_by_rate'][r]['epoch'] for r in RATES},
        best_by_rate=summary['best_by_rate'], evaluation=summary['evaluation'], epochs=history,
        summary_sha256=sha256(file), history_sha256=sha256(path/'history.json'))
    _CACHE[str(path.resolve())] = dict(signature=signature, summary=summary, history=history, item=item)
    return summary, history, item


def validate_reused_control(path):
    """Read-only preflight proof for the sole reused result; never trust a cache."""
    _, _, item = _load(Path(path), 'F66', final=True)
    return {key: item[key] for key in ('path', 'summary_sha256', 'history_sha256',
        'source_commit', 'source_hashes', 'config', 'environment', 'initial_backbone_sha256',
        'total_parameters', 'nested_parameters', 'aggregate')}


def _jobs(state):
    jobs = state['jobs']
    if len(jobs) != 23 or {job.get('variant') for job in jobs} != set(FRESH):
        raise ValueError('Require exactly23 unique approved fresh variants')
    paths = {str(Path(job['output']).resolve()) for job in jobs}
    if len(paths) != 23 or str(Path(REUSED_CONTROL).resolve()) in paths:
        raise ValueError('Require independent output directories; preserve reused F66')
    for job in jobs:
        spec = run_spec(job['variant'])
        if any(key in job and job[key] != value for key, value in spec.items()):
            raise ValueError('Job method/seed differs from variant')
    return {job['variant']: job for job in jobs}


def _write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def summarize(output, state, *, final=False):
    jobs = _jobs(state)
    proof = verify_registry_extension(ROOT, state.get('reused_source', BASE_SOURCE))
    baseline, base_history, base_item = _load(REUSED_CONTROL, 'F66', final=final)
    if base_item['source_hashes'].get(REGISTRY) != proof['base_registry_sha256']:
        raise ValueError('Reused control registry identity differs from verified baseline source')
    base_signature = (base_item['summary_sha256'], base_item['history_sha256'])
    runs, states = {'F66': copy.deepcopy(base_item)}, {}
    for variant in ('F66', *FRESH):
        job = jobs.get(variant, dict(output=str(REUSED_CONTROL), status='reused_complete'))
        path = Path(job['output'])
        file = path/'SUMMARY.json'
        summary = json.loads(file.read_text()) if file.exists() else {}
        states[variant] = dict(job, **{key: summary[key] for key in
            ('status', 'epoch', 'error', 'pid', 'gpu', 'seconds', 'peak_memory_mib',
             'total_parameters', 'nested_parameters') if key in summary})
        states[variant]['reused'] = variant=='F66'
        if variant=='F66' or summary.get('status') != 'complete':
            continue
        _, _, item = _load(path, variant, baseline, base_history, final=final,
                           base_signature=base_signature)
        if item['source_hashes'].get(REGISTRY) != proof['registry_sha256']:
            raise ValueError('Candidate registry identity differs from verified source: '+variant)
        if item['source_commit'] != job.get('source_commit', state.get('source_commit')):
            raise ValueError('Candidate source commit differs from submitted job: '+variant)
        runs[variant] = copy.deepcopy(item)
    if final and len(runs) != 24:
        raise ValueError('Final report requires all24 complete result slots')
    rows = []
    for variant, item in runs.items():
        if item['seed'] == 66:
            item['delta_vs_flat66'] = {scope: {m: item['aggregate'][scope][m]-base_item['aggregate'][scope][m]
                for m in METRICS} for scope in ('mean', 'high')}
        for rate in RATES:
            rows.append(dict(variant=variant, method=item['method'], seed=item['seed'], rate=rate,
                selected_epoch=item['selected_epoch_by_rate'][rate], **item['evaluation'][rate]['router']))
    ranking = sorted((variant for variant in runs if variant in VARIANTS), key=lambda variant: (
        -runs[variant]['aggregate']['mean']['WF1'], -runs[variant]['aggregate']['high']['WF1'],
        runs[variant]['total_parameters'], variant))
    flat_statistics = None
    if all(variant in runs for variant in ('F66', 'F67', 'F68')):
        flat_statistics = {scope: {metric: dict(
            mean=statistics.mean(runs[v]['aggregate'][scope][metric] for v in ('F66', 'F67', 'F68')),
            sample_sd=statistics.stdev(runs[v]['aggregate'][scope][metric] for v in ('F66', 'F67', 'F68')))
            for metric in METRICS} for scope in ('mean', 'high')}
    result = dict(label='INTERNAL TEST-ORACLE SCREEN', complete=final, states=states,
        source_commit=state.get('source_commit'), source_extension_proof=proof,
        runs=runs, nested_ranking=ranking, flat_three_seed=flat_statistics,
        ranking_rule='completed Nested only: mean WF1, high-missing WF1, lower total parameters',
        paired_comparison='Nested seed66 minus Flat128 seed66 only', automatic_promotion=False)
    lines = ['# MOSEI Flat128 finite Nested screen', '', 'INTERNAL TEST-ORACLE SCREEN.',
        'cmerc-repro, original accepted split/exact test masks, GCNet random mask,60epochs,constant LR1e-4,batch32,WD1e-5,MSE.',
        'All Flat hidden128/output1600; CHM remains8x64 with write step0.6. F66 is reused read-only.',
        'Per-rate TEST-best W-F1 selects the earliest maximum and can select different epochs; this is not a single-checkpoint deployment score.',
        'Nested candidates use seed66 only. Paired deltas use Flat128 seed66; Flat three-seed statistics are reported separately.',
        'No independent generalization or statistical-significance claim; no automatic promotion or search expansion.', '',
        '| Complete slot | Seed | ACC | W-F1 | High W-F1 | Paired ΔWF1 vs F66 | Total params | Nested params | Seconds | Peak MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for variant in ('F66', 'F67', 'F68', *ranking):
        if variant not in runs:
            continue
        item = runs[variant]
        mean, high = item['aggregate']['mean'], item['aggregate']['high']
        delta = f"{item['delta_vs_flat66']['mean']['WF1']:+.6f}" if item['seed']==66 else 'N/A'
        lines.append(f"| {variant} | {item['seed']} | {mean['ACC']:.6f} | {mean['WF1']:.6f} | {high['WF1']:.6f} | {delta} | {item['total_parameters']} | {item['nested_parameters']} | {item['seconds']:.1f} | {item['peak_memory_mib']:.1f} |")
    if flat_statistics is not None:
        lines += ['', 'Flat128 seeds66/67/68, mean ± sample standard deviation (descriptive only):', '',
                  '| Scope | ACC | W-F1 | MSE |', '|---|---:|---:|---:|']
        for scope, stats in flat_statistics.items():
            values = [f"{stats[m]['mean']:.6f} ± {stats[m]['sample_sd']:.6f}" for m in METRICS]
            lines.append('| '+scope+' | '+' | '.join(values)+' |')
    lines += ['', 'Completed Nested ranking: '+(', '.join(ranking) or 'none'), '',
              'Progress (all24 result slots):', '', '| Slot | State | Epoch | GPU | PID | Output |',
              '|---|---|---:|---|---:|---|']
    for variant, item in states.items():
        lines.append(f"| {variant} | {item.get('status', 'pending')} | {item.get('epoch', '')} | {item.get('gpu', '')} | {item.get('pid', '')} | {item['output']} |")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    _write(output/'REPORT.json', result)
    with (output/'PER_RATE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant', 'method', 'seed', 'rate', 'selected_epoch', *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    if final:
        _write(output/'RESULT.json', result)
    return result
