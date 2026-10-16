"""Aggregate verified formal artifacts under the explicitly biased test oracle."""
import argparse
import ast
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from common_data import fixed_masks, load_dataset
from runtime import ROOT, compute_metrics, valid_result

METRICS = ('acc', 'wf1', 'ua', 'macro_f1')
SEEDS = (66, 67, 68)
BIAS = ('test_oracle: checkpoints selected by Test ACC; results have optimistic '
        'test-selection bias and are not independent, unbiased final test estimates.')
KEYS = ('dataset', 'model', 'model_version', 'missing_rate')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def model_label(model, version, dataset):
    if model == 'MMIN':
        return 'MMIN utterance-feature-adapted Original'
    if model == 'SDR-GNN':
        label = {'released_compatible': 'SDR-GNN released-code compatible',
                 'implementation_fixed': 'SDR-GNN implementation-fixed'}.get(version)
        if label is None:
            raise ValueError('Unknown SDR-GNN version: ' + version)
        return label + (' (MELD-adapted)' if dataset == 'MELD' else '')
    if model == 'GCNet' and dataset == 'MELD':
        return 'GCNet MELD-adapted Original'
    return model + ('-Causal' if version == 'causal' else ' Original')


def _source_provenance(config, root, snapshot_path):
    """Keep verified source anchors separate from an executed code snapshot.

    An upstream/adapter commit documents code ancestry, not a completed run.
    New local adapter changes are identified by the actual snapshot hash only
    after launch; they must never be presented as a new committed revision.
    """
    current = Path(root)
    snapshot = Path(snapshot_path) if snapshot_path else None
    model = config['model']
    paths = {'ComP': 'adapters/comp/SOURCE.json',
             'MMIN': 'adapters/mmin/provenance.json',
             'GCNet': 'adapters/graph_models/gcnet/provenance.json',
             'SDR-GNN': 'adapters/graph_models/sdrgnn/provenance.json'}
    if model == 'SDR-GNN' and config.get('model_version') == 'released_compatible':
        paths['SDR-GNN'] = 'adapters/graph_models/sdrgnn_released/PROVENANCE.json'
    base = snapshot if snapshot is not None and (snapshot/'adapters').exists() else current
    origin = 'executed_snapshot' if snapshot is not None and base == snapshot else 'current_verified_adapter_ancestry'
    value = read_json(base/paths[model]) if model in paths else {}
    source, adapter, repository, note = '', '', '', ''
    if model == 'ComP':
        source, adapter = value.get('upstream_commit', ''), value.get('source_commit', '')
        repository = value.get('upstream_repository', '')
        note = 'Existing causal adapter anchor; subsequent local compatibility edits are identified by executed snapshot hash.'
    elif model == 'MMIN':
        source, adapter = value.get('official_commit', ''), value.get('gcnet_commit', '')
        repository = value.get('official_repository', '')
        note = value.get('implementation_kind', '')
    elif model == 'GCNet':
        official = value.get('requested_official_reference_verification', {})
        source = official.get('verified_head_commit', '')
        adapter = value.get('commit', '')
        repository = official.get('repository', value.get('repository', ''))
        verified = official.get('files', {})
        note = ('Official/local original network, graph and loss files verified byte-identical; '
                'MELD compatibility edits are identified by executed snapshot hash.'
                if len(verified) == 4 and all(v.get('bytes_equal') for v in verified.values())
                else 'Official/local byte equivalence has not been established by this provenance record.')
    elif model == 'SDR-GNN':
        source = value.get('upstream_commit', value.get('commit', ''))
        repository = value.get('repository_url', value.get('repository', ''))
        note = ('Released-code-compatible: retain tuple graph selection, attention axes and upstream reconstruction loss; only input-width/speaker compatibility and engineering imports. '
                if config.get('model_version') == 'released_compatible' else 'Implementation-fixed local adapter; ')
        note += 'No separate committed revision; use executed snapshot hash.'
    elif model == 'MoMKE':
        module = base/'adapters/momke/__init__.py'
        if module.exists():
            for node in ast.parse(module.read_text()).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'UPSTREAM_COMMIT' for t in node.targets):
                    source = ast.literal_eval(node.value)
        repository = 'https://github.com/wxxv/MoMKE' if source else ''
        note = 'Local causal/random-missingness adapter has no separate committed revision; use executed snapshot hash.'
    return dict(source_repository=repository, source_commit=source, adapter_commit=adapter,
                source_provenance_scope=origin if source else 'UNAVAILABLE', source_provenance_note=note)


def _pid_alive(pid, run_dir=None):
    try:
        os.kill(int(pid), 0)
        # A zombie cannot perform training, even though kill(pid, 0) succeeds.
        stat = Path('/proc/{}/stat'.format(pid))
        if stat.exists() and stat.read_text().split(') ', 1)[1].split()[0] == 'Z':
            return False
        if run_dir is not None:
            return str(run_dir) in Path('/proc/{}/cmdline'.format(pid)).read_bytes().replace(b'\0', b' ').decode()
        return True
    except (OSError, TypeError, ValueError, IndexError):
        return False


def _expected(config, datasets, cache):
    key = (config['dataset'], config['fold'], config['missing_rate'], config['eval_mask_seed'])
    if key not in cache:
        if config['dataset'] not in datasets:
            datasets[config['dataset']] = load_dataset(config['dataset'])
        records = datasets[config['dataset']]['records']
        records = [r for r in records if (r['split'] == 'test' if config['dataset'] == 'MELD'
                                         else r['session'] == config['fold'])]
        expected = {}
        for record in records:
            masks = fixed_masks(config['dataset'], record['id'], record['utterance_ids'],
                                config['missing_rate'], config['eval_mask_seed'])
            for uid, label, mask in zip(record['utterance_ids'], record['labels'], masks):
                sample_id = record['id'] + '::' + str(uid)
                if sample_id in expected:
                    raise ValueError('duplicate expected test sample id')
                expected[sample_id] = (int(label), mask)
        cache[key] = expected
    return cache[key]


def validate_formal(task, datasets, expected_cache):
    """Return actual arrays only after artifact, config, sample, label and mask checks."""
    run = Path(task['run_dir'])
    if task.get('phase') != 'formal':
        return None, 'NON_FORMAL_PHASE'
    if task.get('status') == 'LEGACY':
        return None, 'LEGACY_PROTOCOL'
    if not valid_result(run):
        return None, 'INVALID_OR_INCOMPLETE_ARTIFACTS'
    manifest = read_json(run / 'manifest.json')
    if manifest.get('result_type') != 'formal':
        return None, 'NON_FORMAL_RESULT_TYPE'
    config = task['config']
    recorded = manifest.get('config', {})
    if (read_json(run / 'config.json') != config
            or any(recorded.get(k) != v for k, v in config.items())
            or set(recorded) - set(config) - {'device', 'seq_len'}):
        return None, 'CONFIG_MISMATCH'
    if config.get('seed') not in SEEDS:
        return None, 'UNPLANNED_SEED'
    if config.get('checkpoint_selection') != 'test_best_acc' or config.get('evaluation_protocol') != 'test_oracle':
        return None, 'LEGACY_PROTOCOL'
    expected = _expected(config, datasets, expected_cache)
    with np.load(run / 'predictions.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in ('labels', 'predictions', 'masks', 'sample_ids')}
    sample_ids = arrays['sample_ids'].tolist()
    if len(sample_ids) != len(expected) or set(sample_ids) != set(expected):
        return None, 'TEST_SAMPLE_COVERAGE_MISMATCH'
    for i, uid in enumerate(sample_ids):
        label, mask = expected[uid]
        if arrays['labels'][i] != label:
            return None, 'TEST_LABEL_MISMATCH'
        if not np.array_equal(arrays['masks'][i], mask):
            return None, 'FIXED_TEST_MASK_MISMATCH'
    return arrays, ''


def _array_signature(arrays):
    order = np.argsort(arrays['sample_ids'])
    digest = hashlib.sha256()
    digest.update(json.dumps(arrays['sample_ids'][order].tolist(), separators=(',', ':')).encode())
    for name in ('labels', 'masks'):
        digest.update(np.ascontiguousarray(arrays[name][order]).tobytes())
    return digest.hexdigest()


def aggregate(registry, output_dir=None):
    registry = Path(registry)
    output = Path(output_dir) if output_dir is not None else registry.parent
    output.mkdir(parents=True, exist_ok=True)
    tasks = read_json(registry)['tasks']
    audit_data = read_json(registry.parent/'audit/data_audit.json').get('datasets', {})
    datasets, expected_cache = {}, {}
    progress, folds, seeds_out, summaries, comparisons, table_c = [], [], [], [], [], []
    planned = defaultdict(list)
    accepted = defaultdict(dict)
    for task in tasks:
        c = task['config']; run = Path(task['run_dir'])
        status = read_json(run / 'status.json')
        supervisor = read_json(run/'supervisor.json')
        child = read_json(run/'child.json')
        exit_state = read_json(run/'exit.json')
        pid = child.get('pid', status.get('pid', supervisor.get('pid')))
        state = status.get('state', task.get('status', 'MISSING'))
        alive = _pid_alive(pid, run)
        if alive:
            state = 'RUNNING'
        elif exit_state.get('exit_code', 0) != 0:
            state = 'FAILED'
        if state == 'RUNNING' and not alive:
            state = 'INTERRUPTED'
        arrays, rejection = validate_formal(task, datasets, expected_cache)
        if status.get('state') == 'COMPLETED' and arrays is None and task.get('phase') == 'formal':
            state = 'INVALID_COMPLETED'
        elif arrays is not None:
            state = 'COMPLETED'
        base = {k: c[k] for k in KEYS}
        progress.append(dict(task_id=task['id'], **base, seed=c['seed'], fold=c['fold'],
                             phase=task.get('phase'), status=state, pid=pid or '',
                             pid_alive=alive, gpu_uuid=status.get('gpu_uuid', supervisor.get('gpu_uuid', '')),
                             epoch=status.get('epoch', ''), exit_code=exit_state.get('exit_code', status.get('exit_code', '')),
                             run_dir=str(run), log_path=str(run / 'run.log'), rejection_reason=rejection))
        if task.get('phase') != 'formal':
            continue
        group = tuple(c[k] for k in KEYS)
        planned[group].append(task)
        manifest = read_json(run/'manifest.json')
        data_meta = manifest.get('data', audit_data.get(c['dataset'], {}).get('metadata', {}))
        code_snapshot = manifest.get('code_snapshot', '')
        provenance = read_json(Path(code_snapshot)/'snapshot.json') if code_snapshot else {}
        source = _source_provenance(c, registry.parent, code_snapshot)
        table_c.append(dict(**c, task_id=task['id'], model_label=model_label(c['model'], c['model_version'], c['dataset']),
                            causality=task.get('causality', ''), feature_version=data_meta.get('selected_features_labels_sha256', ''),
                            feature_provenance_scope=('run_manifest' if manifest.get('data') else
                                                      'verified_local_data_audit' if data_meta else 'UNAVAILABLE'),
                            git_commit=source['source_commit'], **source,
                            configuration_scope='executed_run' if manifest else 'planned_unstarted',
                            code_snapshot_sha256=provenance.get('sha256', ''), evaluation_bias=BIAS, status=state))
        if arrays is None:
            continue
        seedkey = group + (c['seed'],)
        if c['fold'] in accepted[seedkey]:
            raise ValueError('duplicate formal seed/fold in registry: {}'.format(seedkey))
        accepted[seedkey][c['fold']] = arrays
        metrics = compute_metrics(arrays['labels'], arrays['predictions'], c['n_classes'])
        saved_metrics = read_json(run / 'metrics.json')
        folds.append(dict(**base, seed=c['seed'], fold=c['fold'], **metrics,
                          n_test=len(arrays['labels']), best_epoch=saved_metrics['best_epoch'],
                          actual_missing_rate=float(1 - arrays['masks'].mean()), run_dir=str(run),
                          checkpoint_selection='test_best_acc', evaluation_protocol='test_oracle'))
    seed_arrays = {}
    for group, group_tasks in sorted(planned.items()):
        base = dict(zip(KEYS, group)); dataset = base['dataset']
        required_folds = {0} if dataset == 'MELD' else set(range(1, 6))
        protocols = {json.dumps({k: v for k, v in t['config'].items() if k not in ('seed', 'fold')}, sort_keys=True)
                     for t in group_tasks}
        compatible = len(protocols) == 1
        complete_seeds = []
        n_classes = group_tasks[0]['config']['n_classes']
        for seed in SEEDS:
            present = accepted.get(group + (seed,), {})
            row = dict(**base, seed=seed, fold_count=len(present),
                       missing_folds=','.join(map(str, sorted(required_folds - set(present)))),
                       status=('INCOMPATIBLE_CONFIGS' if not compatible else
                               'COMPLETE' if set(present) == required_folds else 'INCOMPLETE_FOLDS'),
                       checkpoint_selection='test_best_acc', evaluation_protocol='test_oracle')
            if compatible and set(present) == required_folds:
                pooled = {k: np.concatenate([present[f][k] for f in sorted(required_folds)]) for k in ('labels', 'predictions', 'masks', 'sample_ids')}
                if len(set(pooled['sample_ids'].tolist())) != len(pooled['labels']):
                    raise ValueError('test sample repeated across folds')
                row.update(compute_metrics(pooled['labels'], pooled['predictions'], n_classes))
                row.update(n_test=len(pooled['labels']), actual_missing_rate=float(1 - pooled['masks'].mean()),
                           sample_mask_signature=_array_signature(pooled))
                seed_arrays[group + (seed,)] = (row, pooled)
                complete_seeds.append(row)
            seeds_out.append(row)
        summary = dict(**base, model_label=model_label(base['model'], base['model_version'], dataset),
                       count=len(complete_seeds), missing_seeds=','.join(str(s) for s in SEEDS if not any(r['seed'] == s for r in complete_seeds)),
                       status=('INCOMPATIBLE_CONFIGS' if not compatible else
                               'COMPLETE_3SEED' if len(complete_seeds) == 3 else 'INCOMPLETE_3SEED'),
                       checkpoint_selection='test_best_acc', evaluation_protocol='test_oracle')
        for metric in METRICS:
            values = [r[metric] for r in complete_seeds]
            summary[metric + '_mean'] = float(np.mean(values)) if values else ''
            summary[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else ''
            summary[metric + '_variance'] = float(np.var(values, ddof=1)) if len(values) > 1 else ''
        summaries.append(summary)
    for group in sorted(planned):
        dataset, model, version, mr = group
        if model not in ('MoMKE', 'ComP') or version != 'causal':
            continue
        for seed in SEEDS:
            causal = seed_arrays.get(group + (seed,))
            original = seed_arrays.get((dataset, model, 'original', mr, seed))
            row = dict(dataset=dataset, model=model, missing_rate=mr, seed=seed,
                       comparison='causal_minus_original', status='MISSING_BOTH', evaluation_protocol='test_oracle')
            if causal is not None and original is None:
                row['status'] = 'MISSING_ORIGINAL'
            elif causal is None and original is not None:
                row['status'] = 'MISSING_CAUSAL'
            elif causal is not None and original is not None:
                if causal[0]['sample_mask_signature'] != original[0]['sample_mask_signature']:
                    row['status'] = 'INCOMPATIBLE_SAMPLE_MASK'
                else:
                    row.update(status='PAIRED', **{m + '_delta': causal[0][m] - original[0][m] for m in METRICS})
            comparisons.append(row)
    sdr_comparisons = []
    for dataset, model, version, mr in sorted(planned):
        if model != 'SDR-GNN' or version != 'released_compatible':
            continue
        for seed in SEEDS:
            released = seed_arrays.get((dataset, model, version, mr, seed))
            fixed = seed_arrays.get((dataset, model, 'implementation_fixed', mr, seed))
            row = dict(dataset=dataset, model=model, missing_rate=mr, seed=seed,
                       comparison='implementation_fixed_minus_released_compatible',
                       status='MISSING_BOTH', evaluation_protocol='test_oracle')
            if released is None and fixed is not None:
                row['status'] = 'MISSING_RELEASED_COMPATIBLE'
            elif released is not None and fixed is None:
                row['status'] = 'MISSING_IMPLEMENTATION_FIXED'
            elif released is not None and fixed is not None:
                if released[0]['sample_mask_signature'] != fixed[0]['sample_mask_signature']:
                    row['status'] = 'INCOMPATIBLE_SAMPLE_MASK'
                else:
                    row.update(status='PAIRED', **{m + '_delta': fixed[0][m] - released[0][m] for m in METRICS})
            sdr_comparisons.append(row)
    write_csv(output/'sdr_version_comparison.csv', sdr_comparisons, ['dataset', 'model', 'missing_rate', 'seed', 'comparison', 'status', *[m + '_delta' for m in METRICS], 'evaluation_protocol'])
    write_csv(output/'progress.csv', progress, ['task_id', *KEYS, 'seed', 'fold', 'phase', 'status', 'pid', 'pid_alive', 'gpu_uuid', 'epoch', 'exit_code', 'run_dir', 'log_path', 'rejection_reason'])
    write_csv(output/'results.csv', folds, [*KEYS, 'seed', 'fold', *METRICS, 'n_test', 'best_epoch', 'actual_missing_rate', 'run_dir', 'checkpoint_selection', 'evaluation_protocol'])
    write_csv(output/'results_seed.csv', seeds_out, [*KEYS, 'seed', 'fold_count', 'missing_folds', 'status', *METRICS, 'n_test', 'actual_missing_rate', 'sample_mask_signature', 'checkpoint_selection', 'evaluation_protocol'])
    write_csv(output/'results_summary.csv', summaries, [*KEYS, 'model_label', 'count', 'missing_seeds', 'status', *[m + '_' + s for m in METRICS for s in ('mean', 'std', 'variance')], 'checkpoint_selection', 'evaluation_protocol'])
    write_csv(output/'comparison.csv', comparisons, ['dataset', 'model', 'missing_rate', 'seed', 'comparison', 'status', *[m + '_delta' for m in METRICS], 'evaluation_protocol'])
    c_fields = sorted({k for row in table_c for k in row})
    write_csv(output/'TABLE_C.csv', table_c, c_fields)
    _write_tables(output, summaries, planned)
    return {'tasks': len(progress), 'verified_formal_folds': len(folds),
            'complete_seed_results': sum(r['status'] == 'COMPLETE' for r in seeds_out),
            'complete_3seed_results': sum(r['status'] == 'COMPLETE_3SEED' for r in summaries)}


def _write_tables(output, summaries, planned):
    for letter in ('A', 'B', 'D'):
        lines = ['**' + BIAS + '**', '', '| Dataset | Model | Causality | MR | ACC | WF1 | UA | Macro F1 | Seeds | Status |',
                 '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
        for row in summaries:
            appendix_model = row['model'] in ('MoMKE', 'ComP')
            if letter == 'A' and row['model'] == 'SDR-GNN' and row['model_version'] == 'implementation_fixed':
                continue
            if letter == 'A' and appendix_model and row['model_version'] == 'original':
                continue
            if letter == 'B' and not appendix_model:
                continue
            if letter == 'D' and row['model'] != 'SDR-GNN':
                continue
            task = planned[tuple(row[k] for k in KEYS)][0]
            scores = ['{:.2f} ± {:.2f}'.format(row[m+'_mean'], row[m+'_std'])
                      if row['status'] == 'COMPLETE_3SEED' else '—' for m in METRICS]
            lines.append('| {} | {} | {} | {:.1f} | {} | {} | {} | {} | {}/3 | {} |'.format(
                row['dataset'], row['model_label'], task.get('causality', ''), row['missing_rate'],
                *scores, row['count'], row['status']))
        if letter == 'A':
            for dataset, mr in sorted({(r['dataset'], r['missing_rate']) for r in summaries}):
                lines.append('| {} | HAMR existing matched results | Causal | {:.1f} | — | — | — | — | 0/3 | NO_MATCHED_RESULT_IMPORTED |'.format(dataset, mr))
        (output / ('TABLE_' + letter + '.md')).write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=BIAS)
    parser.add_argument('--registry', type=Path, default=ROOT/'experiment_registry.json')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.registry, args.output_dir), sort_keys=True))


if __name__ == '__main__':
    main()
