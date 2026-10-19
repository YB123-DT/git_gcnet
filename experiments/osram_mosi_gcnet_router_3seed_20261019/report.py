"""Complete-only paired GCNet-mask reports; never splice historical Test-best."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
CORE = ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/tasr.py', 'gcnet_missing_m3/model.py',
        'gcnet_missing_m3/train_gcnet.py', 'gcnet_modality_jepa/gcnet_legacy_random_mask.py',
        'experiments/osram_text_path_router_20261019/router.py',
        'experiments/osram_text_path_router_joint_20261019/joint.py')


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def masks(summary):
    identity = summary['identity']
    if 'mask_hashes' in summary:
        return summary['mask_hashes']
    return dict(validation=summary.get('validation_mask_hashes', identity.get('validation_mask_hashes')),
                test=identity.get('test_masks'))


def validate_pair(flat, joint, flat_history, joint_history):
    if flat['initial_backbone_sha256'] != joint['initial_backbone_sha256']:
        raise ValueError('Paired initial backbone hashes differ')
    if flat['identity']['config'] != joint['identity']['config']:
        raise ValueError('Paired training config differs')
    if flat['identity']['split'] != joint['identity']['split']:
        raise ValueError('Paired data split differs')
    if masks(flat) != masks(joint):
        raise ValueError('Paired evaluation masks differ')
    if len(flat_history) != len(joint_history) or not flat_history:
        raise ValueError('Missing or unequal training histories')
    if any(a['mask_trace_sha256'] != b['mask_trace_sha256'] for a, b in zip(flat_history, joint_history)):
        raise ValueError('Actual paired training masks differ')
    if any(flat['identity']['source_hashes'][name] != joint['identity']['source_hashes'][name] for name in CORE):
        raise ValueError('Paired numerical model/sampler code differs')
    return dict(initialization_exact=True, config_exact=True, split_exact=True,
                training_masks_exact=True, validation_masks_exact=True, test_mask_hashes_exact=True,
                core_sources_exact=True)


def load_pair(paths, *, dataset, seed, fold=None, epochs=150):
    import numpy as np

    pair, history, checks = {}, {}, {}
    for mode, path in paths.items():
        path = Path(path)
        summary = json.loads((path/'SUMMARY.json').read_text())
        if summary.get('status') != 'complete' or summary.get('phase') != 'train':
            raise ValueError('Incomplete run: ' + str(path))
        if (summary.get('dataset') != dataset or summary.get('seed') != seed or summary.get('mode') != mode
                or summary['identity']['config']['seed'] != seed):
            raise ValueError('Manifest run identity differs: ' + str(path))
        cfg = summary['identity']['config']
        if cfg.get('missing_mask_protocol') != 'gcnet-random-mask-v1':
            raise ValueError('Wrong missing protocol; do not mix old MOSI results')
        if cfg.get('epochs') != epochs or summary.get('epoch') != epochs:
            raise ValueError('Run has not finished the approved complete budget')
        if fold is not None and cfg.get('fold') != fold:
            raise ValueError('Fold identity differs')
        if set(summary.get('evaluation', {})) != set(RATES):
            raise ValueError('Must report all eight rates')
        if set(masks(summary)['validation']) != set(RATES) or set(masks(summary)['test']) != set(RATES):
            raise ValueError('Missing fixed evaluation mask hashes')
        for metric in ('ACC', 'WF1') + (('MSE',) if dataset == 'CMUMOSI' else ('UA', 'CE')):
            values = [summary['evaluation'][rate]['router'][metric] for rate in RATES]
            if not all(math.isfinite(value) for value in values):
                raise ValueError('Nonfinite reported metrics')
            for scope, selected in (('mean', values), ('high', values[5:])):
                if not math.isclose(summary['aggregate']['router'][scope][metric], statistics.mean(selected),
                                    rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError('Stored aggregate does not match per-rate metrics')
        pair[mode] = summary
        history[mode] = json.loads((path/'history.json').read_text())
        if len(history[mode]) != epochs:
            raise ValueError('Incomplete training history')
    checks.update(validate_pair(pair['flat'], pair['joint'], history['flat'], history['joint']))
    checks['test_cohort_and_masks_exact'] = {}
    for rate in RATES:
        arrays = {}
        for mode, path in paths.items():
            artifact = Path(path)/'evaluation'/('random_' + rate.replace('.', 'p') + '.npz')
            expected = pair[mode]['evaluation'][rate].get('output_sha256')
            if expected and hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                raise ValueError('Prediction artifact hash differs: ' + str(artifact))
            with np.load(artifact, allow_pickle=False) as saved:
                arrays[mode] = {key: saved[key] for key in
                                ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability')}
        for key in arrays['flat']:
            np.testing.assert_array_equal(arrays['flat'][key], arrays['joint'][key])
        checks['test_cohort_and_masks_exact'][rate] = True
    return pair, checks


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values))


def report_outputs(output, data, rows, lines):
    output = Path(output)
    write(output/'SUMMARY.json', data)
    (output/'RESULT.md').write_text('\n'.join(lines).rstrip() + '\n')
    with (output/'METRICS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, queue):
    output = Path(output)
    if len(queue['jobs']) != 6 or any(j['status'] != 'complete' for j in queue['jobs']):
        raise ValueError('Require exactly six completed MOSI jobs')
    runs, checks, rows = {}, {}, []
    core, reference_masks = None, None
    lines = ['# MOSI GCNet Random Missing: Flat vs one-stage Router', '',
             'INTERNAL EXPLORATORY EXPERIMENT', '',
             'Three seeds, six fresh runs, 150 epochs. Original GCNet random-mask-v1.',
             'One validation-selected checkpoint per run; not per-rate Test-best.',
             'Historical 80.747 used conversation-mask-v2 and is not a paired baseline.', '',
             '| Seed | Model | Selected epoch | ACC | W-F1 | High ACC | High W-F1 |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for seed in (66, 67, 68):
        selected = [j for j in queue['jobs'] if j['seed'] == seed]
        if {j['mode'] for j in selected} != {'flat', 'joint'} or len(selected) != 2:
            raise ValueError('Invalid MOSI seed/model plan')
        pair, check = load_pair({j['mode']: j['output'] for j in selected}, dataset='CMUMOSI', seed=seed)
        new_core = {name: pair['flat']['identity']['source_hashes'][name] for name in CORE}
        if core is not None and new_core != core:
            raise ValueError('Numerical core changed between MOSI seeds')
        core = new_core
        cohort = dict(masks=masks(pair['flat']), split=pair['flat']['identity']['split'])
        if reference_masks is not None and cohort != reference_masks:
            raise ValueError('MOSI evaluation masks/split changed across seeds')
        reference_masks = cohort
        checks[str(seed)], runs[str(seed)] = check, pair
        for mode, summary in pair.items():
            values = [summary['aggregate']['router'][scope][metric] for scope in ('mean', 'high') for metric in ('ACC', 'WF1')]
            lines.append(f'| {seed} | {mode} | {summary["selected_epoch"]} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
            for rate in RATES:
                rows.append(dict(dataset='CMUMOSI', seed=seed, mode=mode, rate=rate,
                                 selected_epoch=summary['selected_epoch'], **summary['evaluation'][rate]['router']))
    aggregate = {mode: {scope: {metric: stats([runs[str(seed)][mode]['aggregate']['router'][scope][metric]
                                            for seed in (66, 67, 68)])
                               for metric in ('ACC', 'WF1', 'MSE')} for scope in ('mean', 'high')}
                 for mode in ('flat', 'joint')}
    lines += ['', '## Three-seed mean ± sample standard deviation', '',
              '| Model | ACC | W-F1 | High ACC | High W-F1 |', '|---|---:|---:|---:|---:|']
    for mode, result in aggregate.items():
        cells = [f'{result[scope][metric]["mean"]:.4f} ± {result[scope][metric]["std"]:.4f}'
                 for scope in ('mean', 'high') for metric in ('ACC', 'WF1')]
        lines.append('| ' + mode + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Per-rate three-seed means', '',
              '| Rate | Flat ACC | Joint ACC | Flat W-F1 | Joint W-F1 |', '|---|---:|---:|---:|---:|']
    per_rate = {}
    for rate in RATES:
        per_rate[rate] = {mode: {metric: stats([runs[str(seed)][mode]['evaluation'][rate]['router'][metric]
                                              for seed in (66, 67, 68)]) for metric in ('ACC', 'WF1', 'MSE')}
                          for mode in ('flat', 'joint')}
        values = [per_rate[rate][mode][metric]['mean'] for metric in ('ACC', 'WF1') for mode in ('flat', 'joint')]
        lines.append('| ' + rate + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', 'No independent-test paper claim. No automatic method, seed or HPO expansion.',
              'Joint-vs-Flat is the full training-method comparison; Joint full candidate is not a separately trained Flat.']
    report_outputs(output, dict(runs=runs, checks=checks, aggregate=aggregate, per_rate=per_rate,
                               mask_protocol='gcnet-random-mask-v1'), rows, lines)
