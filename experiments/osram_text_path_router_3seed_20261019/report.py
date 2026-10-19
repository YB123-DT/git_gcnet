"""Three-seed paired summaries, reusing the immutable seed66 results."""
import csv
import json
from pathlib import Path

import numpy as np

SEED66_ROOT = Path('/home/qinjinghui/remote_experiments/osram_text_path_router_joint_20261019/execution_v1/runs')
CORE = ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/tasr.py', 'gcnet_missing_m3/model.py',
        'gcnet_missing_m3/train_gcnet.py', 'gcnet_modality_jepa/gcnet_legacy_random_mask.py',
        'experiments/osram_text_path_router_20261019/router.py',
        'experiments/osram_text_path_router_joint_20261019/joint.py')


def write_report(output, queue):
    output = Path(output)
    baseline = Path(queue.get('seed66_root', SEED66_ROOT))
    runs, checks, rows, grouped = {}, {}, [], {}
    lines = ['# One-stage Text-source Router: seeds66/67/68', '',
             'INTERNAL EXPLORATORY EXPERIMENT', '',
             'Seed66 reused without retraining. Seed67/68 add matched fresh Flat and Joint jobs.',
             'MELD official split; IEMOCAP-6 fold5 only. Fixed split and evaluation masks across seeds.',
             'One validation-selected checkpoint per job, not per-rate Test-best.', '',
             '| Dataset | Seed | Model | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for dataset in ('MELD', 'IEMOCAPSix'):
        runs[dataset], checks[dataset] = {}, {}
        base_core = None
        for seed in (66, 67, 68):
            root = baseline / dataset if seed == 66 else output / 'runs' / dataset / f'seed_{seed}'
            pair, history = {}, {}
            for mode in ('flat', 'joint'):
                summary = json.loads((root / mode / 'SUMMARY.json').read_text())
                if summary.get('status') != 'complete' or summary.get('phase') != 'train':
                    raise ValueError(f'Incomplete run: {dataset}/{seed}/{mode}')
                if summary['seed'] != seed or summary['identity']['config']['seed'] != seed:
                    raise ValueError('Run seed does not match config/manifest')
                core = {key:summary['identity']['source_hashes'][key] for key in CORE}
                if base_core is None:
                    base_core = core
                if core != base_core:
                    raise ValueError('Numerical model/Router/sampler code changed across runs')
                pair[mode] = summary
                history[mode] = json.loads((root / mode / 'history.json').read_text())
                result = summary['aggregate']['router']
                values = [result[scope][metric] for scope in ('mean', 'high') for metric in ('ACC', 'UA', 'WF1')]
                lines.append(f'| {dataset} | {seed} | {mode} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
                for rate, metric in summary['evaluation'].items():
                    rows.append(dict(dataset=dataset, seed=seed, mode=mode, rate=rate, **metric['router']))
            if pair['flat']['initial_backbone_sha256'] != pair['joint']['initial_backbone_sha256']:
                raise ValueError('Paired initializations differ')
            if pair['flat']['validation_mask_hashes'] != pair['joint']['validation_mask_hashes']:
                raise ValueError('Paired validation masks differ')
            if len(history['flat']) != len(history['joint']) or not all(
                    a['mask_trace_sha256'] == b['mask_trace_sha256']
                    for a,b in zip(history['flat'],history['joint'])):
                raise ValueError('Paired training budgets/mask traces differ')
            for rate in pair['flat']['evaluation']:
                name = 'random_' + rate.replace('.', 'p') + '.npz'
                with np.load(root/'flat'/'evaluation'/name,allow_pickle=False) as a, \
                        np.load(root/'joint'/'evaluation'/name,allow_pickle=False) as b:
                    for key in ('labels','speakers','conversation_ids','utterance_indices','availability'):
                        np.testing.assert_array_equal(a[key],b[key])
            checks[dataset][str(seed)] = dict(core_source_equal=True, initialization_exact=True,
                validation_masks_exact=True, training_masks_exact=True, test_cohort_masks_exact=True)
            runs[dataset][str(seed)] = pair
        grouped[dataset] = {}
        for mode in ('flat','joint'):
            grouped[dataset][mode] = {}
            for scope in ('mean','high'):
                grouped[dataset][mode][scope] = {
                    metric:dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)))
                    for metric in ('ACC','UA','WF1','CE')
                    for values in [[runs[dataset][str(seed)][mode]['aggregate']['router'][scope][metric]
                                    for seed in (66,67,68)]]}
    lines += ['', '## Three-seed mean ± sample standard deviation', '',
              '| Dataset | Model | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset, modes in grouped.items():
        for mode, scopes in modes.items():
            cells = [f"{scopes[s][m]['mean']:.4f} ± {scopes[s][m]['std']:.4f}"
                     for s in ('mean','high') for m in ('ACC','UA','WF1')]
            lines.append(f'| {dataset} | {mode} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'No independent-test or multi-fold claim. Do not pool eight masks as eight independent seeds.',
              'No additional methods, hyperparameter search or automatic expansion are authorized.']
    (output/'RESULT.md').write_text('\n'.join(lines).rstrip()+'\n')
    (output/'SUMMARY.json').write_text(json.dumps(dict(runs=runs,checks=checks,aggregate=grouped),
                                                 indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (output/'METRICS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
