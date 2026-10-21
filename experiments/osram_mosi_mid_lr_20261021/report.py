"""Complete-only LR comparison using exact saved per-rate selected predictions."""
import json
import math
from pathlib import Path

import numpy as np

from experiments.osram_mosi_gcnet_router_3seed_20261019.report import RATES, report_outputs
from experiments.osram_mosi_gcnet_testbest_20261020.report import validate_policy
from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
from experiments.osram_mosi_gcnet_router_3seed_20261019.run import validate_summary
from experiments.osram_mosi_router_lr1e5_20261020.worker import REFERENCE


def write_report(output, queue):
    jobs = queue['jobs']
    if (len(jobs) != 2 or {j['backbone_lr'] for j in jobs} != {5e-4, 3e-4}
            or any(j['status'] != 'complete' or j['seed'] != 66 or j['mode'] != 'flat' for j in jobs)):
        raise ValueError('Require exactly two completed original Flat seed66 runs')
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_text_path_router_joint_mosi_20261019.run import summarize_arrays
    original_path = REFERENCE/'flat'
    original = json.loads((original_path/'SUMMARY.json').read_text())
    if original['status'] != 'complete' or original['epoch'] != 150:
        raise ValueError('Original same-protocol reference is incomplete')
    runs = {'original_1e-3': original}
    paths = {'original_1e-3': original_path}
    for job in jobs:
        s = validate_summary(job)
        if s['epoch'] != 150 or s['router_parameters'] != 0:
            raise ValueError('Require150epoch Flat, not Router')
        if s['identity']['backbone_learning_rates'] != dict(first_100=job['backbone_lr'], final_50=job['backbone_lr']/10):
            raise ValueError('Wrong LR schedule')
        label = f"flat_{job['backbone_lr']:g}"
        runs[label], paths[label] = s, Path(job['output'])
    rows = []
    for label, s in runs.items():
        validate_policy(s)
        validate_selected(paths[label], s['best_by_rate'])
        history = json.loads((paths[label]/'history.json').read_text())
        if len(history) != 150:
            raise ValueError('Incomplete training history')
        if s['mask_hashes'] != original['mask_hashes'] or s['initial_backbone_sha256'] != original['initial_backbone_sha256']:
            raise ValueError('Reference masks/initialization differ')
        actual_evaluation = {}
        for rate in RATES:
            scores = [epoch['test'][rate]['router']['WF1'] for epoch in history]
            if s['selected_epoch_by_rate'][rate] != scores.index(max(scores)) + 1:
                raise ValueError('Not earliest per-rate TEST W-F1 best')
            artifact = paths[label]/'evaluation'/('random_'+rate.replace('.', 'p')+'.npz')
            with np.load(artifact, allow_pickle=False) as saved:
                arrays = {k: saved[k] for k in saved.files}
            if label != 'original_1e-3':
                with np.load(original_path/'evaluation'/artifact.name, allow_pickle=False) as baseline:
                    for key in ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability'):
                        np.testing.assert_array_equal(arrays[key], baseline[key])
            actual = summarize_arrays(arrays, trainer)['router']
            for metric in ('ACC', 'WF1', 'MSE'):
                if not math.isclose(actual[metric], s['evaluation'][rate]['router'][metric], abs_tol=1e-8):
                    raise ValueError('Stored metric differs from selected predictions')
            actual_evaluation[rate] = {'router': actual}
            rows.append(dict(model=label, seed=66, rate=rate, selected_epoch=s['selected_epoch_by_rate'][rate], **actual))
        for scope in ('mean', 'high'):
            for metric in ('ACC', 'WF1', 'MSE'):
                rates = RATES if scope == 'mean' else RATES[5:]
                recomputed = sum(actual_evaluation[r]['router'][metric] for r in rates) / len(rates)
                if not math.isclose(recomputed, s['aggregate']['router'][scope][metric], abs_tol=1e-8):
                    raise ValueError('Aggregate does not match selected predictions')
    lines = ['# MOSI original Flat intermediate LR screen', '', 'INTERNAL TEST-ORACLE SCREEN', '',
             'Seed66,150epochs,original GCNet missing; no Router/new block. Fresh initialization.',
             'LR decays10x at epoch101. Each rate selects its own earliest TEST-best W-F1 epoch.',
             'The eight-rate mean combines different checkpoints; not a single deployment score.', '',
             '| Model | Mean ACC | Mean W-F1 | High ACC | High W-F1 | ΔMean W-F1(pp) |',
             '|---|---:|---:|---:|---:|---:|']
    for label, s in runs.items():
        values = [s['aggregate']['router'][scope][metric] for scope in ('mean', 'high') for metric in ('ACC', 'WF1')]
        delta = s['aggregate']['router']['mean']['WF1'] - original['aggregate']['router']['mean']['WF1']
        lines.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in values)+f' | {delta:+.4f} |')
    lines += ['', '| Model | Rate | Selected epoch | ACC | W-F1 |', '|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['model']} | {r['rate']} | {r['selected_epoch']} | {r['ACC']:.4f} | {r['WF1']:.4f} |" for r in rows]
    report_outputs(output, dict(status='complete', runs=runs, reference_path=str(original_path),
                   label='INTERNAL TEST-ORACLE SCREEN'), rows, lines)
