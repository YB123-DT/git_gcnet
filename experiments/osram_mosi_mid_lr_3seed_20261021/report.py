"""Paired three-seed results; reuse66 and never mix completed/partial runs."""
import json
import math
from pathlib import Path
import statistics

import numpy as np

from experiments.osram_mosi_gcnet_router_3seed_20261019.report import RATES, report_outputs, CORE
from experiments.osram_mosi_gcnet_router_3seed_20261019.run import validate_summary
from experiments.osram_mosi_gcnet_testbest_20261020.report import validate_policy
from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
from experiments.osram_mosi_mid_lr_20261021.worker import validate_reference
from experiments.osram_mosi_router_lr1e5_20261020.worker import REFERENCE, check_mask_parity

SEED66 = Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003')


def write_report(output, queue):
    jobs = queue['jobs']
    if (len(jobs) != 2 or {j['seed'] for j in jobs} != {67, 68}
            or any(j['status'] != 'complete' or j['mode'] != 'flat' for j in jobs)):
        raise ValueError('Require both completed seed67/68 original Flat runs')
    for job in jobs:
        validate_summary(job)
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_text_path_router_joint_mosi_20261019.run import summarize_arrays
    paths = {66: SEED66, **{j['seed']: Path(j['output']) for j in jobs}}
    runs, rows = {}, []
    shared_masks, shared_core = None, None
    for seed, path in sorted(paths.items()):
        s = json.loads((path/'SUMMARY.json').read_text())
        original_path = REFERENCE.parent/f'seed_{seed}'/'flat'
        original = json.loads((original_path/'SUMMARY.json').read_text())
        if (s['status'] != 'complete' or s['epoch'] != 150 or s['seed'] != seed
                or s['mode'] != 'flat' or s['router_parameters'] != 0
                or s['identity']['backbone_learning_rates'] != dict(first_100=3e-4, final_50=3e-4 / 10)):
            raise ValueError('Wrong or incomplete seed/LR/architecture')
        validate_reference(s['identity'], original, s['initial_backbone_sha256'], backbone_lr=3e-4)
        actual_core = {name: s['identity']['source_hashes'][name] for name in CORE}
        if shared_core is not None and actual_core != shared_core:
            raise ValueError('Numerical core differs across seeds')
        if shared_masks is not None and s['mask_hashes'] != shared_masks:
            raise ValueError('Common evaluation bank differs across seeds')
        shared_core, shared_masks = actual_core, s['mask_hashes']
        old_history = json.loads((original_path/'history.json').read_text())
        history = json.loads((path/'history.json').read_text())
        if len(history) != 150 or len(old_history) != 150:
            raise ValueError('Incomplete history')
        for index, row in enumerate(history):
            check_mask_parity(row, old_history[index])
            if row['learning_rates'] != [3e-4 if index < 100 else 3e-4 / 10]:
                raise ValueError('Actual LR differs from approved100+50 schedule')
        for name, summary, folder in (('original', original, original_path), ('lr3e4', s, path)):
            validate_policy(summary)
            validate_selected(folder, summary['best_by_rate'])
            train_history = old_history if name == 'original' else history
            measured = {}
            for rate in RATES:
                scores = [epoch['test'][rate]['router']['WF1'] for epoch in train_history]
                if summary['selected_epoch_by_rate'][rate] != scores.index(max(scores)) + 1:
                    raise ValueError('Not earliest per-rate Test-best W-F1')
                artifact = folder/'evaluation'/('random_'+rate.replace('.', 'p')+'.npz')
                with np.load(artifact, allow_pickle=False) as saved:
                    arrays = {key: saved[key] for key in saved.files}
                if name == 'lr3e4':
                    with np.load(original_path/'evaluation'/artifact.name, allow_pickle=False) as old:
                        for key in ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability'):
                            np.testing.assert_array_equal(arrays[key], old[key])
                values = summarize_arrays(arrays, trainer)['router']
                for metric in ('ACC', 'WF1', 'MSE'):
                    if not math.isclose(values[metric], summary['evaluation'][rate]['router'][metric], abs_tol=1e-8):
                        raise ValueError('Selected prediction metrics differ')
                measured[rate] = values
                rows.append(dict(seed=seed, model=name, rate=rate,
                                 selected_epoch=summary['selected_epoch_by_rate'][rate], **values))
            for scope, rates in (('mean', RATES), ('high', RATES[5:])):
                for metric in ('ACC', 'WF1', 'MSE'):
                    value = statistics.mean(measured[r][metric] for r in rates)
                    if not math.isclose(value, summary['aggregate']['router'][scope][metric], abs_tol=1e-8):
                        raise ValueError('Aggregate differs from selected predictions')
        runs[str(seed)] = dict(original=original['aggregate']['router'], lr3e4=s['aggregate']['router'],
                              source_commit=s['source_commit'], output=str(path),
                              selected_epochs=s['selected_epoch_by_rate'])
    aggregate = {name: {scope: {metric: dict(
        mean=statistics.mean(runs[str(seed)][name][scope][metric] for seed in (66, 67, 68)),
        std=statistics.stdev(runs[str(seed)][name][scope][metric] for seed in (66, 67, 68)))
        for metric in ('ACC', 'WF1')} for scope in ('mean', 'high')} for name in ('original', 'lr3e4')}
    lines = ['# MOSI Flat3e-4→3e-5 three-seed replication', '', 'INTERNAL TEST-ORACLE SCREEN', '',
             'Seed66 is reused read-only;only67/68 newly trained.150epochs,original GCNet missing.',
             'Per-rate TEST W-F1-best checkpoints,not a single deployment model. No new block or loss.', '',
             '| Seed | Model | ACC | W-F1 | High ACC | High W-F1 |', '|---:|---|---:|---:|---:|---:|']
    for seed, models in runs.items():
        for name in ('original', 'lr3e4'):
            values = [models[name][scope][metric] for scope in ('mean', 'high') for metric in ('ACC', 'WF1')]
            lines.append('| '+seed+' | '+name+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '| Model | Mean ACC ± SD | Mean W-F1 ± SD | High ACC ± SD | High W-F1 ± SD |',
              '|---|---:|---:|---:|---:|']
    for name in ('original', 'lr3e4'):
        values = [aggregate[name][scope][metric] for scope in ('mean', 'high') for metric in ('ACC', 'WF1')]
        lines.append('| '+name+' | '+' | '.join(f"{v['mean']:.4f} ± {v['std']:.4f}" for v in values)+' |')
    lines += ['', 'Per-seed/per-rate scores and selected epochs are in METRICS.csv. Standard deviations are sample SD across three seeds.']
    report_outputs(output, dict(status='complete', label='INTERNAL TEST-ORACLE SCREEN',
                   runs=runs, aggregate=aggregate, seed66_reused=True), rows, lines)
