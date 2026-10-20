"""Strict complete-only Test-best ACC report; never mix validation-selected runs."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.osram_mosi_gcnet_router_3seed_20261019.report import (
    CORE, RATES, report_outputs, stats,
)


def validate_policy(summary):
    identity = summary['identity']
    cfg, split = identity['config'], identity['split']
    if (cfg.get('checkpoint_selection') != 'test-oracle-per-rate'
            or cfg.get('checkpoint_selection_metric') != 'accuracy'
            or cfg.get('missing_mask_protocol') != 'gcnet-random-mask-v1'
            or cfg.get('epochs') != 100):
        raise ValueError('Require original GCNet100epoch per-rate Test-best ACC')
    train = set(split.get('train_indices', []))
    original = set(split.get('original_train_indices', []))
    test = set(split.get('test_indices', []))
    if (split.get('adapter_holdout') is not False or not train or train != original
            or not test or train & test):
        raise ValueError('Original full training fold required, without extra holdout')
    epochs = summary.get('selected_epoch_by_rate', {})
    if set(epochs) != set(RATES):
        raise ValueError('Require eight actual selected epochs')
    if any(type(e) is not int or not 1 <= e <= 100 for e in epochs.values()):
        raise ValueError('Selected epoch outside complete100epoch budget')
    return epochs


def load_run(job):
    import numpy as np
    from experiments.osram_text_path_router_joint_20261019.run import summarize_arrays
    path = Path(job['output'])
    summary = json.loads((path/'SUMMARY.json').read_text())
    if (summary.get('status') != 'complete' or summary.get('phase') != 'train'
            or summary.get('dataset') != 'IEMOCAPSix' or summary.get('epoch') != 100
            or any(summary.get(k) != job[k] for k in ('fold', 'seed', 'mode', 'source_commit'))):
        raise ValueError('Incomplete or mismatched run: '+str(path))
    epochs = validate_policy(summary)
    history = json.loads((path/'history.json').read_text())
    if len(history) != 100 or set(summary.get('evaluation', {})) != set(RATES):
        raise ValueError('Require complete training and eight rates')
    cohort = {}
    for rate in RATES:
        artifact = path/'evaluation'/('random_'+rate.replace('.', 'p')+'.npz')
        row = summary['evaluation'][rate]
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != row['output_sha256']:
            raise ValueError('Selected prediction integrity failed')
        with np.load(artifact, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        actual = summarize_arrays(arrays, 6)['router']
        for metric in ('ACC', 'UA', 'WF1', 'CE'):
            if not math.isclose(actual[metric], row['router'][metric], abs_tol=1e-8):
                raise ValueError('Selected metrics disagree with same-call predictions')
        candidates = [epoch['test'][rate]['router']['ACC'] for epoch in history]
        selected = candidates.index(max(candidates)) + 1
        if epochs[rate] != selected:
            raise ValueError('Not earliest per-rate Test-best ACC epoch')
        for metric in ('ACC', 'UA', 'WF1', 'CE'):
            if not math.isclose(row['router'][metric],
                    history[selected - 1]['test'][rate]['router'][metric], abs_tol=1e-8):
                raise ValueError('Saved metrics do not match selected epoch')
        cohort[rate] = {k: arrays[k] for k in
            ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability')}
    for scope, rates in (('mean', RATES), ('high', RATES[5:])):
        for metric in ('ACC', 'UA', 'WF1', 'CE'):
            computed = statistics.mean(summary['evaluation'][r]['router'][metric] for r in rates)
            if not math.isclose(computed, summary['aggregate']['router'][scope][metric], abs_tol=1e-8):
                raise ValueError('Aggregate differs from per-rate metrics')
    return summary, history, cohort


def write_report(output, queue):
    import numpy as np
    jobs = queue['jobs']
    expected = {(f, s, m) for f in range(1, 6) for s in (66, 67, 68) for m in ('flat', 'joint')}
    if (len(jobs) != 30 or {(j['fold'], j['seed'], j['mode']) for j in jobs} != expected
            or any(j['status'] != 'complete' or j.get('reused') for j in jobs)):
        raise ValueError('Require30new completed fold/seed/model runs, no old validation reuse')
    runs, checks, rows, core = {}, {}, [], None
    lines = ['# IEMOCAP-6 corrected per-rate Test-best ACC', '',
        'INTERNAL TEST-ORACLE DIAGNOSTIC ONLY', '',
        '30fresh runs: fivefold × threeseed × Flat/Router,100epochs each.',
        'Original full training fold restored; no extra10% adapter holdout.',
        'GCNet random-mask-v1; fold-specific fixed common evaluation masks.',
        'Each rate selects its own TEST ACC-best epoch, earliest tie; UA/W-F1 use the same predictions.',
        'Eight-rate means may combine eight checkpoints, not a single deployment model.', '',
        '| Fold | Seed | Model | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for fold in range(1, 6):
        runs[str(fold)], checks[str(fold)], reference = {}, {}, None
        for seed in (66, 67, 68):
            pair, histories, cohorts = {}, {}, {}
            for mode in ('flat', 'joint'):
                job = next(j for j in jobs if (j['fold'], j['seed'], j['mode']) == (fold, seed, mode))
                pair[mode], histories[mode], cohorts[mode] = load_run(job)
                new_core = {n: pair[mode]['identity']['source_hashes'][n] for n in CORE}
                if core is not None and core != new_core:
                    raise ValueError('Model/sampler sources changed across runs')
                core = new_core
            flat, joint = pair['flat'], pair['joint']
            for key in ('config', 'split', 'seed_protocol'):
                if flat['identity'][key] != joint['identity'][key]:
                    raise ValueError('Paired '+key+' differs')
            if flat['initial_backbone_sha256'] != joint['initial_backbone_sha256']:
                raise ValueError('Paired initialization differs')
            if any(a['mask_trace_sha256'] != b['mask_trace_sha256']
                   for a, b in zip(histories['flat'], histories['joint'])):
                raise ValueError('Paired actual training masks differ')
            for rate in RATES:
                for name, value in cohorts['flat'][rate].items():
                    np.testing.assert_array_equal(value, cohorts['joint'][rate][name])
                    if reference is not None:
                        np.testing.assert_array_equal(value, reference[rate][name])
            reference = cohorts['flat']
            runs[str(fold)][str(seed)] = pair
            checks[str(fold)][str(seed)] = dict(initialization_exact=True, training_masks_exact=True,
                same_fold_cross_seed_test_masks_exact=True, same_call_selected_metrics_verified=True)
            for mode, summary in pair.items():
                values = [summary['aggregate']['router'][scope][metric] for scope in ('mean', 'high')
                    for metric in ('ACC', 'UA', 'WF1')]
                lines.append(f'| {fold} | {seed} | {mode} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
                for rate in RATES:
                    rows.append(dict(dataset='IEMOCAPSix', fold=fold, seed=seed, mode=mode, rate=rate,
                        selected_epoch=summary['selected_epoch_by_rate'][rate],
                        **summary['evaluation'][rate]['router']))
    per_seed = {str(s): {m: {scope: {metric: statistics.mean([
        runs[str(f)][str(s)][m]['aggregate']['router'][scope][metric] for f in range(1, 6)])
        for metric in ('ACC', 'UA', 'WF1', 'CE')} for scope in ('mean', 'high')}
        for m in ('flat', 'joint')} for s in (66, 67, 68)}
    aggregate = {m: {scope: {metric: stats([per_seed[str(s)][m][scope][metric] for s in (66, 67, 68)])
        for metric in ('ACC', 'UA', 'WF1', 'CE')} for scope in ('mean', 'high')}
        for m in ('flat', 'joint')}
    per_rate = {r: {m: {metric: stats([statistics.mean([
        runs[str(f)][str(s)][m]['evaluation'][r]['router'][metric] for f in range(1, 6)])
        for s in (66, 67, 68)]) for metric in ('ACC', 'UA', 'WF1', 'CE')}
        for m in ('flat', 'joint')} for r in RATES}
    lines += ['', '## Three-seed mean ± sample standard deviation (fivefold means)', '',
        '| Model | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode in ('flat', 'joint'):
        values = [aggregate[mode][scope][metric] for scope in ('mean', 'high') for metric in ('ACC', 'UA', 'WF1')]
        lines.append('| '+mode+' | '+' | '.join(f'{v["mean"]:.4f} ± {v["std"]:.4f}' for v in values)+' |')
    lines += ['', '## Per-rate three-seed mean of fivefold metrics', '',
        '| Rate | Flat ACC | Router ACC | Flat UA | Router UA | Flat W-F1 | Router W-F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for rate in RATES:
        values = [per_rate[rate][mode][metric]['mean'] for metric in ('ACC', 'UA', 'WF1') for mode in ('flat', 'joint')]
        lines.append('| '+rate+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', 'This repairs both selection and the unauthorized training holdout; it is NOT a selection-only ablation.',
        'Fixed legal test masks are not the historical per-epoch resampled masks; do not claim exact recovery of61.714.',
        'Original protected baseline remains intact. No new architecture, auxiliary loss or automatic search.']
    report_outputs(output, dict(runs=runs, checks=checks, per_seed=per_seed,
        aggregate=aggregate, per_rate=per_rate, fold_aggregation='unweighted-fold-metric-mean'), rows, lines)
