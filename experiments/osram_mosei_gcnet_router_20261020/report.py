"""Complete-only paired seed66 MOSEI internal Test-best report."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.osram_mosi_gcnet_router_3seed_20261019.report import RATES, CORE, report_outputs


def validate_policy(summary):
    cfg = summary['identity']['config']
    if (summary.get('status') != 'complete' or summary.get('phase') != 'train'
            or summary.get('dataset') != 'CMUMOSEI' or summary.get('epoch') != 60
            or cfg.get('dataset') != 'CMUMOSEI' or cfg.get('epochs') != 60
            or cfg.get('missing_mask_protocol') != 'gcnet-random-mask-v1'
            or cfg.get('checkpoint_selection') != 'test-oracle-per-rate'
            or cfg.get('checkpoint_selection_metric') != 'weighted_f1'):
        raise ValueError('Require complete60epoch MOSEI GCNet per-rate Test-best WF1')
    epochs = summary.get('selected_epoch_by_rate', {})
    if set(epochs) != set(RATES) or any(type(e) is not int or not 1 <= e <= 60 for e in epochs.values()):
        raise ValueError('Require eight valid selected epochs within60')
    return epochs


def write_report(output, queue):
    import numpy as np
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
    jobs = queue['jobs']
    if (len(jobs) != 2 or {j['mode'] for j in jobs} != {'flat', 'joint'}
            or any(j['status'] != 'complete' or j['seed'] != 66 or j['dataset'] != 'CMUMOSEI' for j in jobs)):
        raise ValueError('Require exactly two completed MOSEI seed66 Flat/Router runs')
    runs, histories, cohorts, rows = {}, {}, {}, []
    for job in jobs:
        path, mode = Path(job['output']), job['mode']
        s = json.loads((path/'SUMMARY.json').read_text())
        selected = validate_policy(s)
        if any(s.get(k) != job[k] for k in ('mode', 'seed', 'source_commit')):
            raise ValueError('Run provenance differs from queue')
        history = json.loads((path/'history.json').read_text())
        if len(history) != 60 or set(s.get('evaluation', {})) != set(RATES):
            raise ValueError('Missing epoch scores or evaluation rates')
        validate_selected(path, s['best_by_rate'])
        cohorts[mode] = {}
        for rate in RATES:
            p = path/'evaluation'/('random_'+rate.replace('.', 'p')+'.npz')
            metric = s['evaluation'][rate]
            if hashlib.sha256(p.read_bytes()).hexdigest() != metric['output_sha256']:
                raise ValueError('Selected prediction file SHA mismatch')
            with np.load(p, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            actual = scalar.summarize_arrays(arrays, trainer, dataset='CMUMOSEI')['router']
            scores = [h['test'][rate]['router']['WF1'] for h in history]
            epoch = scores.index(max(scores)) + 1
            if selected[rate] != epoch or s['best_by_rate'][rate]['epoch'] != epoch:
                raise ValueError('Not earliest per-rate Test-best WF1 epoch')
            for key in ('ACC', 'WF1', 'MSE'):
                if (not math.isclose(actual[key], metric['router'][key], abs_tol=1e-8)
                        or not math.isclose(actual[key], history[epoch-1]['test'][rate]['router'][key], abs_tol=1e-8)):
                    raise ValueError('Metrics disagree with selected epoch or exact predictions')
            cohorts[mode][rate] = {k: arrays[k] for k in
                ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability')}
            rows.append(dict(dataset='CMUMOSEI', seed=66, mode=mode, rate=rate,
                selected_epoch=epoch, **metric['router']))
        for scope, rates in (('mean', RATES), ('high', RATES[5:])):
            for key in ('ACC', 'WF1', 'MSE'):
                actual = statistics.mean(s['evaluation'][r]['router'][key] for r in rates)
                if not math.isclose(actual, s['aggregate']['router'][scope][key], abs_tol=1e-8):
                    raise ValueError('Aggregate disagrees with selected rate scores')
        runs[mode], histories[mode] = s, history
    flat, joint = runs['flat'], runs['joint']
    for key in ('config', 'split', 'seed_protocol'):
        if flat['identity'][key] != joint['identity'][key]:
            raise ValueError('Paired '+key+' differs')
    if flat['initial_backbone_sha256'] != joint['initial_backbone_sha256']:
        raise ValueError('Paired initialization differs')
    for name in CORE + ('experiments/osram_text_path_router_joint_mosi_20261019/run.py',):
        if flat['identity']['source_hashes'][name] != joint['identity']['source_hashes'][name]:
            raise ValueError('Paired model/sampler/scalar source differs')
    if any(a['mask_trace_sha256'] != b['mask_trace_sha256']
           for a, b in zip(histories['flat'], histories['joint'])):
        raise ValueError('Actual paired training masks differ')
    for rate in RATES:
        for key, value in cohorts['flat'][rate].items():
            np.testing.assert_array_equal(value, cohorts['joint'][rate][key])
    lines = ['# MOSEI seed66: original GCNet missing Flat vs one-stage Router', '',
        'INTERNAL TEST-ORACLE DIAGNOSTIC ONLY', '',
        'Two fresh60epoch trajectories, original MOSEI official train/dev/test.',
        'Each rate selects its own TEST W-F1-best epoch; ACC uses the same predictions.',
        'Fixed legal common GCNet masks; no extra training holdout or validation selection.',
        'Eight-rate scores may combine eight checkpoints, not one deployable model.', '',
        '| Rate | Flat ACC | Router ACC | Flat W-F1 | Router W-F1 | Delta W-F1 pp |',
        '|---|---:|---:|---:|---:|---:|']
    for rate in RATES:
        f, j = flat['evaluation'][rate]['router'], joint['evaluation'][rate]['router']
        lines.append(f'| {rate} | {f["ACC"]:.4f} | {j["ACC"]:.4f} | {f["WF1"]:.4f} | {j["WF1"]:.4f} | {j["WF1"]-f["WF1"]:+.4f} |')
    for scope in ('mean', 'high'):
        f, j = flat['aggregate']['router'][scope], joint['aggregate']['router'][scope]
        lines.append(f'| {scope} | {f["ACC"]:.4f} | {j["ACC"]:.4f} | {f["WF1"]:.4f} | {j["WF1"]:.4f} | {j["WF1"]-f["WF1"]:+.4f} |')
    lines += ['', 'Seed66 only; no automatic extra seeds or methods.',
        'Older MOSEI scores used different missing masks/budgets and are not paired comparators.',
        'No independent test-generalization claim. No new architecture or auxiliary loss.']
    report_outputs(Path(output), dict(runs=runs, paired_initialization=True,
        paired_actual_training_masks=True, paired_exact_test_cohort_masks=True,
        same_call_metrics_recomputed=True), rows, lines)
