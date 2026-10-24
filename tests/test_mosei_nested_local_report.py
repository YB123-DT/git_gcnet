"""CPU-only checks for the finite, paired MOSEI local screen."""
import copy
import importlib
import json
import sys
import types

import pytest


def module():
    return importlib.import_module('experiments.osram_mosei_nested_local_20261024.report')


def pair(nested_lr=None):
    base = dict(config={'osram_meaningful_block': 'none', 'learning_rate': 1e-4},
        initial_backbone_sha256='same', total_parameters=100, nested_parameters=0,
        identity=dict(mask_reference='mask', split='split', source_hashes={
            'gcnet_missing_m3/osram.py': 'same',
            'gcnet_missing_m3/meaningful_new40_registry.py': 'old'}))
    candidate = copy.deepcopy(base)
    candidate.update(total_parameters=120, nested_parameters=20)
    candidate['config']['osram_meaningful_block'] = 'nested_dim32'
    candidate['identity']['source_hashes']['gcnet_missing_m3/meaningful_new40_registry.py'] = 'new'
    if nested_lr is not None:
        candidate['nested_learning_rate'] = nested_lr
        candidate['identity']['nested_learning_rate'] = nested_lr
        candidate['optimizer_groups'] = [dict(name='backbone', lr=1e-4, parameters=100),
                                         dict(name='nested', lr=nested_lr, parameters=20)]
        candidate['identity']['optimizer_groups'] = copy.deepcopy(candidate['optimizer_groups'])
    history = [dict(epoch=i, mask_trace_sha256=f'trace{i}', optimizer_steps=2,
                    learning_rates=[1e-4]) for i in range(60)]
    candidate_history = copy.deepcopy(history)
    if nested_lr is not None:
        for row in candidate_history:
            row['learning_rates'].append(nested_lr)
    return candidate, candidate_history, base, history


def test_partial_does_not_import_training_modules_or_rank(tmp_path):
    with pytest.raises(ValueError, match='four|4'):
        module().write_report(tmp_path, dict(jobs=[], source_commit='sha'))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('lr', [None, 5e-5, 2e-4])
def test_pair_accepts_only_explicit_learning_rate_difference(lr):
    candidate, history, base, old = pair(lr)
    module().validate_pair(candidate, history, base, old, nested_learning_rate=lr)
    assert history[0]['learning_rates'] == ([1e-4] if lr is None else [1e-4, lr])


@pytest.mark.parametrize('mutation', ['backbone_lr', 'nested_lr', 'extra_group', 'schedule',
    'identity_lr', 'group_order', 'group_count', 'mask', 'split', 'source', 'trace', 'initialization'])
def test_pair_rejects_schedule_group_or_pairing_drift(mutation):
    candidate, history, base, old = pair(5e-5)
    if mutation == 'backbone_lr': history[0]['learning_rates'][0] = 2e-4
    elif mutation == 'nested_lr': history[0]['learning_rates'][1] = 1e-4
    elif mutation == 'extra_group': history[0]['learning_rates'].append(5e-5)
    elif mutation == 'schedule': history[-1]['learning_rates'][1] = 1e-5
    elif mutation == 'identity_lr': candidate['identity']['nested_learning_rate'] = 2e-4
    elif mutation == 'group_order': candidate['optimizer_groups'].reverse()
    elif mutation == 'group_count': candidate['optimizer_groups'][0]['parameters'] += 1
    elif mutation in ('mask', 'split'): candidate['identity']['mask_reference' if mutation == 'mask' else 'split'] = 'different'
    elif mutation == 'source': candidate['identity']['source_hashes']['gcnet_missing_m3/osram.py'] = 'different'
    elif mutation == 'trace': history[0]['mask_trace_sha256'] = 'different'
    else: candidate['initial_backbone_sha256'] = 'different'
    with pytest.raises(ValueError):
        module().validate_pair(candidate, history, base, old, nested_learning_rate=5e-5)


def test_source_extension_allows_five_named_entries_only(tmp_path):
    report = module()
    old, new = tmp_path/'old', tmp_path/'new'
    for root in (old, new):
        for package in ('gcnet_missing_m3', 'gcnet_modality_jepa'):
            (root/package).mkdir(parents=True)
            (root/package/'model.py').write_text('original numerical code\n')
    (old/report.REGISTRY).write_text('before\nafter\n')
    additions = ''.join(f"    '{name}': {{'dim': 32}},\n" for name in report.REGISTRY_EXTENSIONS)
    (new/report.REGISTRY).write_text('before\n'+additions+'after\n')
    assert report.validate_source_extension(new, old)['unchanged_numerical_files'] == 2
    (new/'gcnet_missing_m3/model.py').write_text('changed\n')
    with pytest.raises(ValueError, match='source'):
        report.validate_source_extension(new, old)


def make_complete(tmp_path, monkeypatch):
    report = module()
    monkeypatch.setattr(report, 'validate_source_extension', lambda root: {'verified': True})
    monkeypatch.setattr(report, 'CONTROL_ROOT', tmp_path/'controls')
    fake = types.ModuleType('experiments.osram_nested_cmerc_tuning_20261023.run')
    fake.validate_summary = lambda job: json.loads((__import__('pathlib').Path(job['output'])/'SUMMARY.json').read_text())
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    selection = types.ModuleType('experiments.osram_iemocap6_gcnet_testbest_20261020.worker')
    selection.validate_selected = lambda path, best: None
    monkeypatch.setitem(sys.modules, selection.__name__, selection)
    jobs = []
    arms = [('flat', 'none', None, report.CONTROL_ROOT/'none'),
            ('nested32', 'nested_dim32', None, report.CONTROL_ROOT/'nested_dim32')]
    for variant, (method, lr) in report.VARIANTS.items():
        path = tmp_path/variant
        jobs.append(dict(variant=variant, method=method, dataset='CMUMOSEI', seed=66, fold=1,
            epochs=60, learning_rate=1e-4, nested_learning_rate=lr, output=str(path),
            status='complete', source_commit='sha'))
        arms.append((variant, method, lr, path))
    for variant, method, lr, path in arms:
        path.mkdir(parents=True)
        summary, history, base, old = pair(lr)
        if method == 'none': summary, history = base, old
        summary['config']['osram_meaningful_block'] = method
        best, evaluation = {}, {}
        for i, rate in enumerate(report.RATES):
            metrics = dict(router=dict(ACC=80.+i, WF1=79.+i, MSE=.5+i/100))
            best[rate] = dict(epoch=1, metrics=metrics, same_evaluation_call=True)
            evaluation[rate] = dict(metrics, selected_epoch=1)
            for row in history:
                row.setdefault('test', {})[rate] = copy.deepcopy(metrics)
        aggregate = {scope: {metric: sum(evaluation[r]['router'][metric] for r in rates)/len(rates)
                            for metric in report.METRICS}
                     for scope, rates in [('mean', report.RATES), ('high', report.RATES[5:])]}
        summary.update(status='complete', phase='train', epoch=60, seed=66, dataset='CMUMOSEI',
            fold=1, method=method, source_commit='sha', best_by_rate=best, evaluation=evaluation,
            aggregate=dict(router=aggregate), seconds=100., peak_memory_mib=1000.)
        (path/'SUMMARY.json').write_text(json.dumps(summary))
        (path/'history.json').write_text(json.dumps(history))
    return dict(source_commit='sha', jobs=jobs)


def test_complete_report_contains_all_six_arms_and_48_rates(tmp_path, monkeypatch):
    report = module()
    state = make_complete(tmp_path, monkeypatch)
    output = tmp_path/'report'
    report.write_report(output, state)
    result = json.loads((output/'RESULT.json').read_text())
    assert len(result['runs']) == 6
    assert len((output/'PER_RATE.csv').read_text().splitlines()) == 49
    assert set(result['runs'][0]['delta_vs_flat']['high']) == set(report.METRICS)
    assert set(result['runs'][0]['delta_vs_nested32']['high']) == set(report.METRICS)
    assert 'INTERNAL TEST-ORACLE SCREEN' in (output/'RESULT.md').read_text()


@pytest.mark.parametrize('mutation', ['duplicate_variant', 'wrong_method', 'partial', 'nan', 'selected_epoch'])
def test_report_rejects_incomplete_or_invalid_artifacts(tmp_path, monkeypatch, mutation):
    report = module()
    state = make_complete(tmp_path, monkeypatch)
    if mutation == 'duplicate_variant': state['jobs'][1]['variant'] = state['jobs'][0]['variant']
    elif mutation == 'wrong_method': state['jobs'][0]['method'] = 'none'
    elif mutation == 'partial': state['jobs'][-1]['status'] = 'running'
    else:
        from pathlib import Path
        path = Path(state['jobs'][0]['output'])/'SUMMARY.json'
        summary = json.loads(path.read_text())
        if mutation == 'nan': summary['aggregate']['router']['mean']['WF1'] = float('nan')
        else: summary['best_by_rate']['0.0']['epoch'] = 61
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        report.write_report(tmp_path/'report', state)
