"""The supplement must never turn two finished seeds into a three-seed result."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = 'experiments.osram_mosi_head48_3seed_20261022.report'


def test_reporter_exists():
    assert importlib.util.find_spec(MODULE) is not None, 'head48 report is not implemented'


def test_omitted_legacy_mixed_rate_hash_uses_only_audited_declared_git_commits():
    from experiments.osram_mosi_head48_3seed_20261022 import report
    record={'identity':{'source_commit':'5877678ce3fac9358a94f1217e3dbac5509f6f58',
            'source_hashes':{p:'known' for p in report.CORE if p!='gcnet_missing_m3/mixed_rate.py'}}}
    recovered=report.core_hashes(record)
    assert recovered['gcnet_missing_m3/mixed_rate.py']=='0d3e9af0b3c35c0fde4a148851390cf47850a50d9bdb0737a6997dc9438b22c1'
    record['identity']['source_commit']='unknown-unverified'
    with pytest.raises(ValueError): report.core_hashes(record)


def records(tmp_path, monkeypatch):
    from experiments.osram_mosi_head48_3seed_20261022 import report
    from gcnet_missing_m3.train_gcnet import TrainConfig
    cfg = TrainConfig(dataset='CMUMOSI', fold=1, seed=66, epochs=150,
        learning_rate=3e-4, batch_size=32, weight_decay=1e-5, optimizer='adam',
        lr_schedule='constant', backbone_type='osram', training_objective='emotion-only',
        disable_unused_aux_modules=True, missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
        evaluate_test=True, osram_output_dim=1600, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
        osram_bidirectional=False, train_rate_mode='cyclic', task_regression_loss='mse')
    flat_paths = {seed: tmp_path / f'flat_{seed}' for seed in (66, 67, 68)}
    old = tmp_path / 'reused_66'
    output = tmp_path / 'new'; output.mkdir()
    monkeypatch.setattr(report, 'FLATS', flat_paths)
    monkeypatch.setattr(report, 'SEED66', old)
    for seed in (66, 67, 68):
        flat = dict(status='complete', epoch=150, initial_backbone_sha256=f'init{seed}',
            identity=dict(config=asdict(replace(cfg, seed=seed)), split={'train': [1]},
                seed_protocol={'seed': seed}, data_files={'feature': 'hash'},
                mask_hashes={r: r for r in report.RATES},
                source_hashes={p: p for p in report.CORE}, source_commit='original'),
            evaluation={r: {'router': {'ACC': 80.0, 'WF1': 79.0, 'MSE': 1.0}}
                        for r in report.RATES},
            best_by_rate={r: {'epoch': 100} for r in report.RATES})
        save(flat_paths[seed], flat)
        candidate = deepcopy(flat)
        candidate.update(parameters={'total': 11245857}, mode='head_48', seed=seed)
        identity = candidate['identity']
        identity['config'].update(osram_key_dim=48, osram_value_dim=48)
        identity['mask_hashes'] = {'test': identity['mask_hashes']}
        identity.update(source_commit=f'candidate{seed}',
            initialization_proof=dict(same_shape_parameters_exact=True,
                source=f'fresh untrained original seed{seed} Flat; not checkpoint weights'))
        for r in report.RATES:
            candidate['evaluation'][r]['router']['WF1'] = 79.0 + (seed - 65) * .1
        save(old if seed == 66 else output / 'runs' / f'seed_{seed}', candidate)
    return report, output, flat_paths, old


def save(folder, record):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'SUMMARY.json').write_text(json.dumps(record))


def change(folder, transform):
    path = folder / 'SUMMARY.json'; record = json.loads(path.read_text())
    transform(record); path.write_text(json.dumps(record))


def test_complete_three_seed_means_sample_sd_and_fixed_rate_epochs(tmp_path, monkeypatch):
    report, output, _, _ = records(tmp_path, monkeypatch)
    result = report.write_report(output, {'jobs': [{'seed': 67}, {'seed': 68}]})
    assert result['three_seed_candidate']['mean']['WF1']['mean'] == pytest.approx(79.2)
    assert result['three_seed_candidate']['mean']['WF1']['std'] == pytest.approx(.1)
    assert result['paired_difference_pp']['67']['mean']['WF1'] == pytest.approx(.2)
    assert len(result['per_rate']) == 48
    assert all(row['epoch'] == 100 for row in result['per_rate'])
    assert 'INTERNAL TEST-ORACLE SCREEN' in (output / 'RESULT.md').read_text()
    assert len((output / 'PER_RATE.csv').read_text().splitlines()) == 49


def test_incomplete_candidates_never_enter_scores_or_three_seed_statistics(tmp_path, monkeypatch):
    report, output, _, _ = records(tmp_path, monkeypatch)
    change(output / 'runs' / 'seed_68', lambda s: s.update(status='running', epoch=80))
    result = report.summarize(output)
    assert set(result['candidate']) == {66, 67}
    assert result['three_seed_candidate'] is None
    assert len(result['per_rate']) == 40


@pytest.mark.parametrize('field', ['config', 'mask', 'core', 'proof', 'seed_source',
                                  'data', 'params', 'metrics', 'epoch', 'selection_epoch'])
def test_mismatched_protocol_or_invalid_completion_is_rejected(tmp_path, monkeypatch, field):
    report, output, _, _ = records(tmp_path, monkeypatch)
    def mutate(s):
        if field == 'config': s['identity']['config']['learning_rate'] = 1e-3
        if field == 'mask': s['identity']['mask_hashes']['test']['0.3'] = 'other'
        if field == 'core': s['identity']['source_hashes'][report.CORE[0]] = 'other'
        if field == 'proof': s['identity']['initialization_proof']['same_shape_parameters_exact'] = False
        if field == 'seed_source': s['identity']['initialization_proof']['source'] = 'fresh untrained original seed66 Flat'
        if field == 'data': s['identity']['data_files'] = {'feature': 'other'}
        if field == 'params': s['parameters']['total'] = 1
        if field == 'metrics': s['evaluation']['0.2']['router']['WF1'] = float('nan')
        if field == 'epoch': s['epoch'] = 149
        if field == 'selection_epoch': s['best_by_rate']['0.2']['epoch'] = 151
    change(output / 'runs' / 'seed_67', mutate)
    with pytest.raises(ValueError): report.summarize(output)


def test_write_report_rejects_extra_or_duplicate_jobs(tmp_path, monkeypatch):
    report, output, _, _ = records(tmp_path, monkeypatch)
    for seeds in ((66, 67, 68), (67, 67), (67,)):
        with pytest.raises(ValueError):
            report.write_report(output, {'jobs': [{'seed': s} for s in seeds]})
