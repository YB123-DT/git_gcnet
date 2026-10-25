"""Finite MOSI control and independent selected-prediction validation."""
import importlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

PACKAGE = 'experiments.osram_signature_noarea_mosi66_20261025'


def test_only_eight_mosi_seed66_jobs_and_one_smoke():
    module = importlib.import_module(PACKAGE+'.scheduler')
    jobs = module.jobs_for('train')
    assert len(jobs) == len({j['id'] for j in jobs}) == 8
    assert {(j['dataset'], j['seed'], j['rate']) for j in jobs} == {
        ('CMUMOSI', 66, f'{i/10:.1f}') for i in range(8)}
    assert [j['rate'] for j in module.jobs_for('smoke')] == ['0.7']
    with pytest.raises(ValueError):
        module.parse_gpu_map('4=GPU-bad')


def test_detached_launch_has_immutable_source_and_uuid(tmp_path, monkeypatch):
    module = importlib.import_module(PACKAGE+'.scheduler')
    captured = {}
    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    args = SimpleNamespace(root=tmp_path, resume=False, source_commit='source')
    job = module.jobs_for('train')[0]
    module.launch(args, job, 'GPU-ok', 'train')
    assert captured['start_new_session'] is True
    assert captured['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-ok'
    assert module.MODULE in captured['command']
    assert job['source_commit'] == 'source'


def test_prediction_metrics_exclude_gold_zero_but_not_training_loss():
    import numpy as np
    module = importlib.import_module(PACKAGE+'.summarize')
    score = module.prediction_metrics(np.array([-1., 0., 1.]), np.array([-2., 4., 2.]))
    assert score['ACC'] == score['WF1'] == 100.
    assert score['N'] == 2


def test_paired_identity_requires_original_initialization_mask_and_source():
    module = importlib.import_module(PACKAGE+'.summarize')
    identity = dict(dataset='CMUMOSI', seed=66, stage2_epochs=150,
        source_commit='new', evaluation_rate='0.3', cyclic_training=True,
        initial_block_sha256='init', test_mask_sha256='mask',
        effective_stage2_config={'learning_rate': 3e-4}, parent={'parent_id': 'parent'},
        mask_protocol='gcnet-random-mask-v1', ablation='degree2_no_area')
    normal = deepcopy(identity)
    normal['source_commit'] = module.NORMAL_SOURCE
    normal.pop('ablation')
    module.validate_pair(identity, normal, 'new', '0.3')
    for key in ('initial_block_sha256', 'test_mask_sha256', 'parent', 'effective_stage2_config'):
        wrong = deepcopy(normal)
        wrong[key] = 'different'
        with pytest.raises(ValueError):
            module.validate_pair(identity, wrong, 'new', '0.3')
    normal['source_commit'] = 'unreviewed'
    with pytest.raises(ValueError):
        module.validate_pair(identity, normal, 'new', '0.3')


def test_incomplete_results_never_form_an_eight_rate_mean(tmp_path):
    module = importlib.import_module(PACKAGE+'.summarize')
    result = module.summarize(tmp_path, 'source', [dict(id='pending', status='pending')])
    assert result['status'] == 'incomplete'
    assert result['aggregate'] == []
    assert result['validated_jobs'] == 0
    assert json.loads((tmp_path/'RESULT.json').read_text())['expected_jobs'] == 8
