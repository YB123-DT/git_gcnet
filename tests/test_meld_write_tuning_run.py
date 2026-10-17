import copy
import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def runner():
    try:
        return importlib.import_module('experiments.osram_meld_write_tuning_20261017.run')
    except ModuleNotFoundError:
        pytest.fail('Finite write-tuning runner is not implemented')


def base():
    return json.loads((ROOT/'experiments/osram_meld_memory_sensitivity_20261017/FROZEN_SUMMARY.json').read_text())['source_config']


def test_five_configs_change_only_predeclared_write_scalars():
    source = base()
    original = copy.deepcopy(source)
    configs = {name: runner().make_config(source, name) for name in ('W0', 'W1', 'W2', 'W3', 'W4')}
    assert source == original
    assert len({(c['osram_write_step'], c['osram_write_ridge']) for c in configs.values()}) == 5
    for name, config in configs.items():
        changed = {key for key in source if source[key] != config[key]}
        assert changed <= {'osram_write_step', 'osram_write_ridge'}
        assert len(changed) == (0 if name == 'W0' else 1)
        assert config['checkpoint_selection'] == 'test-oracle-per-rate'
    assert len({runner().shared_protocol_hash(c) for c in configs.values()}) == 1
    with pytest.raises(ValueError):
        runner().make_config(source, 'W5')
    with pytest.raises(ValueError):
        runner().make_config({**source, 'missing_mask_protocol': 'conversation-mask-v2'}, 'W0')


def test_noncontrol_admission_waits_for_real_control_epoch(tmp_path, monkeypatch):
    run = runner()
    scheduler = run.WriteScheduler(tmp_path/'queue', model_source=tmp_path/'source', control=tmp_path/'W0')
    assert scheduler.admit({'id': 'W1', 'method': 'Flat'}) is None
    scheduler.control.mkdir()
    (scheduler.control/'EPOCH_STATUS.json').write_text(json.dumps({'epoch':1, 'peak_memory_mib':3000}))
    monkeypatch.setattr(run.Scheduler, 'admit', lambda self, row: (5, 'GPU-test'))
    assert scheduler.admit({'id':'W1', 'method':'Flat'}) == (5, 'GPU-test')
    assert scheduler.costs({'method':'Flat'})[0] >= 3750


def test_worker_command_keeps_uuid_and_independent_variant(tmp_path):
    run = runner()
    scheduler = run.WriteScheduler(tmp_path/'queue', model_source=tmp_path/'b4', control=tmp_path/'W0')
    command = scheduler.command({'id':'W3','output':str(tmp_path/'W3')}, tmp_path/'cfg.json', 'GPU-safe')
    assert '--worker' in command
    assert command[command.index('--variant')+1] == 'W3'
    assert command[command.index('--gpu-uuid')+1] == 'GPU-safe'
    assert command[command.index('--model-source')+1] == str(tmp_path/'b4')


def test_failed_control_does_not_leave_dependents_pending_forever(tmp_path):
    run = runner()
    scheduler = run.WriteScheduler(tmp_path/'queue',model_source=tmp_path/'b4',control=tmp_path/'W0')
    scheduler.state['jobs']['W0'] = {'status':'failed'}
    row = {'id':'W1','method':'Flat','status':'pending','failures':[]}
    assert scheduler.admit(row) is None
    assert row['status']=='failed'
    assert 'control' in row['failures'][0]['reason'].lower()


def test_aggregation_is_exact_eight_rates_and_fixed_high_subset():
    scores = {str(i/10): {key:float(i) for key in ('ACC','UA','WF1','CE')} for i in range(8)}
    result = runner().aggregate(scores)
    assert result['mean']['ACC'] == pytest.approx(3.5)
    assert result['high']['UA'] == pytest.approx(6.)
    with pytest.raises(ValueError):
        runner().aggregate({'0.0': scores['0.0']})
