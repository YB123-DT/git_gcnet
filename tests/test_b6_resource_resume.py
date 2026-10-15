"""A resource-only controller handoff must not change training/search scope."""
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'experiments/osram_meld_b6_threshold_hpo_20261015/resume_resources.py'
    spec = importlib.util.spec_from_file_location('b6_resource_resume', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_only_authorized_healthy_cards_and_unchanged_pending_budget():
    m = module()
    assert m.parse_args(['--source-root', '/tmp/frozen']).gpus == [2, 3, 5]
    for cards in ('4', '2,4', '0,2,3', '2,3,5,6', '3,3'):
        with pytest.raises(SystemExit):
            m.parse_args(['--source-root', '/tmp/frozen', '--gpus', cards])


def test_resume_rejects_modified_model_config_or_study_protocol():
    m = module()
    config = {'seed': 66, 'epochs': 50}
    initial = [{'threshold_init': .1, 'b6_lr_multiplier': 1.}]
    protocol = {'budget': 20, 'max_pending': 6, 'base_config': config,
                'initial': initial, 'warmstart_parallel': 5}
    m.validate_protocol(protocol, config, initial)
    for key, value in [('budget', 25), ('max_pending', 18), ('base_config', {'seed': 67}),
                       ('initial', []), ('warmstart_parallel', 1)]:
        with pytest.raises(ValueError):
            m.validate_protocol(dict(protocol, **{key: value}), config, initial)


def test_scheduler_keeps_workers_at_original_immutable_code_root(tmp_path):
    m = module()
    from experiments.osram_meld_b6_threshold_hpo_20261015 import run
    scheduler = m.make_scheduler(run, tmp_path / 'output', tmp_path / 'frozen',
                                 [2, 3, 5], 1005.)
    assert scheduler.code_root == (tmp_path / 'frozen').resolve()
    assert scheduler.gpus == (2, 3, 5)
    assert scheduler.capacity == {2: 6, 3: 6, 5: 6}
    assert scheduler.global_per_gpu_limit == 6
