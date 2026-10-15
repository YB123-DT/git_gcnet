"""Postprocessing-only repair must never turn a training failure into success."""
import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1] / 'experiments/osram_meld_b6_top2_matched_flat_3seed_20261015/recover.py'
    spec = importlib.util.spec_from_file_location('b6_top2_recovery', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_only_recovers_completed_training_with_known_positional_replay_error():
    m = module()
    raw = dict(status='failed', error='AssertionError: Arrays are not equal')
    assert m.recovery_action(raw, 50, 8) == 'recover'
    assert m.recovery_action(dict(raw, status='running'), 50, 8) == 'wait'
    assert m.recovery_action(raw, 49, 8) == 'failed'
    assert m.recovery_action(raw, 50, 7) == 'failed'
    assert m.recovery_action(dict(raw, error='CUDA out of memory'), 50, 8) == 'failed'
    assert m.recovery_action(dict(status='complete'), 50, 8) == 'existing_complete'
