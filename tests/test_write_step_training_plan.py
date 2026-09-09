import importlib.util
from pathlib import Path


def test_training_changes_only_step_and_starts_from_scratch():
    path = Path(__file__).resolve().parents[1] / 'experiments/osram_write_step_train_20260909/run.py'
    spec = importlib.util.spec_from_file_location('step_training', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = dict(seed=66, initial_backbone_checkpoint=None, osram_bidirectional=False,
               fusion_type='mean', train_rate_mode='cyclic', epochs=100,
               learning_rate=.001, backbone_type='osram', checkpoint_selection='test-oracle')
    new = module.training_settings(old)
    assert new == dict(old, osram_write_step=.6)
    assert 'osram_write_step' not in old
    for patch in ({'initial_backbone_checkpoint': 'best.pt'}, {'osram_bidirectional': True},
                  {'osram_forward_slot_reuse': True}, {'osram_write_step': .6}):
        import pytest
        with pytest.raises(ValueError):
            module.training_settings(dict(old, **patch))
