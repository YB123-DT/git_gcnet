"""Capacity workers preserve the fixed second-stage optimization protocol."""
import importlib
from types import SimpleNamespace

import pytest


def worker():
    return importlib.import_module('experiments.osram_mosei_frozen_nested_capacity_20261027.run')


def test_only_fixed_lr_is_allowed_and_every_optimizer_group_is_checked():
    run = worker()
    optimizer = SimpleNamespace(param_groups=[{'lr': 3e-4}, {'lr': 3e-4}])
    run.set_learning_rate(optimizer, 3e-4)
    assert run.optimizer_learning_rates(optimizer, 3e-4) == [3e-4, 3e-4]
    for invalid in (1e-4, 1e-3, float('nan')):
        with pytest.raises(ValueError):
            run.set_learning_rate(optimizer, invalid)


def test_smoke_prefix_is_honest_and_completion_has_no_nonapplicable_false_checks():
    run = worker()
    reference = dict(epoch=1, optimizer_steps=71, training_rates=[0., .1], N=100,
                     mask_trace_sha256='full')
    row = dict(epoch=1, optimizer_steps=1, training_rates=[0.], N=20,
               learning_rate=3e-4, live_scan_calls=[1])
    assert run.validate_smoke_prefix(row, reference, 3e-4)['full_epoch_mask_hash_verified'] is False
    for smoke in [True, False]:
        assert all(run.completion_checks(smoke).values())


def test_worker_import_does_not_import_torch_or_eager_trainers():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c', "import sys; import experiments.osram_mosei_frozen_nested_capacity_20261027.run; assert 'torch' not in sys.modules; assert 'gcnet_missing_m3.train_gcnet' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_initial_state_requires_variant_count_and_reproducible_seeded_draws():
    run = worker()
    from experiments.osram_mosei_frozen_nested_capacity_20261027.protocol import CONFIGS
    for config, options in CONFIGS.items():
        run.validate_initial_block(config, 'same', options['parameters'], 'same')
        with pytest.raises(ValueError):
            run.validate_initial_block(config, 'same', options['parameters'], 'different')
        with pytest.raises(ValueError):
            run.validate_initial_block(config, 'same', 159235, 'same')


def test_fixed_optimizer_full_state_resume_preserves_config_and_rejects_other_variant():
    from experiments.osram_recent_noarea_serial_20261025 import stage2
    from tests.test_recent_noarea_serial_stage2 import ToyModel, toy_bundle
    run, model = worker(), ToyModel()
    optimizer = stage2.build_optimizer(model)
    run.set_learning_rate(optimizer, 3e-4)
    row = stage2.train_epoch(toy_bundle(), model, optimizer, 0, '0.3', True)
    assert row['learning_rate'] == 3e-4 and row['updated_parameters'] > 0
    identity = {'config': 'dim32', 'learning_rate': 3e-4}
    saved = stage2.checkpoint_payload(model, optimizer, identity, 1, [row], {'epoch': 1})
    restored = ToyModel()
    optimizer = stage2.build_optimizer(restored)
    stage2.restore_checkpoint(restored, optimizer, saved, identity, 2)
    assert run.optimizer_learning_rates(optimizer, 3e-4) == [3e-4]
    with pytest.raises(ValueError):
        stage2.restore_checkpoint(restored, optimizer, saved, dict(identity, config='mlp128'), 2)
