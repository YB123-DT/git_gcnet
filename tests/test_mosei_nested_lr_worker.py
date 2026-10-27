"""Only the approved LR differs from the completed original Nested run."""
import importlib
from types import SimpleNamespace

import pytest


def worker():
    return importlib.import_module('experiments.osram_mosei_frozen_nested_lr_20261027.run')


@pytest.mark.parametrize('lr', [1e-4, 1e-3])
def test_lr_override_reaches_every_adam_group_and_survives_full_state_resume(lr):
    import torch
    from experiments.osram_recent_noarea_serial_20261025 import stage2
    from tests.test_recent_noarea_serial_stage2 import ToyModel, toy_bundle
    run, model = worker(), ToyModel()
    optimizer = stage2.build_optimizer(model)
    assert optimizer.param_groups[0]['lr'] == 3e-4
    run.set_learning_rate(optimizer, lr)
    assert run.optimizer_learning_rates(optimizer, lr) == [lr]
    row = stage2.train_epoch(toy_bundle(), model, optimizer, 0, '0.3', True)
    assert row['learning_rate'] == lr and row['updated_parameters'] > 0
    identity = {'learning_rate': lr}
    saved = stage2.checkpoint_payload(model, optimizer, identity, 1, [row], {'epoch': 1})
    restored_model = ToyModel()
    restored_optimizer = stage2.build_optimizer(restored_model)
    stage2.restore_checkpoint(restored_model, restored_optimizer, saved, identity, 2)
    assert run.optimizer_learning_rates(restored_optimizer, lr) == [lr]
    with pytest.raises(ValueError, match='learning rate'):
        run.optimizer_learning_rates(restored_optimizer, 3e-4)


def test_all_optimizer_groups_are_changed_and_nonfinite_or_unapproved_lr_is_rejected():
    run = worker()
    optimizer = SimpleNamespace(param_groups=[{'lr': 3e-4}, {'lr': 3e-4}])
    run.set_learning_rate(optimizer, 1e-3)
    assert run.optimizer_learning_rates(optimizer, 1e-3) == [1e-3, 1e-3]
    for invalid in [3e-4, float('nan'), float('inf'), -1e-4]:
        with pytest.raises(ValueError):
            run.set_learning_rate(optimizer, invalid)


def test_initial_nested_identity_requires_actual_same_state_and_parameter_count():
    run = worker()
    reference = dict(total_parameters=159235, identity={'initial_block_sha256': 'same'})
    run.validate_initial_block(reference, 'same', 159235)
    for digest, count in [('other', 159235), ('same', 159234)]:
        with pytest.raises(ValueError, match='initial'):
            run.validate_initial_block(reference, digest, count)


def test_smoke_records_honest_prefix_without_claiming_full_epoch_mask_hash():
    run = worker()
    original = dict(epoch=2, optimizer_steps=71, training_rates=[.7, 0., .1], N=16326,
                    mask_trace_sha256='full', learning_rate=3e-4)
    row = dict(epoch=2, optimizer_steps=1, training_rates=[.7], N=210,
               mask_trace_sha256='one-batch', learning_rate=1e-4, live_scan_calls=[1])
    proof = run.validate_smoke_prefix(row, original, 1e-4)
    assert proof['rate_prefix_verified'] is True
    assert proof['full_epoch_mask_hash_verified'] is False
    with pytest.raises(ValueError):
        run.validate_smoke_prefix(dict(row, training_rates=[.6]), original, 1e-4)
    with pytest.raises(ValueError):
        run.validate_smoke_prefix(dict(row, optimizer_steps=2), original, 1e-4)


def test_worker_import_does_not_import_torch_trainer_or_signature_models():
    import subprocess
    import sys
    script = """import sys
import experiments.osram_mosei_frozen_nested_lr_20261027.run
assert 'gcnet_missing_m3.train_gcnet' not in sys.modules
assert 'osram_four_block_wiring' not in sys.modules
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('smoke', [False, True])
def test_completion_checks_only_contain_verified_applicable_claims(smoke):
    checks = worker().completion_checks(smoke)
    assert all(checks.values())
    assert ('smoke_rate_step_prefix_verified' in checks) == smoke
    assert ('original_training_trace_exact' in checks) != smoke
    assert ('cyclic_train_rates_verified' in checks) != smoke
