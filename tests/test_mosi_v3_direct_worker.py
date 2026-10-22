from dataclasses import asdict, replace
import inspect

import numpy as np
import pytest

from test_mosi_v3_capacity_worker import baseline_config


def test_direct_screen_is_one_seed_one_candidate_with_100_epoch_budget():
    from experiments.osram_mosi_v3_direct_20261022.worker import candidate_config, MODE
    cfg = baseline_config()
    direct = candidate_config(cfg, MODE)
    assert direct.epochs == 100
    assert direct.osram_source_role_v3_mode == 'direct'
    assert (direct.osram_adapter_hidden_dim, direct.osram_source_role_v3_dim,
            direct.osram_source_role_v3_pair_hidden) == (512, 128, 256)
    changed = {k for k, v in asdict(cfg).items() if v != asdict(direct)[k]}
    assert changed == {'epochs', 'osram_adapter_hidden_dim', 'osram_source_role_v3',
                       'osram_source_role_v3_dim', 'osram_source_role_v3_pair_hidden',
                       'osram_source_role_v3_mode'}
    plain = candidate_config(cfg, 'f512_plain')
    assert not plain.osram_source_role_v3 and plain.osram_source_role_v3_mode == 'residual'
    with pytest.raises(ValueError):
        candidate_config(replace(cfg, seed=67), MODE)
    with pytest.raises(ValueError):
        candidate_config(cfg, 'f1024_v128')


def test_default_capacity_worker_preserves_original_initialization_and_schedule_checks():
    from experiments.osram_mosi_v3_capacity_20261022 import worker
    params = inspect.signature(worker.main).parameters
    assert params['schedule_factory'].default is worker.settings
    assert params['initialization_check'].default is worker.check_zero_initialization
    arr = {'pred': np.array([.1, -.2])}
    assert worker.check_zero_initialization(arr, arr) == {'zero_init_predictions_exact': True}
    with pytest.raises(AssertionError):
        worker.check_zero_initialization(arr, {'pred': np.array([1., 1.])})


def test_direct_initialization_does_not_claim_identity_of_random_read_replacement():
    from experiments.osram_mosi_v3_direct_20261022.worker import initialization_check, schedule
    info = initialization_check({'pred': np.array([.1])}, {'pred': np.array([.1])})
    assert info['decoder_initialization'] == 'random'
    assert info['readout_identity_initialization'] is False
    assert 'zero_init_predictions_exact' not in info
    assert schedule('CMUMOSI') == dict(epochs=100, first_lr=3e-4, final_lr=3e-4)


def test_direct_wrapper_uses_one_mode_and_tracks_own_sources(monkeypatch):
    from experiments.osram_mosi_v3_direct_20261022 import worker
    calls = []
    monkeypatch.setattr(worker.base, 'main', lambda *a, **k: calls.append((a, k)))
    worker.main(['--phase', 'smoke'])
    args, kwargs = calls[0]
    assert kwargs['allowed_modes'] == ('f512_v128',)
    assert kwargs['config_factory'] is worker.candidate_config
    assert kwargs['initialization_check'] is worker.initialization_check
    assert kwargs['schedule_factory'] is worker.schedule
    assert args == (['--phase', 'smoke'],)
    assert 'experiments/osram_mosi_v3_direct_20261022/worker.py' in worker.source_catalog([])
