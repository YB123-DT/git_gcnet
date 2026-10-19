"""MOSI reruns change only missing protocol and retain an auditable legacy bridge."""
import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def worker():
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker
    return worker


def historical_config():
    return dict(dataset='CMUMOSI', seed=66, fold=1, epochs=150, learning_rate=1e-4,
                batch_size=32, dropout=.5, projector_dropout=.1, latent_dim=256,
                osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64,
                osram_value_dim=64, osram_write_step=.6, training_objective='emotion-only',
                train_rate_mode='cyclic', missing_mask_protocol='conversation-mask-v2',
                task_regression_loss='mse', mosi_task_mode='regression', optimizer='adam',
                weight_decay=1e-5, lr_schedule='constant')


@pytest.mark.parametrize('seed', (66, 67, 68))
def test_new_config_changes_missing_without_mutating_legacy(seed):
    raw = historical_config()
    old = copy.deepcopy(raw)
    legacy, current = worker().configs(raw, seed)
    assert raw == old
    assert legacy.seed == 66 and legacy.missing_mask_protocol == 'conversation-mask-v2'
    assert current.seed == seed and current.missing_mask_protocol == 'gcnet-random-mask-v1'
    assert current.epochs == 150 and current.learning_rate == .001
    assert current.fold == 1 and current.osram_output_dim == 1600
    assert current.task_regression_loss == 'mse' and current.mosi_task_mode == 'regression'
    assert current.checkpoint_selection == 'validation'
    assert current.checkpoint_selection_metric == 'weighted_f1'
    assert not current.paired_history_views and current.osram_readout_fusion == 'flat'


def test_invalid_seed_or_changed_historical_hyperparameter_is_rejected():
    with pytest.raises(ValueError, match='66.*67.*68'):
        worker().configs(historical_config(), 65)
    with pytest.raises(ValueError, match='batch_size'):
        worker().configs(dict(historical_config(), batch_size=64), 66)


def test_gcnet_sampler_retains_host_guest_padded_rectangle_and_high_rate_one_hot():
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_modality_jepa.gcnet_legacy_random_mask import random_mask
    _, cfg = worker().configs(historical_config(), 68)
    umask = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]])
    for rate in (0., .3, .7):
        schedule = trainer._schedules(cfg, 'train')[rate]
        np.random.seed(891)
        actual = schedule.generate_batch(['first', 'second'], umask, 3)
        np.random.seed(891)
        expected = [torch.as_tensor(random_mask(3, 8, rate).reshape(2, 4, 3).transpose(1, 0, 2))
                    for _ in range(2)]
        valid = umask.T.bool()
        for a, e in zip(actual, expected):
            assert torch.equal(a[valid], e[valid])
            assert a[~valid].count_nonzero() == 0
            assert a[valid].sum(-1).min() >= 1
            if rate == .7:
                assert torch.equal(a[valid].sum(-1), torch.ones(6, dtype=a.dtype))
            if rate == 0.:
                assert torch.equal(a[valid], torch.ones_like(a[valid]))
        if rate == .3:
            assert not torch.equal(actual[0], actual[1])


def test_bridge_validation_requires_exact_all_three_arrays_and_metrics():
    arrays = dict(predictions=np.array([.1, -.2]), labels=np.array([1., -1.]),
                  availability=np.array([[1, 0, 0], [0, 1, 1]]))
    measured = dict(mask_sha256='mask', accuracy=.8, weighted_f1=.7)
    ref = dict(mask_sha256={'0.3': 'mask'}, test={'0.3': dict(accuracy=.8, weighted_f1=.7)})
    worker().validate_bridge(arrays, arrays, measured, ref)
    for field in arrays:
        changed = {k: v.copy() for k, v in arrays.items()}
        changed[field].flat[0] += 1
        with pytest.raises(AssertionError, match=field):
            worker().validate_bridge(changed, arrays, measured, ref)
    with pytest.raises(ValueError, match='weighted_f1'):
        worker().validate_bridge(arrays, arrays, dict(measured, weighted_f1=.6), ref)


def test_eval_banks_force_seed66_without_changing_training_cfg_or_rng(monkeypatch):
    from experiments.osram_text_path_router_joint_mosi_20261019 import run
    _, cfg = worker().configs(historical_config(), 68)
    bundle = dict(cfg=cfg)
    seen = []
    monkeypatch.setattr(run, 'build_masks', lambda b, split: seen.append((b['cfg'].seed, b['cfg'].missing_mask_protocol, split)) or {})
    before = torch.get_rng_state().clone()
    assert worker().build_masks(bundle, 'validation') == {}
    assert seen == [(66, 'gcnet-random-mask-v1', 'validation')]
    assert bundle['cfg'] is cfg and cfg.seed == 68
    assert torch.equal(before, torch.get_rng_state())


def test_fresh_backbone_factory_uses_requested_seed_but_historical_init_fold5(monkeypatch):
    from experiments.osram_meld_flat_source_combinations_20261013 import evaluate
    _, cfg = worker().configs(historical_config(), 67)
    seen = []
    monkeypatch.setattr(evaluate, '_build_model', lambda c, dims, dev: seen.append((c.seed, c.fold, dims, dev)) or torch.nn.Linear(2, 1))
    actual = worker().fresh_backbone(cfg, (2, 3, 4), torch.device('cpu'))
    assert seen == [(67, 5, (2, 3, 4), torch.device('cpu'))]
    assert cfg.fold == 1 and all(p.requires_grad for p in actual.parameters())


def test_checkpoint_roundtrip_restores_rng_optimizer_and_rejects_identity_changes():
    from experiments.osram_text_path_router_joint_20261019.run import checkpoint_payload, restore
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam([dict(params=model.parameters(), name='backbone', lr=.001)])
    model(torch.ones(2, 2)).square().mean().backward()
    opt.step()
    identity = dict(seed=67, missing_mask_protocol='gcnet-random-mask-v1')
    saved = checkpoint_payload(model, opt, epoch=101, identity=identity, history=[], best_key=(80., -1.))
    draw = torch.rand(3)
    with torch.no_grad():
        model.weight.zero_()
    restore(model, opt, saved, identity)
    assert torch.equal(model.weight.cpu(), saved['model']['weight'])
    assert torch.equal(torch.rand(3), draw)
    assert opt.state_dict()['state']
    with pytest.raises(ValueError, match='identity'):
        restore(model, opt, saved, dict(identity, seed=68))


def test_main_defers_config_dependent_trainer_import_until_dataset_env_setup():
    import inspect
    source = inspect.getsource(worker().main)
    assert source.index('bundle = setup(') < source.index('from gcnet_modality_jepa.train_gcnet import')


def test_missing_manifest_can_be_reconstructed_only_byte_identically(tmp_path):
    payload = dict(dataset_root='/data', feature_roots=['a', 't', 'v'], files={'a': 'hash'})
    expected = hashlib.sha256((json.dumps(payload, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    combined = tmp_path / 'MANIFEST.json'
    combined.write_text(json.dumps(dict(data=dict(CMUMOSI=payload))))
    missing = tmp_path / 'DATA.json'
    actual, proof = worker().load_manifest(missing, expected, combined=combined)
    assert actual == payload
    assert proof['byte_identical_reconstruction'] is True
    assert proof['data_sha256'] == expected and proof['manifest_origin'] == str(combined)
    assert proof['combined_manifest_sha256'] == hashlib.sha256(combined.read_bytes()).hexdigest()
    assert not missing.exists()
    combined.write_text(json.dumps(dict(data=dict(CMUMOSI=dict(payload, dataset_root='/other')))))
    with pytest.raises(ValueError, match='hash'):
        worker().load_manifest(missing, expected, combined=combined)


def test_existing_wrong_manifest_never_silently_falls_back(tmp_path):
    payload = dict(files={})
    expected = hashlib.sha256((json.dumps(payload, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    combined = tmp_path / 'MANIFEST.json'
    combined.write_text(json.dumps(dict(data=dict(CMUMOSI=payload))))
    existing = tmp_path / 'DATA.json'
    existing.write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        worker().load_manifest(existing, expected, combined=combined)
    existing.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    actual, proof = worker().load_manifest(existing, expected, combined=combined)
    assert actual == payload and proof['byte_identical_reconstruction'] is False
    assert proof['manifest_origin'] == str(existing)
