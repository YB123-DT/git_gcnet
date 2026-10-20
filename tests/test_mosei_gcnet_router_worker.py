"""MOSEI is a separate 60-epoch GCNet/Test-best run, not MOSI continuation."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch


def worker():
    from experiments.osram_mosei_gcnet_router_20261020 import worker
    return worker


def historical_config():
    return dict(dataset='CMUMOSEI', fold=1, seed=66, epochs=100, batch_size=32,
        learning_rate=.001, weight_decay=1e-5, lr_schedule='constant', optimizer='adam',
        dropout=.5, projector_dropout=.1, latent_dim=256, backbone_type='osram',
        training_objective='emotion-only', osram_output_dim=1600, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
        osram_bidirectional=False, osram_readout_fusion='flat', fusion_type='mean',
        train_rate_mode='cyclic', checkpoint_selection='test-oracle-per-rate',
        evaluation_protocol='official', mosi_task_mode='regression',
        task_regression_loss='mse', disable_unused_aux_modules=True)


def test_config_preserves_mosei_baseline_with_explicit_approved_changes_only():
    raw = historical_config()
    before = copy.deepcopy(raw)
    cfg = worker().configs(raw)
    assert raw == before
    assert cfg.dataset == 'CMUMOSEI' and cfg.epochs == 60
    assert cfg.seed == 66 and cfg.fold == 1 and cfg.batch_size == 32
    assert cfg.learning_rate == .001 and cfg.weight_decay == 1e-5
    assert cfg.lr_schedule == 'constant' and cfg.optimizer == 'adam'
    assert cfg.checkpoint_selection == 'test-oracle-per-rate'
    assert cfg.checkpoint_selection_metric == 'weighted_f1' and cfg.evaluate_test
    assert cfg.missing_mask_protocol == 'gcnet-random-mask-v1'
    assert cfg.osram_output_dim == 1600 and cfg.osram_num_heads == 8
    assert not cfg.osram_bidirectional and cfg.osram_write_step == .6
    assert tuple(cfg.train_missing_rates) == tuple(i / 10 for i in range(8))
    assert not cfg.paired_history_views and not cfg.osram_tasr
    assert cfg.initial_backbone_checkpoint is None


@pytest.mark.parametrize('field,value', [('dataset', 'CMUMOSI'), ('batch_size', 64),
    ('learning_rate', .0001), ('osram_bidirectional', True), ('epochs', 150),
    ('paired_history_views', True), ('osram_output_dim', 512)])
def test_config_rejects_wrong_dataset_and_unauthorized_settings(field, value):
    with pytest.raises(ValueError, match=field):
        worker().configs(dict(historical_config(), **{field: value}))
    with pytest.raises(ValueError, match='66'):
        worker().configs(historical_config(), seed=67)


def test_manifest_checks_document_hash_and_actual_feature_hashes(tmp_path):
    feature = tmp_path / 'feature.bin'
    feature.write_bytes(b'actual feature')
    data = dict(data={'CMUMOSEI': dict(dataset_root='/dataset', feature_roots=['a', 't', 'v'],
        files={str(feature): hashlib.sha256(feature.read_bytes()).hexdigest()})})
    path = tmp_path / 'INPUTS.json'
    path.write_text(json.dumps(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest, proof = worker().load_manifest(path, digest)
    assert manifest == data['data']['CMUMOSEI']
    assert proof['files_verified'] == 1 and proof['sha256'] == digest
    feature.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='feature'):
        worker().load_manifest(path, digest)
    with pytest.raises(ValueError, match='manifest'):
        worker().load_manifest(path, 'wrong')


def test_loaders_use_complete_official_training_set_without_adapter_holdout():
    cfg = worker().configs(historical_config())
    seen = []
    trainer = SimpleNamespace(get_loaders=lambda *args, **kwargs: seen.append((args, kwargs)) or 'loaders')
    manifest = dict(feature_roots=['A', 'T', 'V'])
    assert worker().build_loaders(manifest, cfg, trainer) == 'loaders'
    assert seen == [(('A', 'T', 'V'), dict(dataset='CMUMOSEI', num_folder=1,
        batch_size=32, num_workers=0, seed=66, validation_fraction=.1,
        evaluation_protocol='official'))]


def test_model_factory_uses_mosei_fold1_not_mosi_hardcoded_fold5(monkeypatch):
    from experiments.osram_meld_flat_source_combinations_20261013 import evaluate
    cfg = worker().configs(historical_config())
    seen = []
    monkeypatch.setattr(evaluate, '_build_model', lambda cfg, dims, device:
        seen.append((cfg.dataset, cfg.seed, cfg.fold, dims, device)) or torch.nn.Linear(2, 1))
    model = worker().fresh_backbone(cfg, (512, 1024, 1024), torch.device('cpu'))
    assert seen == [('CMUMOSEI', 66, 1, (512, 1024, 1024), torch.device('cpu'))]
    assert model.training and all(p.requires_grad for p in model.parameters())


def test_cli_is_bounded_and_smoke_reaches_more_than_zero_rate_batch():
    parser = worker().argument_parser()
    args = parser.parse_args(['--phase', 'smoke', '--mode', 'joint', '--seed', '66',
        '--gpu-uuid', 'GPU-healthy', '--source-commit', 'sha', '--output', '/tmp/mosei'])
    assert args.seed == 66 and args.phase == 'smoke'
    assert worker().SMOKE_BATCHES == 3
    assert worker().EPOCHS == 60
    with pytest.raises(SystemExit):
        parser.parse_args(['--phase', 'train', '--mode', 'flat', '--seed', '68',
            '--gpu-uuid', 'GPU-healthy', '--source-commit', 'sha', '--output', '/tmp/mosei'])


def test_optimizer_matches_constant_mosei_backbone_and_router_rates():
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    cfg = worker().configs(historical_config())
    model = SimpleNamespace(backbone=torch.nn.Linear(2, 1), router=torch.nn.Linear(2, 1))
    opt = scalar.make_optimizer(model, cfg, 'joint')
    assert [(g['name'], g['lr'], g['weight_decay']) for g in opt.param_groups] == [
        ('backbone', .001, 1e-5), ('router', .0003, 1e-5)]
    assert {id(p) for g in opt.param_groups for p in g['params']} == {
        id(p) for module in (model.backbone, model.router) for p in module.parameters()}
