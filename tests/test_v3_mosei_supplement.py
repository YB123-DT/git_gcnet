"""The supplementary V3 seeds must match the completed 1e-3 Flat runs."""
from dataclasses import asdict, replace
import importlib
import inspect

import pytest
import torch


def worker():
    name = 'experiments.osram_v3_three_seed_20261021.mosei_worker'
    assert importlib.util.find_spec(name.rsplit('.', 1)[0]) is not None, 'Supplementary V3 entry point is missing'
    return importlib.import_module(name)


def config():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    return TrainConfig(dataset='CMUMOSEI', seed=66, fold=1, epochs=60,
        missing_mask_protocol='gcnet-random-mask-v1', backbone_type='osram',
        training_objective='emotion-only', checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1', osram_output_dim=1600,
        osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
        osram_write_step=.6, osram_bidirectional=False, osram_readout_fusion='flat')


@pytest.mark.parametrize('seed', (67, 68))
def test_only_seed_and_v3_switch_change(seed):
    cfg = worker().candidate_config(config(), seed)
    assert asdict(cfg) == dict(asdict(config()), seed=seed, osram_source_role_v3=True)


@pytest.mark.parametrize('change', [dict(learning_rate=1e-4), dict(epochs=100),
    dict(osram_source_role_v3=True), dict(osram_source_role=True),
    dict(missing_mask_protocol='conversation-mask-v2')])
def test_reject_wrong_flat_protocol(change):
    with pytest.raises(ValueError):
        worker().candidate_config(replace(config(), **change), 67)


def test_reference_paths_point_to_completed_matched_flat_not_old_low_lr():
    w = worker()
    for seed in (67, 68):
        path = str(w.flat_reference(seed))
        assert 'osram_mosei_nested_gcnet_3seed_20261021/execution_v1/runs/CMUMOSEI' in path
        assert path.endswith(f'/seed_{seed}/flat')
    with pytest.raises(ValueError):
        w.flat_reference(66)


def test_setup_rebuilds_both_models_and_training_loader_for_actual_seed(monkeypatch):
    w = worker()
    from experiments.osram_mosei_gcnet_router_20261020 import worker as old
    from experiments.osram_text_path_router_joint_20261019 import run as protocol
    calls = []

    class Model(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            torch.manual_seed(cfg.seed)
            self.original = torch.nn.Linear(2, 2)
            self.osram = torch.nn.Module()
            if cfg.osram_source_role_v3:
                self.osram.source_role_v3 = torch.nn.Linear(2, 2)

    monkeypatch.setattr(old, 'setup', lambda uuid, output, seed:
        dict(cfg=config(), model=Model(config()), dimensions=(2, 2, 2),
             train_loader=object(), split={'accepted': 66}, data_files={'file': 'digest'}))
    def fresh(cfg, dimensions, device):
        calls.append((cfg.seed, cfg.osram_source_role_v3))
        return Model(cfg)
    monkeypatch.setattr(old, 'fresh_backbone', fresh)
    monkeypatch.setattr(protocol, 'seed_training_loader', lambda bundle, seed:
        dict(training_seed=seed, split_seed=66, evaluation_seed=66))
    for seed in (67, 68):
        bundle, baseline = w.setup('CMUMOSEI', 'GPU-healthy', '/unused', seed)
        assert bundle['cfg'].seed == seed
        assert calls[-2:] == [(seed, False), (seed, True)]
        assert bundle['seed_protocol'] == dict(training_seed=seed, split_seed=66, evaluation_seed=66)
        assert w.v3.original_digest(bundle['model']) == w.v3.original_digest(baseline)
        assert bundle['initial_backbone_sha256'] == w.v3.original_digest(baseline)
        assert bundle['split'] == {'accepted': 66}


def test_real_cpu_factory_preserves_original_parameters_but_changes_training_seed():
    from experiments.osram_mosei_gcnet_router_20261020 import worker as old
    w = worker(); digests = []
    for seed in (67, 68):
        cfg = w.candidate_config(config(), seed)
        flat = old.fresh_backbone(replace(cfg, osram_source_role_v3=False),
                                  (512, 1024, 1024), torch.device('cpu'))
        original = w.v3.original_digest(flat)
        del flat
        model = old.fresh_backbone(cfg, (512, 1024, 1024), torch.device('cpu'))
        assert w.v3.original_digest(model) == original
        assert model.osram.source_role_v3 is not None
        digests.append(original)
        del model
    assert digests[0] != digests[1]


def test_cli_bounds_and_reused_loop_receive_real_seed_and_same_seed_reference(monkeypatch):
    w = worker(); seen = []
    monkeypatch.setattr(w.v3, 'main', lambda argv, **kwargs: seen.append((argv, kwargs)))
    argv = ['--phase', 'train', '--seed', '68', '--gpu-uuid', 'GPU-healthy',
        '--source-commit', 'sha', '--output', '/tmp/unused', '--resume']
    w.main(argv)
    forwarded, options = seen.pop()
    assert forwarded[forwarded.index('--seed') + 1] == '68'
    assert '--resume' in forwarded
    assert options['seeds'] == (67, 68)
    assert options['setup_factory'] is w.setup
    assert options['references']['CMUMOSEI'] == w.flat_reference(68)
    for seed in ('66', '69'):
        with pytest.raises(SystemExit):
            w.argument_parser().parse_args([*argv[:3], seed, *argv[4:]])


def test_original_loop_defaults_remain_seed66_and_all_stochasticity_uses_argument():
    from experiments.osram_source_role_v3_20261021 import worker as v3
    signature = inspect.signature(v3.main)
    assert 'seeds' in signature.parameters, 'V3 loop lacks bounded seed extension'
    assert signature.parameters['seeds'].default == (66,)
    source = inspect.getsource(v3.main)
    assert 'reset_training_stochasticity(args.seed, cfg.fold)' in source
    assert source.count('seed=args.seed') >= 2
    assert 'check_training_parity(row, reference_history[epoch])' in source


def test_source_catalog_includes_new_factory_and_seed_protocol_helpers():
    paths = worker().source_catalog(())
    assert 'experiments/osram_v3_three_seed_20261021/mosei_worker.py' in paths
    assert 'experiments/osram_mosei_nested_gcnet_3seed_20261021/worker.py' in paths
    assert 'experiments/osram_text_path_router_joint_20261019/run.py' in paths
