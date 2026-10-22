from dataclasses import asdict, replace
import inspect

import pytest


def baseline_config(seed):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    return TrainConfig(dataset='CMUMOSI', fold=1, seed=seed, backbone_type='osram',
        osram_bidirectional=False, osram_write_step=.6, epochs=150,
        learning_rate=3e-4, missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
        training_objective='emotion-only', osram_output_dim=1600,
        osram_num_heads=8, osram_key_dim=64, osram_value_dim=64)


def test_supplement_changes_only_reviewed_capacity_fields_and_keeps_run_seed():
    from experiments.osram_mosi_v3_capacity_3seed_20261022.worker import candidate_config
    allowed = {'osram_adapter_hidden_dim', 'osram_source_role_v3',
               'osram_source_role_v3_dim', 'osram_source_role_v3_pair_hidden'}
    for seed in (67, 68):
        cfg = baseline_config(seed)
        candidate = candidate_config(cfg, 'f512_v128')
        assert candidate.seed == seed
        assert candidate.osram_adapter_hidden_dim == 512
        assert candidate.osram_source_role_v3
        assert (candidate.osram_source_role_v3_dim, candidate.osram_source_role_v3_pair_hidden) == (128, 256)
        assert {k for k, v in asdict(cfg).items() if v != asdict(candidate)[k]} == allowed
        plain = candidate_config(cfg, 'f512_plain')  # Internal zero-init parity model only.
        assert plain.seed == seed and not plain.osram_source_role_v3
    for cfg, mode in ((baseline_config(66), 'f512_v128'),
                      (baseline_config(67), 'f1024_v128'),
                      (replace(baseline_config(67), learning_rate=1e-3), 'f512_v128')):
        with pytest.raises(ValueError):
            candidate_config(cfg, mode)


def test_supplement_uses_completed_same_seed_flat_references_and_records_new_sources():
    from experiments.osram_mosi_v3_capacity_3seed_20261022.worker import reference_for_seed, source_catalog
    for seed in (67, 68):
        assert str(reference_for_seed(seed)) == (
            '/data2/yb/remote_experiments/osram_mosi_mid_lr_3seed_20261021/'
            f'execution_v1/runs/MOSI/seed_{seed}/flat')
    with pytest.raises(ValueError):
        reference_for_seed(66)
    sources = source_catalog(['original-source.py'])
    assert 'original-source.py' in sources
    for name in ('worker.py', 'run.py', 'report.py'):
        assert 'experiments/osram_mosi_v3_capacity_3seed_20261022/' + name in sources


def test_default_worker_still_restricts_six_modes_to_seed66(capsys):
    from experiments.osram_mosi_v3_capacity_20261022 import worker as base
    defaults = inspect.signature(base.main).parameters
    assert defaults['allowed_seeds'].default == (66,)
    assert defaults['allowed_modes'].default == base.MODES
    assert defaults['config_factory'].default is base.candidate_config
    with pytest.raises(SystemExit) as stopped:
        base.main(['--help'])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert '--seed {66}' in help_text
    assert all(mode in help_text for mode in base.MODES)


def test_supplement_cli_rejects_seed66_and_plain_training(tmp_path):
    from experiments.osram_mosi_v3_capacity_3seed_20261022.worker import main
    required = ['--phase', 'train', '--gpu-uuid', 'GPU-healthy', '--source-commit', 'abc',
                '--output', str(tmp_path / 'not-created')]
    for mode, seed in (('f512_v128', '66'), ('f512_plain', '67'), ('f1024_v128', '68')):
        with pytest.raises(SystemExit) as stopped:
            main(['--mode', mode, '--seed', seed] + required)
        assert stopped.value.code == 2
    assert not (tmp_path / 'not-created').exists()


def test_supplement_injects_narrow_scope_into_existing_worker(monkeypatch):
    from experiments.osram_mosi_v3_capacity_3seed_20261022 import worker
    calls = []
    monkeypatch.setattr(worker.base, 'main', lambda *args, **kwargs: calls.append((args, kwargs)))
    argv = ['--mode', 'f512_v128', '--seed', '67']
    worker.main(argv)
    args, kwargs = calls[0]
    assert args == (argv,)
    assert kwargs == dict(allowed_seeds=(67, 68), allowed_modes=('f512_v128',),
        config_factory=worker.candidate_config, reference_factory=worker.reference_for_seed,
        source_catalog_factory=worker.source_catalog)
