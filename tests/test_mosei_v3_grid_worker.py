from dataclasses import asdict, replace
import inspect

import pytest


def baseline_config():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    return TrainConfig(dataset='CMUMOSEI', fold=1, backbone_type='osram',
        osram_bidirectional=False, osram_write_step=.6, epochs=60,
        missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
        training_objective='emotion-only', osram_output_dim=1600,
        osram_num_heads=8, osram_key_dim=64, osram_value_dim=64)


def test_four_new_combinations_change_only_authorized_widths_and_keep_protocol():
    from experiments.osram_mosei_v3_grid_20261022.worker import MODES, candidate_config, expected_counts
    assert MODES == {'f512_plain': (512, 0, 0), 'f512_v128': (512, 128, 256),
                     'f512_v256': (512, 256, 512), 'f1024_v128': (1024, 128, 256)}
    expected = {'f512_plain': (7032929, 3058240, 0),
                'f512_v128': (7864639, 3058240, 831710),
                'f512_v256': (9349567, 3058240, 2316638),
                'f1024_v128': (10912575, 6106176, 831710)}
    cfg = baseline_config()
    allowed = {'osram_adapter_hidden_dim', 'osram_source_role_v3',
               'osram_source_role_v3_dim', 'osram_source_role_v3_pair_hidden'}
    for mode, (flat, width, pair) in MODES.items():
        candidate = candidate_config(cfg, mode)
        assert candidate.osram_adapter_hidden_dim == flat
        assert candidate.osram_source_role_v3 == bool(width)
        assert (candidate.osram_source_role_v3_dim, candidate.osram_source_role_v3_pair_hidden) == (width or 64, pair or 128)
        assert {k for k, v in asdict(cfg).items() if v != asdict(candidate)[k]} <= allowed
        assert expected_counts(mode) == dict(zip(('total', 'flat_adapter', 'v3'), expected[mode]))


def test_grid_rejects_reused_arms_other_seeds_and_protocol_changes(tmp_path):
    from experiments.osram_mosei_v3_grid_20261022.worker import candidate_config, expected_counts, main
    cfg = baseline_config()
    for mode in ('small_flat', 'wide_v3', 'f1024_plain', 'f1024_v256', 'extra_hpo'):
        with pytest.raises(ValueError):
            candidate_config(cfg, mode)
        with pytest.raises(ValueError):
            expected_counts(mode)
    for changed in (replace(cfg, seed=67), replace(cfg, learning_rate=1e-4),
                    replace(cfg, missing_mask_protocol='conversation-mask-v2'),
                    replace(cfg, osram_source_role_v3=True)):
        with pytest.raises(ValueError):
            candidate_config(changed, 'f512_v128')
    required = ['--phase', 'train', '--gpu-uuid', 'GPU-healthy', '--source-commit', 'abc',
                '--output', str(tmp_path / 'not-created')]
    for mode, seed in (('f512_v128', '67'), ('small_flat', '66'), ('f1024_v256', '66')):
        with pytest.raises(SystemExit) as stopped:
            main(['--mode', mode, '--seed', seed] + required)
        assert stopped.value.code == 2
    assert not (tmp_path / 'not-created').exists()


def test_original_two_arm_defaults_and_counts_are_unchanged(capsys):
    from experiments.osram_mosei_v3_capacity_20261022 import worker as base
    defaults = inspect.signature(base.main).parameters
    assert defaults['allowed_modes'].default == ('small_flat', 'wide_v3')
    assert defaults['config_factory'].default is base.candidate_config
    assert defaults['expected_counts_factory'].default is base.expected_counts
    assert defaults['source_catalog_additions'].default == ()
    assert defaults['capture_first_epoch_only'].default is False
    assert base.expected_counts('small_flat') == dict(total=10080865, flat_adapter=6106176, v3=0)
    assert base.expected_counts('wide_v3') == dict(total=12397503, flat_adapter=6106176, v3=2316638)
    with pytest.raises(SystemExit) as stopped:
        base.main(['--help'])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert '--mode {small_flat,wide_v3}' in help_text and '--seed {66}' in help_text


def test_gradient_hook_removal_keeps_scan_counter_and_stops_input_capture():
    import torch
    from experiments.osram_mosei_v3_capacity_20261022.worker import GradientCapture

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, local, source_base, source_gap):
            return self.linear(local + source_base + source_gap)

    module = Module()
    capture = GradientCapture(module)
    scans = []
    scan_hook = module.register_forward_hook(lambda *unused: scans.append(True))
    inputs = [torch.ones(1, 2, requires_grad=True) for _ in range(3)]
    module(*inputs).sum().backward()
    assert all(capture.observed[key] for key in ('linear.weight', 'linear.bias', 'input.source_base', 'input.source_gap'))
    capture.remove()
    assert not capture.handles and not module._forward_pre_hooks
    assert all(not parameter._backward_hooks for parameter in module.parameters())
    assert all(not value._backward_hooks for value in inputs)
    capture.observed.clear()
    module(*inputs).sum().backward()
    assert len(scans) == 2 and not capture.observed
    scan_hook.remove()


def test_grid_injects_only_four_new_arms_and_records_its_sources(monkeypatch):
    from experiments.osram_mosei_v3_grid_20261022 import worker
    calls = []
    monkeypatch.setattr(worker.base, 'main', lambda *args, **kwargs: calls.append((args, kwargs)))
    argv = ['--mode', 'f512_v128']
    worker.main(argv)
    args, kwargs = calls[0]
    assert args == (argv,)
    assert kwargs == dict(allowed_modes=worker.MODES, config_factory=worker.candidate_config,
        expected_counts_factory=worker.expected_counts,
        source_catalog_additions=worker.SOURCE_CATALOG_ADDITIONS, capture_first_epoch_only=True)
    for name in ('worker.py', 'run.py', 'report.py'):
        assert 'experiments/osram_mosei_v3_grid_20261022/' + name in worker.SOURCE_CATALOG_ADDITIONS
