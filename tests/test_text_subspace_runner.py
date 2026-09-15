import importlib.util

import pytest
import torch


def test_runner_available():
    assert importlib.util.find_spec('experiments.osram_text_subspace_20260915.run') is not None


def test_fit_rejects_test_split_and_saves_validation_selection(tmp_path):
    from experiments.osram_text_subspace_20260915.run import fit_stage1
    from gcnet_missing_m3.text_subspace import TextSubspacePretrainer
    def rows():
        return [{**{m: torch.randn(8, 256) for m in ('audio', 'text', 'visual')}, 'labels': torch.randn(8)}]
    data = {'train': rows(), 'validation': rows()}
    model = TextSubspacePretrainer()
    with pytest.raises(ValueError, match='train/validation'):
        fit_stage1(model, {**data, 'test': rows()}, tmp_path/'bad', 'hash',
                   epochs=1, learning_rate=.001, weight_decay=0, beta=1, device='cpu', smoke=True)
    report = fit_stage1(model, data, tmp_path/'good', 'hash', epochs=1, learning_rate=.001,
                        weight_decay=0, beta=1, device='cpu', smoke=True)
    assert report['selection_split'] == 'validation' and report['optimizer_steps'] == 1
    cp = torch.load(tmp_path/'good'/'subspace.pt', weights_only=False)
    assert cp['epoch'] == report['selected_epoch'] == 1
    assert cp['config']['smoke_only']
    assert set(report['selected_metrics']) == {'train', 'validation'}
    assert len(report['selected_metrics']['validation']['target']['per_dim_std']) == 32


def test_stage2_configuration_is_per_rate_and_fresh():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_text_subspace_20260915.run import stage2_config
    base = TrainConfig(dataset='CMUMOSI', backbone_type='osram', osram_bidirectional=False,
                       osram_write_step=.6, osram_output_dim=700, osram_num_heads=8)
    cfg = stage2_config(base, 'teacher.pt', 'subspace.pt', 'predictable-subspace')
    assert cfg.checkpoint_selection == 'test-oracle-per-rate'
    assert cfg.train_rate_mode == 'cyclic' and cfg.epochs == base.epochs
    assert cfg.initial_backbone_checkpoint is None
    assert cfg.jepa_weight == .1 and cfg.target_space == 'predictable-subspace'


def test_smoke_has_hard_two_update_limit(tmp_path):
    from experiments.osram_text_subspace_20260915.run import fit_stage1
    from gcnet_missing_m3.text_subspace import TextSubspacePretrainer
    row = {**{m: torch.randn(8, 256) for m in ('audio', 'text', 'visual')}, 'labels': torch.randn(8)}
    report = fit_stage1(TextSubspacePretrainer(), {'train': [row]*4, 'validation': [row]},
                        tmp_path/'bounded', 'h', epochs=1, learning_rate=.001,
                        weight_decay=0, beta=1, device='cpu', smoke=True)
    assert report['optimizer_steps'] == 2
    with pytest.raises(ValueError, match='one epoch'):
        fit_stage1(TextSubspacePretrainer(), {'train': [row], 'validation': [row]},
                   tmp_path/'too_long', 'h', epochs=2, learning_rate=.001,
                   weight_decay=0, beta=1, device='cpu', smoke=True)
