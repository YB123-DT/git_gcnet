"""Training-only text coordinates preserve the original inference model."""
from unittest.mock import patch

import pytest
import torch

from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
from test_pretrained_teacher import make, source_checkpoint
from test_future_state import batch, call


def config(**overrides):
    settings = dict(teacher_mode='pretrained-frozen', teacher_checkpoint='teacher.pt',
                    backbone_type='osram', osram_bidirectional=False, osram_write_step=.6)
    settings.update(overrides)
    return TrainConfig(**settings)


def test_default_target_space_and_parser():
    assert TrainConfig().target_space == 'all-modalities'
    assert TrainConfig().text_subspace_checkpoint is None
    args = build_parser().parse_args(['--target-space', 'predictable-subspace',
                                     '--text-subspace-checkpoint', 'r.pt',
                                     '--audio-feature', 'a', '--text-feature', 't',
                                     '--video-feature', 'v', '--output-dir', 'out'])
    assert args.target_space == 'predictable-subspace'
    assert args.text_subspace_checkpoint == 'r.pt'
    model = make()
    assert model.text_subspace is None
    assert not any(k.startswith('text_subspace.') for k in model.state_dict())


@pytest.mark.parametrize('extra', [dict(target_space='unknown'),
    dict(target_space='full-text', teacher_mode='ema', teacher_checkpoint=None),
    dict(target_space='full-text', training_objective='emotion-only'),
    dict(target_space='full-text', text_subspace_checkpoint='r.pt'),
    dict(target_space='predictable-subspace'),
    dict(target_space='full-text', jepa_weight=.2),
    dict(target_space='full-text', temperature=.1),
    dict(target_space='full-text', jepa_rate_weighting='rate'),
    dict(target_space='full-text', b2_base_checkpoint='b2.pt')])
def test_config_rejects_incompatible_text_modes(extra):
    with pytest.raises(ValueError):
        config(**extra)


@pytest.mark.parametrize('extra', [dict(target_space='unknown'),
    dict(target_space='full-text', teacher_mode='ema', teacher_checkpoint=None),
    dict(target_space='full-text', training_objective='emotion-only'),
    dict(target_space='full-text', text_subspace_checkpoint='r.pt'),
    dict(target_space='predictable-subspace'),
    dict(target_space='full-text', future_state_jepa=True),
    dict(target_space='full-text', write_state_completion=True),
    dict(target_space='full-text', complete_state_jepa=True),
    dict(target_space='full-text', completion_path='pre_osram_b2')])
def test_model_rejects_incompatible_text_modes(extra):
    settings = dict(teacher_mode='pretrained-frozen', teacher_checkpoint='teacher.pt')
    settings.update(extra)
    with pytest.raises(ValueError):
        make(**settings)


def test_full_text_preserves_shared_initialization_rng_and_inference(tmp_path):
    path, _ = source_checkpoint(tmp_path)
    old = make(teacher_mode='pretrained-frozen', teacher_checkpoint=str(path))
    rng = torch.get_rng_state()
    new = make(teacher_mode='pretrained-frozen', teacher_checkpoint=str(path),
               target_space='full-text')
    assert torch.equal(rng, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    x, a, u, q, mask = batch()
    old.eval(); new.eval()
    with patch.object(new.missing_predictor, 'forward', side_effect=AssertionError('aux')):
        assert torch.equal(call(old, x, a, u, q, mask)[0], call(new, x, a, u, q, mask)[0])


def subspace_checkpoint(tmp_path, teacher):
    from gcnet_missing_m3.text_subspace import TextSubspacePretrainer, save_subspace
    path = tmp_path / 'subspace.pt'
    save_subspace(path, TextSubspacePretrainer(latent_dim=8), teacher.teacher_integrity(),
                  epoch=1, validation_loss=.2, config={})
    return path


def test_subspace_is_frozen_rng_neutral_and_absent_from_inference(tmp_path):
    from gcnet_missing_m3.train_gcnet import _optimizer_parameter_groups
    from gcnet_missing_m3.text_subspace import text_jepa_loss
    path, _ = source_checkpoint(tmp_path)
    settings = dict(teacher_mode='pretrained-frozen', teacher_checkpoint=str(path))
    old = make(**settings)
    r_path = subspace_checkpoint(tmp_path, old)
    old = make(**settings)
    rng = torch.get_rng_state()
    new = make(**settings, target_space='predictable-subspace',
               text_subspace_checkpoint=str(r_path))
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    before = new.text_subspace_integrity()
    assert before == new.text_subspace_provenance['projector_sha256']
    new.train()
    assert not new.text_subspace.training
    groups, _ = _optimizer_parameter_groups(new, config())
    optimized = {id(p) for group in groups for p in group['params']}
    assert all(not p.requires_grad and id(p) not in optimized for p in new.text_subspace.parameters())
    optimizer = torch.optim.Adam(groups)
    x, a, u, q, mask = batch()
    _, _, _, pred = new([x*mask], a, q, u, [3, 2], predict_missing=True)
    loss = text_jepa_loss(pred, new.encode_teacher_targets([x]), new.text_subspace)
    loss.total.backward()
    optimizer.step()
    assert before == new.text_subspace_integrity()
    old.load_state_dict({k: v for k, v in new.state_dict().items()
                         if not k.startswith('text_subspace.')}, strict=True)
    old.eval(); new.eval()
    with (patch.object(new.text_subspace, 'forward', side_effect=AssertionError('R')),
          patch.object(new.missing_predictor, 'forward', side_effect=AssertionError('predictor')),
          patch.object(new.teacher, 'forward', side_effect=AssertionError('teacher'))):
        assert torch.equal(call(old, x, a, u, q, mask)[0], call(new, x, a, u, q, mask)[0])


def test_checkpoint_saves_text_subspace_provenance(tmp_path):
    from gcnet_missing_m3.train_gcnet import _save_best_checkpoint
    path = tmp_path / 'best.pt'
    provenance = {'projector_sha256': 'rhash'}
    _save_best_checkpoint(path, {}, config(), 1, .3,
                          text_subspace_provenance=provenance)
    saved = torch.load(path, weights_only=False)
    assert saved['text_subspace_provenance'] == provenance
    assert saved['config']['target_space'] == 'all-modalities'


@pytest.mark.parametrize('target_space', ['full-text', 'predictable-subspace'])
def test_trainer_dispatches_text_only_loss_and_counts(monkeypatch, target_space):
    from gcnet_missing_m3 import train_gcnet as tr
    from gcnet_missing_m3.mixed_rate import MISSING_RATES
    from test_missing_m3 import _LifecycleModel, _CountingOptimizer, _install_train_epoch_lifecycle_fakes
    _install_train_epoch_lifecycle_fakes(monkeypatch)
    model = _LifecycleModel()
    model.text_subspace = torch.nn.Linear(1, 32, bias=False).requires_grad_(False) if target_space == 'predictable-subspace' else None
    def targets(complete):
        model.teacher_calls += 1
        return {'text': complete[0]}
    model.encode_teacher_targets = targets
    monkeypatch.setattr(tr, 'missing_m3_loss', lambda *a, **k: pytest.fail('legacy loss'))
    cfg = config(target_space=target_space,
                 text_subspace_checkpoint='r.pt' if model.text_subspace is not None else None,
                 train_rate_mode='fixed', fixed_missing_rate=.6)
    result = tr.train_epoch(model, [['batch']], _CountingOptimizer(model.weight), cfg,
                           {r: r for r in MISSING_RATES}, 0, (1, 1, 1), torch.device('cpu'))
    assert result['jepa_target_count'] == 2
    assert result['rate_jepa_target_counts']['0.6'] == 2
    assert model.ema_calls == 0
