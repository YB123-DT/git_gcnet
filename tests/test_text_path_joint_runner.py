"""Training/recovery contract for the finite one-stage Router experiment."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from tests.test_text_path_joint import fixture


def test_optimizer_preserves_backbone_lr_and_gives_only_router_separate_lr():
    from experiments.osram_text_path_router_joint_20261019.run import make_optimizer, wrap_model
    backbone, router, _ = fixture()
    model = wrap_model(backbone, 'joint', router=router)
    cfg = SimpleNamespace(learning_rate=1e-5, weight_decay=1e-5)
    optimizer = make_optimizer(model, cfg, 'joint')
    assert [group['lr'] for group in optimizer.param_groups] == [1e-5, 3e-4]
    ids = [set(id(p) for p in group['params']) for group in optimizer.param_groups]
    assert not ids[0] & ids[1]
    assert ids[0] | ids[1] == set(id(p) for p in model.parameters())
    flat = wrap_model(copy.deepcopy(backbone), 'flat')
    assert len(make_optimizer(flat, cfg, 'flat').param_groups) == 1
    assert not flat.osram.osram_source_tracking


@pytest.mark.parametrize('mode', ('flat', 'joint'))
def test_noeligible_batch_still_updates_backbone_with_allvalid_task_loss(mode):
    from experiments.osram_text_path_router_joint_20261019.run import optimize_view, make_optimizer, wrap_model
    backbone, router, view = fixture(all_text=True)
    view['labels'] = torch.tensor([[0, 1, 2, 1], [1, 2, 0, 0]])
    cfg = SimpleNamespace(learning_rate=1e-3, weight_decay=1e-5, gradient_clip_norm=1.)
    model = wrap_model(backbone, mode, router=router if mode == 'joint' else None)
    optimizer = make_optimizer(model, cfg, mode)
    before = backbone.encoder.weight.detach().clone()
    torch.manual_seed(391)
    result = optimize_view(model, optimizer, view, cfg, mode)
    assert result['N'] == 7 and result['eligible_N'] == 0
    assert result['optimizer_steps'] == 1
    assert result['gradient_norm'] > 0
    assert not torch.equal(before, backbone.encoder.weight)


def test_flat_normal_ce_and_forward_match_original_contract():
    from experiments.osram_text_path_router_joint_20261019.run import forward_result, wrap_model
    backbone, _, view = fixture()
    backbone.eval()
    model = wrap_model(backbone, 'flat').eval()
    expected = backbone([view['incomplete']], view['availability'], view['qmask'], view['umask'], view['lengths'])[0]
    result = forward_result(model, view, 'flat')
    assert torch.equal(result['log_probs'], F.log_softmax(expected, -1))
    assert torch.equal(result['candidates'][..., 0, :], expected)
    assert not result['eligible'].any()


def test_full_checkpoint_recovery_preserves_model_optimizer_rng_and_identity():
    from experiments.osram_text_path_router_joint_20261019.run import (
        checkpoint_payload, restore, make_optimizer, wrap_model, optimize_view)
    from experiments.osram_tasr_20261018 import runner_helpers as h
    backbone, router, view = fixture()
    view['labels'] = torch.tensor([[0, 1, 2, 1], [1, 2, 0, 0]])
    cfg = SimpleNamespace(learning_rate=1e-3, weight_decay=1e-5, gradient_clip_norm=1.)
    model = wrap_model(backbone, 'joint', router=router)
    optimizer = make_optimizer(model, cfg, 'joint')
    optimize_view(model, optimizer, view, cfg, 'joint')
    identity = dict(version='unit', mode='joint')
    state = checkpoint_payload(model, optimizer, epoch=1, identity=identity, history=[{'loss': 1.}], best_key=(60., -1.))
    expected_state = {k: v.clone() for k, v in model.state_dict().items()}
    expected_draw = torch.rand(6)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.)
    restore(model, optimizer, state, identity)
    assert all(torch.equal(expected_state[k], v) for k, v in model.state_dict().items())
    assert torch.equal(torch.rand(6), expected_draw)
    with pytest.raises(ValueError, match='identity'):
        restore(model, optimizer, state, dict(version='different'))
    assert all('exp_avg' in item for item in optimizer.state.values())


def test_final_metrics_use_same_schema_for_flat_and_joint():
    from experiments.osram_text_path_router_joint_20261019.run import summarize_arrays
    import numpy as np
    full = np.array([[3., 0., 1.], [0., 4., 1.]])
    candidates = np.repeat(full[:, None], 3, axis=1)
    arrays = dict(labels=np.array([0, 2]), candidates=candidates,
                  log_probs_router=F.log_softmax(torch.tensor(full), -1).numpy(),
                  eligible=np.zeros(2, dtype=bool), weights=np.array([[1., 0., 0.]] * 2))
    result = summarize_arrays(arrays, classes=3)
    assert result['full']['ACC'] == result['router']['ACC'] == 50.
    assert result['eligible_N'] == 0
    assert result['paired_vs_full']['corrections'] == result['paired_vs_full']['harms'] == 0
