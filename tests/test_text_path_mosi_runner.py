"""Scalar loss, original LR transition, metadata and joint checkpoint contracts."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_text_path_joint import fixture


def runner():
    from experiments.osram_text_path_router_joint_mosi_20261019 import run
    return run


def test_original_lr_continuation_is_one_joint_stage_with_adam_state_retained():
    r = runner()
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([dict(params=[parameter], name='backbone', lr=.001),
                                  dict(params=[torch.nn.Parameter(torch.ones(1))], name='router', lr=.0003)])
    for epoch, expected in ((0, [.001, .0003]), (99, [.001, .0003]), (100, [.0001, .00003]), (149, [.0001, .00003])):
        r.set_epoch_lrs(optimizer, epoch)
        assert [group['lr'] for group in optimizer.param_groups] == expected
    with pytest.raises(ValueError):
        r.set_epoch_lrs(optimizer, 150)


def test_metadata_keeps_continuous_mosi_labels_and_batch_major_identity():
    data = [None] * 10
    data[6] = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
    data[7] = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    data[8] = torch.tensor([[-.25, 0., 1.7], [.75, -2.1, 999.]])
    data[-1] = ['first', 'second']
    meta = runner().metadata(data)
    assert np.issubdtype(meta['labels'].dtype, np.floating)
    np.testing.assert_array_equal(meta['labels'], np.array([-.25, 0., 1.7, .75, -2.1], dtype=np.float32))
    assert meta['conversation_ids'].tolist() == ['first'] * 3 + ['second'] * 2
    assert meta['utterance_indices'].tolist() == [0, 1, 2, 0, 1]


def test_scalar_loss_calls_original_task_loss_and_keeps_neutral_training_samples():
    from gcnet_missing_m3 import train_gcnet as trainer
    cfg = SimpleNamespace(dataset='CMUMOSI', mosi_task_mode='regression', task_regression_loss='mse', task_smooth_l1_beta=1.)
    prediction = torch.tensor([[[.3], [.2]], [[-.8], [99.]]], requires_grad=True)
    view = dict(labels=torch.tensor([[0., -.3], [.8, 999.]]), umask=torch.tensor([[1., 1.], [1., 0.]]))
    expected = ((.3 - 0.)**2 + (-.8 + .3)**2 + (.2 - .8)**2) / 3
    actual = runner().scalar_loss(trainer, cfg, prediction, view)
    assert float(actual) == pytest.approx(expected)
    actual.backward()
    assert prediction.grad[0, 0].abs().sum() > 0
    assert prediction.grad[1, 1] == 0


def test_selection_is_single_validation_eight_rate_wf1_then_lower_mse():
    validation = {str(i / 10): dict(router=dict(WF1=70. + i, MSE=1. + i / 10)) for i in range(8)}
    assert runner().selection_key(validation) == pytest.approx((73.5, -1.35))
    with pytest.raises(ValueError, match='eight'):
        runner().selection_key({'0.0': validation['0.0']})


def test_all_eight_test_mask_hashes_match_original_time_major_valid_order():
    import hashlib
    r = runner()
    availability = torch.tensor([[[1., 1., 0.], [0., 1., 0.]],
                                 [[1., 0., 0.], [0., 0., 0.]]])
    banks = {str(i / 10): [dict(availability=availability)] for i in range(8)}
    sha = lambda value: hashlib.sha256(value.numpy().tobytes()).hexdigest()
    expected = {rate: sha(availability[availability.bool().any(-1)]) for rate in banks}
    actual = r.verify_test_mask_hashes(banks, expected, SimpleNamespace(_sha256_tensor=sha))
    assert actual == expected
    wrong = dict(expected, **{'0.7': 'wrong'})
    with pytest.raises(ValueError, match='0.7'):
        r.verify_test_mask_hashes(banks, wrong, SimpleNamespace(_sha256_tensor=sha))


def test_scalar_metric_filter_and_corrections_exclude_neutral_without_rounding_labels():
    values = dict(labels=np.array([-.2, .2, 0., .8]),
                  candidates=np.array([[[-.4], [.1], [-1.]], [[-.3], [.4], [.5]],
                                       [[.1], [0.], [-.2]], [[.3], [.4], [-.5]]]),
                  prediction=np.array([[-.2], [.2], [-.3], [-.1]]),
                  eligible=np.array([True, True, False, True]), weights=np.ones((4, 3)) / 3)
    from gcnet_missing_m3 import train_gcnet as trainer
    metrics = runner().summarize_arrays(values, trainer)
    assert metrics['full']['ACC'] == pytest.approx(200 / 3)
    assert metrics['router']['ACC'] == pytest.approx(200 / 3)
    assert metrics['paired_vs_full'] == dict(corrections=1, harms=1, N=3)
    assert metrics['router']['MSE'] == pytest.approx(float(np.mean((values['labels'] - values['prediction'].ravel())**2)))


def test_joint_scalar_training_and_checkpoint_recovery_even_without_eligible_rows():
    r = runner()
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_text_path_router_joint_mosi_20261019.router import RegressionTextPathRouter
    from experiments.osram_text_path_router_joint_20261019.run import checkpoint_payload, restore
    backbone, _, view = fixture(all_text=True)
    backbone.smax_fc = torch.nn.Linear(8, 1)
    view['labels'] = torch.tensor([[0., -.25, 1., .5], [-1., 1.5, 0., 0.]])
    cfg = SimpleNamespace(dataset='CMUMOSI', mosi_task_mode='regression', task_regression_loss='mse',
                          task_smooth_l1_beta=1., weight_decay=1e-5, learning_rate=.001, gradient_clip_norm=1.)
    router = RegressionTextPathRouter(local_dim=4, context_dim=2, projection_dim=3, hidden_dim=5)
    model = r.wrap_model(backbone, 'joint', router=router)
    optimizer = r.make_optimizer(model, cfg, 'joint')
    before = backbone.encoder.weight.detach().clone()
    record = r.optimize_view(model, optimizer, view, cfg, 'joint', trainer)
    assert record['N'] == 7 and record['eligible_N'] == 0
    assert record['gradients']['backbone']['nonzero'] > 0
    assert not torch.equal(before, backbone.encoder.weight)
    identity = dict(mode='joint', seed=66)
    saved = checkpoint_payload(model, optimizer, epoch=1, identity=identity, history=[record], best_key=(70., -1.))
    state = copy.deepcopy(model.state_dict())
    draw = torch.rand(4)
    with torch.no_grad():
        backbone.encoder.weight.zero_()
    restore(model, optimizer, saved, identity)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert torch.equal(torch.rand(4), draw)
