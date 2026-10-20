from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_mosei_scalar_loss_uses_real_dataset_and_original_task_mse():
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    cfg = SimpleNamespace(dataset='CMUMOSEI', mosi_task_mode='regression',
        task_regression_loss='mse', task_smooth_l1_beta=1.)
    prediction = torch.tensor([[[.3]], [[-.8]], [[4.]]], requires_grad=True)
    view = dict(labels=torch.tensor([[.7, -1., 0.]]), umask=torch.tensor([[1., 1., 0.]]))
    expected = trainer._task_loss('CMUMOSEI', prediction, view['labels'], view['umask'])
    actual = scalar.scalar_loss(trainer, cfg, prediction, view)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad[:2].ne(0).any()
    assert prediction.grad[2].eq(0).all()
    cfg.dataset = 'IEMOCAPSix'
    with pytest.raises(ValueError):
        scalar.scalar_loss(trainer, cfg, prediction, view)


def test_mosei_metrics_keep_neutral_filter_and_mosi_default_unchanged():
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    labels = np.array([-1., 0., 1., -2.], dtype=np.float32)
    pred = np.array([-.7, 1., -.2, .1], dtype=np.float32)
    arrays = dict(labels=labels, prediction=pred[:, None],
        candidates=np.repeat(pred[:, None, None], 3, axis=1), eligible=np.zeros(4, dtype=bool),
        weights=np.tile([1., 0., 0.], (4, 1)))
    actual = scalar.summarize_arrays(arrays, trainer, dataset='CMUMOSEI')
    expected = trainer._metrics('CMUMOSEI', labels, pred, 'regression')
    assert actual['router']['ACC'] == 100 * expected['accuracy']
    assert actual['router']['WF1'] == 100 * expected['weighted_f1']
    assert actual['paired_vs_full']['N'] == 3
    assert scalar.summarize_arrays(arrays, trainer) == scalar.summarize_arrays(arrays, trainer, dataset='CMUMOSI')
