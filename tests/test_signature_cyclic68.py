"""The selected TEST rate never selects the stage-two TRAIN rate."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F


@pytest.mark.parametrize('epoch', [0, 7])
@pytest.mark.parametrize('test_rate', ['0.0', '0.7'])
def test_train_epoch_cycles_requested_protocol_with_frozen_parent(monkeypatch, epoch, test_rate):
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule, MISSING_RATES
    from experiments.osram_signature_cyclic68_20261025.train import train_epoch
    schedules = {rate: object() for rate in MISSING_RATES}
    seen, losses, clips, parent_modes = [], [], [], []
    frame = SimpleNamespace(value=torch.ones(2, 1, 1))
    parent = nn.Linear(1, 1).requires_grad_(False).train()
    initial = {name: p.detach().clone() for name,p in parent.named_parameters()}

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, frame):
            return self.weight.expand_as(frame.value)

    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda value: None)

    def make_view(data, schedule, current_epoch, dimensions):
        seen.append((next(rate for rate,value in schedules.items() if value is schedule),current_epoch))
        return dict(incomplete=torch.zeros(2,1,1), availability=torch.ones(2,1,3),
                    qmask=torch.zeros(1,2), umask=torch.ones(1,2), lengths=[2],
                    labels=torch.zeros(1,2), conversation_ids=[str(data)])

    def task_loss(dataset, prediction, labels, umask, mode, loss_type, beta):
        assert (dataset,mode,loss_type) == ('CMUMOSI','regression','mse')
        assert torch.equal(labels,torch.zeros_like(labels))  # Gold zero remains in ordinary MSE.
        loss = F.mse_loss(prediction[...,0].T,labels)
        losses.append(loss.detach().item())
        return loss

    original_clip = torch.nn.utils.clip_grad_norm_
    def clip(parameters, maximum, *args, **kwargs):
        params = list(parameters)
        assert maximum == .1
        result = original_clip(params,maximum,*args,**kwargs)
        clips.append(torch.sqrt(sum(p.grad.square().sum() for p in params if p.grad is not None)).item())
        return result
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',clip)

    class Reader:
        def __init__(self):
            self.parent = parent

        def __call__(self,*args):
            parent_modes.append(self.parent.training)
            return frame

        def decode_delta(self,frame,delta):
            return frame.value + delta

    cfg = SimpleNamespace(dataset='CMUMOSI',seed=68,train_rate_mode='cyclic',fixed_missing_rate=None,
        train_missing_rates=MISSING_RATES,mosi_task_mode='regression',task_regression_loss='mse',
        task_smooth_l1_beta=1.,gradient_clip_norm=.1)
    trainer = SimpleNamespace(_schedules=lambda cfg,split:schedules,
        _move_batch=lambda raw,device:raw,_prepare_view=make_view,_task_loss=task_loss)
    bundle = dict(trainer=trainer,cfg=cfg,train_loader=Loader([1,2,3]),dimensions=(1,1,1))
    block = Block()
    row = train_epoch(bundle,Reader(),block,torch.optim.SGD(block.parameters(),lr=.01),epoch,test_rate,False)
    expected = [BalancedBatchRateSchedule(MISSING_RATES).rate_for(epoch,index) for index in range(3)]
    assert seen == [(rate,epoch) for rate in expected]
    assert len(set(expected)) == 3
    assert row['training_rates'] == expected and row['optimizer_steps'] == 3
    assert row['task_MSE'] == pytest.approx(sum(losses)/3)
    assert losses[0] == 4.
    assert len(clips) == 3 and all(norm <= .100001 for norm in clips)
    assert parent_modes == [False]*3
    assert all(p.grad is None and not p.requires_grad and torch.equal(p,initial[name])
               for name,p in parent.named_parameters())
    assert block.weight.item() < 1.


def test_cyclic_extension_scope_is_only_mosi68():
    from experiments.osram_signature_cyclic68_20261025.run import validate_scope
    validate_scope('CMUMOSI',68)
    for dataset,seed in [('CMUMOSI',66),('CMUMOSI',67),('CMUMOSEI',68),('CMUMOSEI',66)]:
        with pytest.raises(ValueError):
            validate_scope(dataset,seed)
