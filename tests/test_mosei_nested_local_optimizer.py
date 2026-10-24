"""Real Adam checks for the new opt-in parameter partition (CPU only)."""
import copy
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
from torch import nn
from experiments.osram_nested_cmerc_tuning_20261023.worker import make_task_optimizer
from experiments.osram_text_path_router_joint_20261019 import run as task


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3,2)
        self.osram = nn.Module()
        self.osram.meaningful_block = nn.Sequential(nn.Linear(3,4),nn.Tanh(),nn.Linear(4,2))
    def forward(self,x):
        return self.backbone(x)+self.osram.meaningful_block(x)


def cfg():
    return SimpleNamespace(dataset='CMUMOSEI',seed=66,fold=1,
        osram_meaningful_block='nested_dim32',learning_rate=1e-4,weight_decay=1e-5)


def step(model,opt,x):
    opt.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    opt.step()


def test_only_nested_update_scale_changes_and_adam_resume_is_exact():
    torch.manual_seed(66)
    low=Tiny();high=copy.deepcopy(low);before=copy.deepcopy(low.state_dict())
    x=torch.randn(8,3)
    low_opt=make_task_optimizer(low,cfg(),task,5e-5)
    high_opt=make_task_optimizer(high,cfg(),task,2e-4)
    step(low,low_opt,x);step(high,high_opt,x)
    assert torch.equal(low.backbone.weight,high.backbone.weight)
    for name,p in low.named_parameters():
        if name.startswith('osram.'):
            delta=before[name]-p
            high_delta=before[name]-high.state_dict()[name]
            torch.testing.assert_close(high_delta,4*delta,atol=2e-7,rtol=1e-3)
    restored=copy.deepcopy(low)
    restored_opt=make_task_optimizer(restored,cfg(),task,5e-5)
    restored_opt.load_state_dict(copy.deepcopy(low_opt.state_dict()))
    step(low,low_opt,x);step(restored,restored_opt,x)
    for name,p in low.state_dict().items():
        torch.testing.assert_close(p,restored.state_dict()[name],atol=0,rtol=0)


def test_disabled_partition_preserves_vanilla_adam_and_rng():
    torch.manual_seed(67)
    first=Tiny();second=copy.deepcopy(first);x=torch.randn(8,3)
    rng=torch.get_rng_state().clone()
    actual=make_task_optimizer(first,cfg(),task)
    assert torch.equal(rng,torch.get_rng_state())
    expected=task.make_optimizer(second,cfg(),'flat')
    step(first,actual,x);step(second,expected,x)
    for name,p in first.state_dict().items():
        torch.testing.assert_close(p,second.state_dict()[name],atol=0,rtol=0)
