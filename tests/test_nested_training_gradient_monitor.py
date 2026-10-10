import copy
from types import SimpleNamespace

import torch
from torch import nn

from experiments.osram_nested_training_gradients_20261010.monitor import GradientMonitor


class Evidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.1))

    def forward(self, local, base, gap, availability, umask):
        return local * (1 + self.scale), base * (1 + self.scale), gap * (1 + self.scale)


class Backbone(nn.Module):
    def __init__(self, nested):
        super().__init__()
        self.local_skip = nn.Linear(2, 1)
        self.emotion_adapter = nn.Linear(2 + 4 * 1024, 1)
        if nested:
            self.meaningful_block = Evidence()

    def forward(self, node, latents, availability, qmask, umask):
        local, base, gap = node, latents['base'], latents['gap']
        if hasattr(self, 'meaningful_block'):
            readout, b, g = self.meaningful_block(local, base, gap, availability, umask)
        else:
            readout, b, g = local, base, gap
        return self.local_skip(local) + self.emotion_adapter(torch.cat((readout, b, g.flatten(2)), -1))


class Model(nn.Module):
    def __init__(self, nested):
        super().__init__()
        self.osram = Backbone(nested)


def run(model, monitor=None):
    local = torch.ones(3, 1, 2, requires_grad=True)
    base = torch.ones(3, 1, 1024, requires_grad=True)
    gap = torch.ones(3, 1, 3, 1024, requires_grad=True)
    av = torch.tensor([[[1., 1., 0.]], [[1., 0., 1.]], [[0., 0., 0.]]])
    mask = torch.tensor([[1., 1., 0.]])
    if monitor:
        monitor.start_epoch(model, 0, SimpleNamespace(train_missing_rates=tuple(i/10 for i in range(8))))
    pred = model.osram(local, {'base': base, 'gap': gap}, av, None, mask)
    pred.square().sum().backward()
    if monitor:
        monitor.before_clip(model, 1.)
        rows = monitor.finish_epoch(expected_steps=1)
    else:
        rows = None
    return pred.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}, rows


def test_monitor_preserves_outputs_gradients_rng_and_limits_masks():
    for nested in (False, True):
        torch.manual_seed(5)
        model = Model(nested)
        plain = copy.deepcopy(model)
        before = torch.get_rng_state().clone()
        monitor = GradientMonitor('nested' if nested else 'flat', latent_dim=2)
        pred, gradients, rows = run(model, monitor)
        assert torch.equal(before, torch.get_rng_state())
        expected, expected_gradients, _ = run(plain)
        assert torch.equal(pred, expected)
        assert all(torch.equal(gradients[n], expected_gradients[n]) for n in gradients)
        inputs = rows[0]['inputs']
        assert inputs['Local']['count'] == 2
        assert inputs['Base']['count'] == 1
        assert inputs['Gap-T']['count'] == 1
        assert inputs['Gap-A']['count'] == inputs['Gap-V']['count'] == 0
        assert rows[0]['preclip_global_norm'] > 0
        if nested:
            assert abs(rows[0]['residual']['Base']['ratio_mean'] - .1) < 1e-5
            assert rows[0]['parameter_groups']['nested']['norm'] > 0


def test_evaluation_does_not_add_monitor_rows():
    model = Model(False)
    monitor = GradientMonitor('flat', latent_dim=2)
    _, _, rows = run(model, monitor)
    assert len(rows) == 1
    run(model)
    assert monitor.active is False
    assert len(monitor.rows) == 1
