"""One-stage probability routing keeps the original scan and training graph."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.osram_text_path_router_20261019.router import TextPathRouter


class TinyBackbone(nn.Module):
    """Small causal-source fixture with the production Flat/head interfaces."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.writer = nn.Linear(4, 24)
        self.input_dropout = nn.Dropout(.2)
        self.osram = nn.Module()
        self.osram.num_heads, self.osram.value_dim = 1, 2
        self.osram.local_skip = nn.Linear(4, 8)
        self.osram.emotion_adapter = nn.Sequential(nn.Linear(20, 8), nn.GELU(), nn.Dropout(.4), nn.Linear(8, 8))
        self.osram.emotion_norm = nn.LayerNorm(8)
        self.smax_fc = nn.Linear(8, 3)
        self.calls = 0
        self.dropout_outputs = []
        self.osram.emotion_adapter[2].register_forward_hook(
            lambda module, inputs, output: self.dropout_outputs.append((inputs[0].detach(), output.detach())))

    def forward(self, incomplete, availability, qmask, umask, lengths, predict_missing=False):
        self.calls += 1
        local = self.input_dropout(self.encoder(incomplete[0]))
        # Strictly-past source state, independently initialized on each scan.
        values = self.writer(local).reshape(*local.shape[:2], 4, 3, 2)
        sources = torch.cat((torch.zeros_like(values[:1]), values[:-1]), 0)
        sources = sources * umask.T[..., None, None, None]
        sources[:, :, 1:] = torch.where(
            ~availability.bool()[..., None, None], sources[:, :, 1:], 0.)
        trajectory = local.new_zeros(*local.shape[:2], 3, 4)
        trajectory[1:, :, :, 3] = 1
        self.osram.last_source_trace = dict(source_base=sources[:, :, 0], source_gap=sources[:, :, 1:], trajectory=trajectory)
        context = torch.cat((sources.sum(-2), torch.zeros_like(sources.sum(-2))), -1).flatten(2)
        emotion_input = torch.cat((local, context), -1)
        hidden = self.osram.emotion_norm(self.osram.local_skip(local) + self.osram.emotion_adapter(emotion_input))
        hidden = hidden * umask.T.bool()[..., None].to(hidden.dtype)
        return self.smax_fc(hidden), None, None, None


def fixture(all_text=False):
    torch.manual_seed(614)
    backbone = TinyBackbone()
    router = TextPathRouter(local_dim=4, context_dim=2, projection_dim=3, hidden_dim=5,
                            detach_inputs=False)
    view = dict(incomplete=torch.randn(4, 2, 4), availability=torch.ones(4, 2, 3),
                qmask=torch.zeros(4, 2, 2), umask=torch.tensor([[1., 1., 1., 1.], [1., 1., 1., 0.]]),
                lengths=[4, 3])
    if not all_text:
        view['availability'][1:, :, 1] = 0
    return backbone, router, view


def test_connected_router_flag_preserves_legacy_outputs_but_allows_input_gradients():
    frozen = TextPathRouter(local_dim=4, context_dim=2, projection_dim=3, hidden_dim=5)
    joint = TextPathRouter(local_dim=4, context_dim=2, projection_dim=3, hidden_dim=5, detach_inputs=False)
    joint.load_state_dict(frozen.state_dict(), strict=True)
    with torch.no_grad():
        frozen.network[-1].weight.normal_(std=.03)
        joint.load_state_dict(frozen.state_dict())
    values = [torch.randn(3, 4), torch.randn(3, 2), torch.randn(3, 2),
              torch.tensor([[2., .5, 1., 1.]]).repeat(3, 1), torch.randn(3, 3, 3)]
    values = [x.requires_grad_() for x in values]
    eligible = torch.ones(3, dtype=torch.bool)
    old, _ = frozen(*values, eligible)
    new, _ = joint(*values, eligible)
    assert torch.equal(old, new)
    F.nll_loss(new, torch.tensor([0, 1, 2])).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in values)
    assert values[-1].grad.abs().sum() > 0
    assert values[0].grad.abs().sum() > 0


def test_one_scan_same_candidate_dropout_and_original_post_forward_rng():
    from experiments.osram_text_path_router_joint_20261019.joint import JointTextPathModel
    backbone, router, view = fixture()
    reference = copy.deepcopy(backbone)
    start = torch.random.get_rng_state().clone()
    expected = reference([view['incomplete']], view['availability'], view['qmask'], view['umask'], view['lengths'])[0]
    expected_rng = torch.random.get_rng_state().clone()
    backbone.dropout_outputs.clear()  # deepcopy keeps the observer closure's owner
    torch.random.set_rng_state(start)
    model = JointTextPathModel(backbone, router).train()
    out = model(view)
    assert backbone.calls == 1
    assert torch.equal(out['candidates'][..., 0, :], expected)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert len(backbone.dropout_outputs) == 3
    masks = [after.ne(0) for before, after in backbone.dropout_outputs]
    assert torch.equal(masks[0], masks[1]) and torch.equal(masks[0], masks[2])
    inactive = ~out['eligible']
    assert not out['eligible'][0].any() and not out['eligible'][3, 1]
    for index in (1, 2):
        assert torch.equal(out['candidates'][..., index, :][inactive], expected[inactive])
    assert torch.equal(out['log_probs'][inactive], expected.log_softmax(-1)[inactive])
    assert out['weights'].shape == (4, 2, 3)


def test_joint_backward_updates_writer_encoder_flat_head_and_router():
    from experiments.osram_text_path_router_joint_20261019.joint import JointTextPathModel
    backbone, router, view = fixture()
    model = JointTextPathModel(backbone, router).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    result = model(view)
    valid = view['umask'].T.bool()
    loss = F.nll_loss(result['log_probs'][valid], torch.arange(valid.sum()) % 3)
    loss.backward()
    for module in (backbone.encoder, backbone.writer, backbone.osram.local_skip,
                   backbone.osram.emotion_adapter, backbone.smax_fc, router.network[-1]):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    for name in ('backbone.encoder.weight', 'backbone.writer.weight', 'router.network.3.weight'):
        assert not torch.equal(before[name], dict(model.named_parameters())[name])


def test_all_text_observed_still_trains_backbone_without_router_updates():
    from experiments.osram_text_path_router_joint_20261019.joint import JointTextPathModel
    backbone, router, view = fixture(all_text=True)
    model = JointTextPathModel(backbone, router).train()
    result = model(view)
    assert not result['eligible'].any()
    valid = view['umask'].T.bool()
    F.nll_loss(result['log_probs'][valid], torch.zeros(int(valid.sum()), dtype=torch.long)).backward()
    assert backbone.encoder.weight.grad.abs().sum() > 0
    assert backbone.smax_fc.weight.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in router.parameters())


def test_joint_requires_graph_connected_router():
    from experiments.osram_text_path_router_joint_20261019.joint import JointTextPathModel
    with pytest.raises(ValueError, match='detach'):
        JointTextPathModel(TinyBackbone(), TextPathRouter(local_dim=4, context_dim=2))
