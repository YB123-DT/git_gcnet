"""Write-only interventions must preserve the inference mathematics."""
import copy

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone, MODALITIES


def case(dropout=0.):
    torch.manual_seed(91)
    model = OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
                          key_dim=3, value_dim=4, n_speakers=2,
                          dropout=dropout, bidirectional=False)
    node = torch.randn(4, 2, 8)
    latents = {m: torch.randn_like(node) for m in MODALITIES}
    avail = torch.ones(4, 2, 3)
    avail[1:, :, 1] = 0
    avail[-1, 1] = 0
    umask = torch.tensor([[1., 1., 1., 1.], [1., 1., 1., 0.]])
    return model, (node, latents, avail, torch.zeros(2, 4, dtype=torch.long), umask, [4, 3])


def test_disabled_and_identity_override_preserve_output_rng_grad_and_state():
    model, args = case(.2)
    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.nn.init.normal_(model.emotion_adapter[-1].weight, std=.1)
    clone.load_state_dict(model.state_dict(), strict=True)
    state = torch.get_rng_state()
    first = model(*args)[0]
    first.sum().backward()
    after = torch.get_rng_state()
    torch.set_rng_state(state)
    nodes = {}
    second = clone(*args, write_mask_override=args[2], write_node_collector=nodes)[0]
    second.sum().backward()
    assert torch.equal(after, torch.get_rng_state())
    assert torch.equal(first, second)
    for p, q in zip(model.parameters(), clone.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert list(model.state_dict()) == list(clone.state_dict())


def test_omit_preserves_local_queries_and_outputs_through_source():
    model, args = case()
    model.eval()
    torch.nn.init.normal_(model.emotion_adapter[-1].weight, std=.1)
    queries, locals_ = [], []
    original = model._project_sequence
    def project(*a, **kw):
        result = original(*a, **kw)
        queries.append(result[2].detach().clone())
        return result
    model._project_sequence = project
    hook = model.local_path.register_forward_hook(lambda _m, _i, out: locals_.append(out.detach().clone()))
    keep = model(*args)[0]
    override = args[2].clone()
    override[0, 0, 1] = 0
    with torch.no_grad():
        omit = model(*args, write_mask_override=override)[0]
    hook.remove()
    assert torch.equal(keep[0], omit[0])
    assert torch.equal(keep[:, 1], omit[:, 1])
    assert torch.equal(queries[0], queries[1])
    assert torch.equal(locals_[0], locals_[1])
    assert not torch.equal(keep[1:, 0], omit[1:, 0])
    assert not omit.requires_grad


@pytest.mark.parametrize('kind', ['shape', 'nan', 'fraction', 'absent', 'empty', 'padding'])
def test_override_validation(kind):
    model, args = case()
    override = args[2].clone()
    if kind == 'shape': override = override[..., :2]
    if kind == 'nan': override[0, 0, 0] = float('nan')
    if kind == 'fraction': override[0, 0, 0] = .5
    if kind == 'absent': override[1, 0, 1] = 1
    if kind == 'empty': override[0, 0] = 0
    if kind == 'padding': override[-1, 1, 0] = 1
    with pytest.raises(ValueError):
        model(*args, write_mask_override=override)


def test_actual_source_nodes_connected_but_initial_zero_adapter_blocks_credit():
    model, args = case()
    nodes = {}
    hidden = model(*args, write_node_collector=nodes)[0]
    grad = torch.autograd.grad(hidden[2, 0, 0], nodes[0], retain_graph=True)
    assert all(torch.count_nonzero(g) == 0 for g in grad)
    torch.nn.init.normal_(model.emotion_adapter[-1].weight, std=.1)
    nodes = {}
    hidden = model(*args, write_node_collector=nodes)[0]
    grad = torch.autograd.grad(hidden[2, 0, 0], nodes[0], retain_graph=True)
    assert all(torch.isfinite(g).all() for g in grad)
    assert all(g[0, ..., 1].abs().sum() > 0 for g in grad)
    assert all(torch.count_nonzero(g[1]) == 0 for g in grad)
    params = tuple(model.key_projectors.parameters()) + tuple(model.value_projectors.parameters())
    masked = tuple(g.detach() * torch.tensor([0., 1., 0.]) for g in grad)
    writer_grad = torch.autograd.grad(nodes[0], params, grad_outputs=masked, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in writer_grad)


def test_unsupported_scan_and_reused_collector_rejected():
    model, args = case()
    model.bidirectional = True
    with pytest.raises(ValueError): model(*args, write_node_collector={})
    model.bidirectional = False
    with pytest.raises(ValueError): model(*args, write_node_collector={0: None})


def test_joint_write_remains_column_permutation_invariant():
    model, _ = case()
    memory = torch.randn(2, 2, 4, 3)
    keys, values = torch.randn(2, 2, 3, 3), torch.randn(2, 2, 4, 3)
    mask = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
    order = [2, 0, 1]
    first = model.block_write(memory, keys, values, mask)
    second = model.block_write(memory, keys[..., order], values[..., order], mask[:, order])
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)


def test_ordinary_task_warmup_opens_real_writer_gradient():
    model, args = case()
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    target = torch.randn(4, 2, 10)
    optimizer.zero_grad()
    hidden = model(*args)[0]
    (hidden - target).square().mean().backward()
    assert model.local_skip.weight.grad.abs().sum() > 0
    assert model.emotion_adapter[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()
    nodes = {}
    hidden = model(*args, write_node_collector=nodes)[0]
    gradients = torch.autograd.grad((hidden[2, 0] - target[2, 0]).square().mean(), nodes[0])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_model_threads_hooks_and_strict_legacy_state():
    pytest.importorskip('torch_geometric')
    from gcnet_missing_m3.model import MissingM3GraphModel
    model = MissingM3GraphModel(
        base_model='LSTM', adim=2, tdim=3, vdim=4, D_e=4,
        graph_hidden_size=2, n_speakers=2, window_past=1, window_future=1,
        n_classes=1, dropout=0., no_cuda=True, latent_dim=8, num_experts=2,
        top_k=1, projector_dropout=0., predictor_dropout=0., backbone_type='osram',
        osram_output_dim=10, osram_num_heads=2, osram_key_dim=3,
        osram_value_dim=4, osram_bidirectional=False).eval()
    _, args = case()
    call = ([torch.randn(4, 2, 9)], *args[2:])
    legacy = copy.deepcopy(model.state_dict())
    base = model(*call)[0]
    nodes = {}
    hooked = model(*call, write_node_collector=nodes, write_mask_override=args[2])[0]
    assert torch.equal(base, hooked)
    assert len(nodes) == 4
    model.load_state_dict(legacy, strict=True)
    assert list(model.state_dict()) == list(legacy)
    with pytest.raises(ValueError):
        model(*call, predict_missing=True, write_node_collector={})
