"""Contracts for the frozen MELD Query-versus-Decision diagnostic modules."""

import importlib

import pytest
import torch
from torch.nn import functional as F

from gcnet_missing_m3.osram import OSRAMBackbone
from experiments.osram_meld_write_retention_20261014.run import WriteCapture
from experiments.osram_meld_write_retention_20261014.task_replay import TaskCapture


def _inputs():
    torch.manual_seed(7)
    node = torch.randn(4, 2, 8)
    latents = {name: torch.randn(4, 2, 8) for name in ('audio', 'text', 'visual')}
    availability = torch.tensor([
        [[1., 0., 1.], [0., 1., 1.]], [[1., 1., 0.], [0., 0., 1.]],
        [[1., 1., 1.], [1., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]],
    ])
    qmask = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]])
    umask = torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 0.]])
    return node, latents, availability, qmask, umask, [3, 3]


def _learning():
    name = 'experiments.osram_meld_query_decision_20261014.learning'
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name in (name, name.rsplit('.', 1)[0]):
            pytest.fail('Missing Query-versus-Decision learning module')
        raise


def _reads():
    torch.manual_seed(21)
    memory = torch.randn(4, 2, 4, 3)
    queries = F.normalize(torch.randn(4, 4, 2, 3), dim=-1)
    keys = F.normalize(torch.randn(4, 2, 3, 3), dim=-2)
    availability = torch.tensor([[1., 0., 1.], [0., 1., 1.],
                                 [1., 1., 0.], [1., 1., 1.]])
    return memory, queries, keys, availability


@pytest.mark.parametrize('strength', [0., 0.4, 1.])
def test_reads_match_actual_original_forward_capture(strength):
    learning = _learning()
    model = OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
                          key_dim=3, value_dim=4, n_speakers=2,
                          dropout=0., bidirectional=False,
                          gap_residual_strength=strength).eval()
    args = _inputs()
    with torch.no_grad(), TaskCapture(model) as task, WriteCapture(model, args[4]) as write:
        model(*args)
    for time, memory, keys, _, availability, _ in write.snapshots:
        base, gap = learning.forward_reads(memory, task.queries[time], keys,
                                           availability, model.read_ridge, strength)
        valid = task.valid[time]
        torch.testing.assert_close(base[valid], task.base[time, valid], rtol=0, atol=0)
        torch.testing.assert_close(gap[valid], task.gap[time, valid], rtol=0, atol=0)


def test_read_gradients_reach_only_queries():
    learning = _learning()
    memory, queries, keys, availability = _reads()
    memory.requires_grad_()
    keys.requires_grad_()
    queries.requires_grad_()
    base, gap = learning.forward_reads(memory, queries, keys, availability)
    (base.square().sum() + gap.square().sum()).backward()
    assert memory.grad is None
    assert keys.grad is None
    assert torch.isfinite(queries.grad).all()
    assert queries.grad[:, 0].abs().sum() > 0
    active = ~availability.bool()
    assert queries.grad[:, 1:][active].abs().sum() > 0
    assert torch.count_nonzero(queries.grad[:, 1:][~active]) == 0


def test_empty_memory_has_zero_query_influence():
    learning = _learning()
    memory, queries, keys, availability = _reads()
    memory.zero_()
    queries.requires_grad_()
    base, gap = learning.forward_reads(memory, queries, keys, availability)
    assert torch.count_nonzero(base) == torch.count_nonzero(gap) == 0
    (base.sum() + gap.sum()).backward()
    assert torch.count_nonzero(queries.grad) == 0


def test_inactive_keys_and_gaps_safely_mask_nan():
    learning = _learning()
    memory, queries, keys, availability = _reads()
    clean_base, clean_gap = learning.forward_reads(memory, queries, keys, availability)
    keys = keys.clone()
    keys.masked_fill_(~availability.bool()[:, None, None, :], float('nan'))
    queries = queries.clone()
    queries[:, 1:].masked_fill_(availability.bool()[:, :, None, None], float('nan'))
    queries.requires_grad_()
    base, gap = learning.forward_reads(memory, queries, keys, availability)
    torch.testing.assert_close(base, clean_base)
    torch.testing.assert_close(gap, clean_gap)
    assert torch.isfinite(gap).all()
    assert torch.count_nonzero(gap[availability.bool()]) == 0
    (base.sum() + gap.sum()).backward()
    assert torch.isfinite(queries.grad).all()


def _corrector(learning):
    return learning.QueryCorrector(local_dim=5, key_dim=3, num_heads=2,
                                   hidden_dim=7, type_dim=2, head_dim=2)


def test_corrector_zero_start_and_inactive_queries_unchanged():
    learning = _learning()
    _, queries, _, availability = _reads()
    local = torch.randn(4, 5)
    corrector = _corrector(learning)
    corrected = corrector(local, queries, availability)
    torch.testing.assert_close(corrected, queries, rtol=1e-6, atol=1e-7)
    final = [m for m in corrector.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert torch.count_nonzero(final.weight) == torch.count_nonzero(final.bias) == 0
    with torch.no_grad():
        final.bias.copy_(torch.tensor([0.2, -0.3, 0.4]))
    corrected = corrector(local, queries, availability)
    torch.testing.assert_close(corrected[:, 1:][availability.bool()],
                               queries[:, 1:][availability.bool()], rtol=0, atol=0)
    active = torch.cat((torch.ones(4, 1, dtype=torch.bool), ~availability.bool()), dim=1)
    torch.testing.assert_close(corrected[active].norm(dim=-1),
                               torch.ones_like(corrected[active][..., 0]))
    assert not torch.equal(corrected[:, 0], queries[:, 0])


def test_corrector_detaches_original_inputs_but_parameters_receive_gradients():
    learning = _learning()
    _, queries, _, availability = _reads()
    local = torch.randn(4, 5, requires_grad=True)
    queries.requires_grad_()
    availability.requires_grad_()
    corrector = _corrector(learning)
    corrected = corrector(local, queries, availability)
    (corrected * torch.randn_like(corrected)).sum().backward()
    assert local.grad is queries.grad is availability.grad is None
    parameters = [p.grad for p in corrector.parameters() if p.grad is not None]
    assert parameters and all(torch.isfinite(g).all() for g in parameters)
    assert sum(g.abs().sum() for g in parameters) > 0


def _head_inputs():
    torch.manual_seed(31)
    local = torch.randn(6, 5)
    base = torch.randn(6, 8)
    gap = torch.randn(6, 3, 8)
    availability = torch.tensor([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.],
                                 [1., 1., 1.], [0., 1., 0.], [0., 0., 1.]])
    return local, base, gap, availability


def test_feature_pack_masks_inactive_nan_gaps_and_keeps_availability():
    learning = _learning()
    local, base, gap, availability = _head_inputs()
    gap[availability.bool()] = float('nan')
    packed = learning.pack_features(local, base, gap, availability)
    assert packed.shape == (6, 5 + 8 + 3 * 8 + 3)
    assert torch.isfinite(packed).all()
    assert torch.count_nonzero(packed[:, 13:37].reshape(6, 3, 8)[availability.bool()]) == 0
    assert torch.equal(packed[:, -3:], availability)


def test_normalizer_uses_explicit_training_fit_and_masks_after_normalizing():
    learning = _learning()
    args = _head_inputs()
    head = learning.LightReadout(local_dim=5, context_dim=8, hidden_dim=7)
    head.fit_normalizer(*args)
    packed = learning.pack_features(*args)
    torch.testing.assert_close(head.feature_mean, packed.mean(0))
    torch.testing.assert_close(head.feature_scale, packed.std(0, unbiased=False).clamp_min(1e-6))
    saved = {k: v.clone() for k, v in head.named_buffers()}
    normalized = head.normalized_features(*args)
    assert torch.count_nonzero(normalized[:, 13:37].reshape(6, 3, 8)[args[-1].bool()]) == 0
    for mode in (True, False):
        head.train(mode)
        changed = tuple(v + 1000 if i < 3 else v for i, v in enumerate(args))
        assert head(*changed).shape == (6, 7)
        for key, value in head.named_buffers():
            assert torch.equal(value, saved[key])


def test_frozen_readout_preserves_input_gradients_and_no_dropout():
    learning = _learning()
    local, base, gap, availability = _head_inputs()
    head = learning.LightReadout(local_dim=5, context_dim=8, hidden_dim=7)
    head.fit_normalizer(local, base, gap, availability)
    head.requires_grad_(False)
    base.requires_grad_()
    gap.requires_grad_()
    head(local, base, gap, availability).square().sum().backward()
    assert base.grad.abs().sum() > 0
    assert gap.grad[~availability.bool()].abs().sum() > 0
    assert torch.count_nonzero(gap.grad[availability.bool()]) == 0
    assert all(p.grad is None for p in head.parameters())
    assert not any(isinstance(m, torch.nn.Dropout) for m in head.modules())


def test_default_head_dimension_and_fixed_normalizer():
    learning = _learning()
    head = learning.LightReadout()
    assert head.feature_mean.shape == head.feature_scale.shape == (2307,)
    assert torch.count_nonzero(head.feature_mean) == 0
    assert torch.equal(head.feature_scale, torch.ones(2307))
    assert not head.feature_mean.requires_grad and not head.feature_scale.requires_grad
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(2307, 64), (64, 7)]
