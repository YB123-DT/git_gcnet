"""Frozen interventions must preserve the original causal model outside their scope."""
import copy

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from experiments.osram_meld_write_retention_20261014.run import WriteCapture
from experiments.osram_meld_write_retention_20261014.task_replay import TaskCapture


def tiny_trace():
    model = OSRAMBackbone(latent_dim=4, output_dim=4, num_heads=2,
                          key_dim=2, value_dim=2, bidirectional=False,
                          alpha_init=.8, write_step=.6, dropout=0).eval()
    torch.nn.init.normal_(model.emotion_adapter[-1].weight, std=.2)
    node = torch.randn(5, 2, 4)
    latent = {name: torch.randn_like(node) for name in ('audio', 'text', 'visual')}
    availability = torch.tensor([[[1., 0., 1.], [1., 1., 0.]]] * 5)
    availability[3:, 1] = 0
    umask = torch.tensor([[1., 1., 1., 1., 1.], [1., 1., 1., 0., 0.]])
    return model, (node, latent, availability, torch.zeros(2, 5, dtype=torch.long), umask, [5, 3])


def test_override_changes_only_requested_logits_and_restores_exactly_on_exception():
    from experiments.osram_meld_memory_sensitivity_20261017.sensitivity import parameter_override
    model, _ = tiny_trace()
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with parameter_override(model, beta=.05):
            torch.testing.assert_close(model.beta_logits.sigmoid(), torch.full_like(model.beta_logits, .05))
            assert torch.equal(model.alpha_logits, original['alpha_logits'])
            raise RuntimeError('deliberate')
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
    with parameter_override(model):
        for key, value in model.state_dict().items():
            assert torch.equal(value, original[key])
    assert torch.equal(rng, torch.get_rng_state())
    for invalid in (0, 1, float('nan'), -1):
        with pytest.raises(ValueError):
            with parameter_override(model, alpha=invalid):
                pass


@pytest.mark.parametrize('field,value', [('beta', .05), ('alpha', .95)])
def test_parameter_interventions_leave_current_local_query_and_kv_identical(field, value):
    from experiments.osram_meld_memory_sensitivity_20261017.sensitivity import parameter_override
    model, args = tiny_trace()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    with torch.no_grad(), TaskCapture(model) as normal, WriteCapture(model, args[-2]) as write:
        original, _ = model(*args)
    with torch.no_grad(), parameter_override(model, **{field: value}), TaskCapture(model) as changed, WriteCapture(model, args[-2]) as changed_write:
        result, _ = model(*args)
    assert torch.equal(normal.local, changed.local)
    assert torch.equal(normal.queries, changed.queries)
    assert torch.equal(original[0], result[0])  # read-before-write, empty history
    assert torch.count_nonzero(normal.base[0]) == 0
    assert torch.count_nonzero(changed.gap[0]) == 0
    assert torch.count_nonzero(result[3:, 1]) == 0
    assert not torch.equal(normal.base[1:], changed.base[1:])
    for a, b in zip(write.snapshots, changed_write.snapshots):
        for index in (2, 3, 4):
            assert torch.equal(a[index], b[index])
    assert torch.equal(rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, tensor in before.items():
        assert torch.equal(model.state_dict()[key], tensor)


def test_spectra_use_only_active_columns_and_valid_rows():
    from experiments.osram_meld_memory_sensitivity_20261017.sensitivity import spectral_rows
    keys = torch.zeros(3, 1, 3, 3, dtype=torch.float64)
    keys[0, 0] = torch.eye(3)
    keys[1, 0, 0, :] = 1  # two observed, exactly collinear columns
    keys[2] = float('nan')  # padding must never reach an eigensolve
    availability = torch.tensor([[1., 0., 1.], [1., 1., 0.], [0., 0., 0.]])
    rows = spectral_rows(keys, availability, torch.tensor([True, True, False]),
                         torch.full((1, 3), .5), .001)
    assert len(rows) == 2
    assert rows[0]['active_modalities'] == [0, 2]
    assert rows[0]['gram_eigenvalues'] == pytest.approx([1, 1])
    assert rows[0]['weighted_gram_eigenvalues'] == pytest.approx([.5, .5])
    assert rows[0]['effective_ridge_diagonal'] == pytest.approx([.002, .002])
    assert rows[1]['gram_eigenvalues'] == pytest.approx([0, 2])
    assert rows[1]['weighted_eigenvalues_below_ridge'] == 1


def test_embedded_beta_is_effective_ridge_not_external_write_multiplier():
    from experiments.osram_meld_memory_sensitivity_20261017.sensitivity import spectral_rows
    model, _ = tiny_trace()
    memory = torch.randn(1, 2, 2, 2, dtype=torch.float64)
    keys = torch.randn(1, 2, 2, 3, dtype=torch.float64)
    values = torch.randn_like(keys)
    availability = torch.tensor([[1., 0., 1.]], dtype=torch.float64)
    beta = torch.tensor([[.1, .5, .8], [.2, .7, .3]], dtype=torch.float64)
    actual = model.block_write(memory, keys, values, availability, beta=beta)
    active = [0, 2]
    k, v = keys[..., active], values[..., active]
    ridge = torch.diag_embed(model.write_ridge / beta[:, active])
    expected = memory + model.write_step * (v-memory @ k) @ torch.linalg.solve(k.transpose(-1, -2) @ k + ridge, k.transpose(-1, -2))
    torch.testing.assert_close(actual, expected)
    permutation = [2, 0, 1]
    torch.testing.assert_close(actual, model.block_write(memory, keys[..., permutation], values[..., permutation], availability[..., permutation], beta=beta[..., permutation]))
    assert len(spectral_rows(keys, availability, torch.tensor([True]), beta, model.write_ridge)) == 2


def test_head_mask_only_removes_correct_forward_slice_without_mutating_inputs():
    from experiments.osram_meld_memory_sensitivity_20261017.sensitivity import mask_head
    local_dim, context_dim, value_dim = 4, 8, 2  # two heads, backward half4
    original = torch.arange(2 * (local_dim+4*context_dim), dtype=torch.float32).reshape(2, -1)
    for kind in ('base', 'gap'):
        changed = mask_head(original, local_dim=local_dim, context_dim=context_dim,
                            head=1, value_dim=value_dim, kind=kind)
        expected = original.clone()
        for slot in ([0] if kind == 'base' else [1, 2, 3]):
            expected[..., local_dim+slot*context_dim+2:local_dim+slot*context_dim+4] = 0
        assert torch.equal(changed, expected)
    assert original.count_nonzero() == original.numel()-1
    with pytest.raises(ValueError):
        mask_head(original, local_dim=4, context_dim=8, head=2, value_dim=2, kind='base')


def test_lagged_alpha_forks_start_from_same_postwrite_state_and_replay_normal():
    from experiments.osram_meld_memory_sensitivity_20261017.sensitivity import alpha_forks
    model, args = tiny_trace()
    with torch.no_grad(), WriteCapture(model, args[-2]) as capture:
        model(*args)
    by_time = {row[0]: row for row in capture.snapshots}
    original = list(alpha_forks(model, capture.snapshots, model.alpha_logits.sigmoid()))
    assert {target-source for source, target, _ in original} == {1, 2, 4}
    for source, target, memory in original:
        valid = by_time[target][4].bool().any(-1)
        torch.testing.assert_close(memory[valid], by_time[target][1][valid])
    changed = list(alpha_forks(model, capture.snapshots, .95))
    for source, target, memory in changed:
        if target-source == 1:
            valid = by_time[target][4].bool().any(-1)
            torch.testing.assert_close(memory[valid], by_time[source][-1][valid]*.95)


def test_global_alpha_read_differences_at_t_equal_first_source_anchored_fork():
    """Saved full-trajectory read stats at t=d also audit source0, lag=d.

    The zero initial state makes the first post-write state independent of alpha.
    This identity permits a first-source lag read analysis without new inference.
    """
    from experiments.osram_meld_memory_sensitivity_20261017.sensitivity import alpha_forks, parameter_override
    model, args = tiny_trace()
    with torch.no_grad(), WriteCapture(model, args[-2]) as original:
        model(*args)
    with torch.no_grad(), parameter_override(model, alpha=.95), WriteCapture(model, args[-2]) as changed:
        model(*args)
    torch.testing.assert_close(original.snapshots[0][-1], changed.snapshots[0][-1], rtol=0, atol=0)
    by_time = {row[0]: row for row in changed.snapshots}
    for source, target, memory in alpha_forks(model, original.snapshots, .95):
        if source == 0:
            valid = args[-2][:, target].bool()
            torch.testing.assert_close(memory[valid], by_time[target][1][valid])
