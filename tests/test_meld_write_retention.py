import copy

import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from experiments.osram_meld_write_retention_20261014.run import WriteCapture


def tiny_trace():
    model = OSRAMBackbone(latent_dim=4, output_dim=4, num_heads=1,
                          key_dim=2, value_dim=2, bidirectional=False,
                          alpha_init=.8, write_step=.6, dropout=0).eval()
    memory = torch.zeros(1, 1, 2, 2)
    keys = torch.zeros(1, 1, 2, 3)
    keys[..., 0, 0] = 1
    availability = torch.tensor([[1., 0., 0.]])
    snapshots = []
    for t in range(6):
        values = torch.zeros(1, 1, 2, 3)
        values[..., :, 0] = torch.tensor([float(t + 1), float(6 - t)])
        pre = memory * torch.sigmoid(model.alpha_logits).view(1, 1, 1, 1)
        post = model.block_write(pre, keys, values, availability)
        snapshots.append(tuple(x.detach().clone() if torch.is_tensor(x) else x
                               for x in (t, pre, keys, values, availability, post)))
        memory = post
    return model, snapshots


def test_audit_preserves_state_rng_and_replays_original_joint_writes():
    from experiments.osram_meld_write_retention_20261014.audit import audit_batch
    model, trace = tiny_trace()
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    rows = list(audit_batch(model, trace, ['c'], .7, distances=(1, 2, 4)))
    assert rows
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in original.items():
        assert torch.equal(v, model.state_dict()[k])
    assert all(p.grad is None for p in model.parameters())
    assert all(r['modality'] == 'A' for r in rows)
    assert all(r['target_t'] > r['source_t'] for r in rows if r['phase'] == 'future')
    assert max(r['replay_error'] for r in rows) < 1e-6


def test_four_forks_share_start_hold_is_constant_and_decay_is_analytic():
    from experiments.osram_meld_write_retention_20261014.audit import replay_forks
    model, trace = tiny_trace()
    results = list(replay_forks(model, trace, 0, distances=(1, 2, 4)))
    initial = trace[0][-1]
    alpha = torch.sigmoid(model.alpha_logits).view(1, 1, 1, 1)
    for distance, branches, error in results:
        torch.testing.assert_close(branches['hold'], initial)
        torch.testing.assert_close(branches['decay_only'], initial * alpha ** distance)
        torch.testing.assert_close(branches['normal'], trace[distance][1])
        assert error < 1e-6
    assert not torch.equal(results[-1][1]['write_only'], initial)


def test_fixed_candidate_pool_has_no_future_or_cross_conversation_values():
    from experiments.osram_meld_write_retention_20261014.audit import candidate_pool
    _, trace = tiny_trace()
    assert candidate_pool(trace, 2, 0, 0) is None
    keys, values = candidate_pool(trace, 3, 0, 0)
    assert len(keys) == 4
    torch.testing.assert_close(values[0], trace[3][3][0, ..., 0])
    torch.testing.assert_close(values[-1], trace[0][3][0, ..., 0])


def test_capture_preserves_forward_and_restores_method_on_exception():
    model, _ = tiny_trace()
    node = torch.randn(3, 1, 4)
    latent = {m: torch.randn_like(node) for m in ('audio', 'text', 'visual')}
    availability = torch.tensor([[[1., 0., 1.]], [[1., 1., 0.]], [[0., 1., 1.]]])
    umask = torch.ones(1, 3)
    args = (node, latent, availability, torch.zeros(1, 3, dtype=torch.long), umask, [3])
    with torch.no_grad():
        original, _ = model(*args)
        with WriteCapture(model, umask) as capture:
            replay, _ = model(*args)
    assert torch.equal(original, replay)
    assert len(capture.snapshots) == 3
    assert 'block_write' not in model.__dict__
    try:
        with WriteCapture(model, umask):
            raise RuntimeError('deliberate')
    except RuntimeError:
        pass
    assert 'block_write' not in model.__dict__


def test_vectorized_metrics_match_scalar_for_each_source_head():
    from experiments.osram_meld_write_retention_20261014.audit import (
        _metric_arrays, _metrics, candidate_pool,
    )
    model, trace = tiny_trace()
    pool = candidate_pool(trace, 3, 0, 0)
    memory = trace[3][-1]
    keys = trace[3][2][..., 0]
    values = trace[3][3][..., 0]
    actual = _metric_arrays(memory, keys, values, [pool])
    expected = _metrics(memory[0, 0], keys[0, 0], values[0, 0], pool, 0)
    import math
    for name, value in expected.items():
        assert (math.isnan(actual[name][0, 0]) if value is None
                else abs(actual[name][0, 0] - value) < 1e-5)
