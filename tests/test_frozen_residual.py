import importlib.util
from pathlib import Path

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1] / 'experiments/osram_frozen_residual_20261009/run.py'
spec = importlib.util.spec_from_file_location('frozen_residual', PATH)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_zero_initialization_and_gradient():
    model = run.ResidualHead(516)
    x, y0 = torch.randn(9, 516), torch.randn(9)
    assert torch.equal(model(x, y0), y0)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(3):
        optimizer.zero_grad()
        model(x, y0).square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()


def test_equal_capacity_and_donor_current_local():
    local = np.arange(3 * 256, dtype=np.float32).reshape(3, 256)
    availability = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float32)
    real, donor = np.ones((3, 256)), np.full((3, 256), 2)
    a = run.residual_features(np.ones(3), local, availability, real)
    b = run.residual_features(np.ones(3), local, availability, donor)
    assert a.shape == b.shape == (3, 516)
    assert np.array_equal(a[:, :260], b[:, :260])
    assert sum(p.numel() for p in run.ResidualHead(a.shape[1]).parameters()) == 16577
    assert run.residual_features(np.ones(3), local, availability, np.ones((3, 1)), np.ones(3)).shape == (3, 262)


def test_donor_mask_and_first_turn():
    data = dict(utterance_indices=np.array([0, 1]), availability=np.array([[1, 0, 1], [1, 0, 1]]))
    memory = np.ones((4, 4, 512), np.float32)
    result = run.donor_memory(memory, np.array([-1, 2]), data)
    assert not result[0].any()
    assert not result[1, [1, 3]].any()
    assert result[1, [0, 2]].all()


def test_training_does_not_use_test_labels(tmp_path):
    rng = np.random.default_rng(4)
    features = {s: rng.normal(size=(12, 8)).astype('float32') for s in ('train', 'test')}
    offsets = {s: np.zeros(12, dtype='float32') for s in features}
    labels = {s: rng.normal(size=12).astype('float32') for s in features}
    first, _ = run.fit_head(features, offsets, labels, 66, 2, 'cpu', tmp_path / 'one')
    labels['test'] *= -100
    second, _ = run.fit_head(features, offsets, labels, 66, 2, 'cpu', tmp_path / 'two')
    assert all(torch.equal(first[k], second[k]) for k in first)


def test_frozen_probe_keeps_current_local_with_donor_memory(tmp_path):
    probe = torch.nn.Sequential(torch.nn.Linear(515, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    path = tmp_path / 'probe.pt'
    torch.save(dict(input_dim=515, hidden_dim=64, state_dict=probe.state_dict()), path)
    frozen = run.load_probe(path)
    local = np.ones((2, 256), dtype=np.float32)
    availability = np.ones((2, 3), dtype=np.float32)
    memory = np.ones((2, 4, 512), dtype=np.float32)
    projection = np.full((512, 64), .001, dtype=np.float32)
    result = run.decode(frozen, local, availability, memory, projection)
    manual = np.concatenate((local, (memory @ projection).reshape(2, 256), availability), 1)
    assert np.array_equal(result, frozen(torch.tensor(manual)).detach().numpy().ravel())
    assert not frozen.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_metric_filter_and_threshold():
    result = run.paired_metrics(np.array([-1., 0., 1., 1.]), np.array([0., 1., 1., 0.]), np.array([1., -1., 0., 1.]))
    assert result['nonneutral_n'] == 3
    assert result['corrections'] == 2
    assert result['harms'] == 1
