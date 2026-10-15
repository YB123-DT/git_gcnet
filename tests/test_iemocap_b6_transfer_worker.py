"""Bookkeeping must not mix MELD classes or re-sampled UA into IEMOCAP."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'experiments/osram_iemocap_b6_transfer_20261015/worker.py'
    spec = importlib.util.spec_from_file_location('iemocap_transfer_worker', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_four_class_ua_includes_all_four_classes():
    result = module().class_metrics(np.array([0, 0, 1, 2, 3]),
        np.eye(4)[[0, 1, 1, 2, 2]], 4)
    assert result['ACC'] == pytest.approx(60)
    assert result['UA'] == pytest.approx(62.5)
    assert result['support'] == [2, 1, 1, 1]


def test_six_class_missing_support_is_explicit():
    result = module().class_metrics(np.array([0, 1]), np.eye(6)[[0, 1]], 6)
    assert result['ACC'] == 100
    assert result['UA'] == pytest.approx(100 / 3)


def test_invalid_logits_and_labels_rejected():
    implementation = module()
    with pytest.raises(ValueError):
        implementation.class_metrics(np.array([6]), np.ones((1, 6)), 6)
    with pytest.raises(ValueError):
        implementation.class_metrics(np.array([0]), np.full((1, 4), np.nan), 4)


def test_rng_repair_updates_cached_random_alias():
    from types import SimpleNamespace
    random_api = SimpleNamespace(device_count=lambda: 0)
    cuda = SimpleNamespace(device_count=lambda: 0, random=random_api)
    fake = SimpleNamespace(cuda=cuda, _C=SimpleNamespace(_cuda_getDeviceCount=lambda: 1))
    report = module().repair_uuid_rng(fake)
    assert fake.cuda.device_count() == fake.cuda.random.device_count() == 1
    assert report['original_random_count'] == 0


def test_aggregate_uses_all_eight_and_high_three():
    rates = {str(i / 10): {k: float(i) for k in ('ACC', 'UA', 'WF1', 'CE')}
             for i in range(8)}
    assert module().aggregate(rates)['mean']['UA'] == 3.5
    assert module().aggregate(rates)['high']['ACC'] == 6
