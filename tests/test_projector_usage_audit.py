import numpy as np
import pytest
from experiments.osram_projector_usage_audit_20261025.stats import Stats


def test_condition_specific_channel_is_preserved():
    stats = Stats(3)
    stats.update(np.array([4, 5, 6, 7]), np.array([[1., 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]),
                 np.ones((4, 3)) * np.array([1, 1, 0]))
    result = stats.report([4, 5, 6, 7], min_samples=1)
    assert result['thresholds']['0.1']['all_conditions_indices'] == [2]
    assert stats.report([4, 5, 6, 7])['thresholds']['0.1']['all_conditions_count'] is None


def test_zero_and_invalid_observations():
    stats = Stats(2)
    stats.update(np.array([7, 7]), np.zeros((2, 2)), np.zeros((2, 2)))
    result = stats.report([7], min_samples=1)
    assert result['exact_zero_channels']['7'] == 2
    assert np.isfinite(result['relative_gradient']).all()
    with pytest.raises(ValueError):
        stats.update(np.array([7]), np.array([[np.nan, 0]]), np.zeros((1, 2)))
