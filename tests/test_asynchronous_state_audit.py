from __future__ import annotations

import numpy as np

from scripts.audit_asynchronous_state import (
    asynchronous_state_features,
    dimension_matched_random_history,
)


def test_asynchronous_state_uses_nearest_valid_observations_and_signed_offsets():
    latents = np.zeros((5, 1, 3, 2), dtype=np.float64)
    latents[:, 0, 0] = np.asarray(
        [[1.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 0.0], [5.0, 0.0]],
        dtype=np.float64,
    )
    availability = np.asarray(
        [
            [[1, 0, 0]],
            [[0, 0, 0]],
            [[1, 0, 0]],
            [[0, 0, 0]],
            [[1, 0, 0]],
        ],
        dtype=np.int64,
    )
    valid = np.ones((1, 5), dtype=bool)

    features = asynchronous_state_features(
        latents,
        availability,
        valid,
        decay=1.0,
    )

    # The current audio observation is included with offset 0.  At t=1,
    # observations at offsets -1 and +1 receive equal weight, while the
    # signed offset is zero by symmetry.
    assert features.shape == (5, 1, 3 * (2 + 1))
    np.testing.assert_allclose(features[1, 0, :2], [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(features[1, 0, 2], 0.0, atol=1e-12)


def test_random_history_control_is_deterministic_and_dimension_matched():
    async_features = np.zeros((4, 2, 15), dtype=np.float64)
    first = dimension_matched_random_history(async_features, seed=66)
    second = dimension_matched_random_history(async_features, seed=66)
    different = dimension_matched_random_history(async_features, seed=67)

    assert first.shape == async_features.shape
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
