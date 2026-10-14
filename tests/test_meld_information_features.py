import numpy as np
import pytest

from experiments.osram_meld_information_availability_20261014.features import (
    match_donors,
    mask_forward_memory,
    strict_past_summaries,
    validate_cache,
)


def _rows():
    return {
        "conversation_ids": np.asarray(["c1", "c1", "c1", "c2", "c2"]),
        "utterance_indices": np.asarray([0, 1, 2, 0, 1]),
        "speaker": np.asarray([0, 1, 0, 0, 0]),
        "local": np.asarray([[1, 0], [3, 0], [5, 0], [10, 0], [14, 0]], dtype=np.float32),
        "availability": np.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=np.int64),
    }


def test_strict_past_summaries_exclude_current_and_filter_speaker():
    result = strict_past_summaries(_rows())
    np.testing.assert_array_equal(result["all_count"], [0, 1, 2, 0, 1])
    np.testing.assert_array_equal(result["same_speaker_count"], [0, 0, 1, 0, 1])
    np.testing.assert_array_equal(result["all_present"], [0, 1, 1, 0, 1])
    np.testing.assert_array_equal(result["same_speaker_present"], [0, 0, 1, 0, 1])
    np.testing.assert_allclose(result["all_mean"], [[0, 0], [1, 0], [2, 0], [0, 0], [10, 0]])
    np.testing.assert_allclose(result["same_speaker_mean"], [[0, 0], [0, 0], [1, 0], [0, 0], [10, 0]])


def test_strict_past_rejects_noncontiguous_or_unsorted_rows():
    rows = _rows()
    rows["utterance_indices"] = np.asarray([0, 2, 1, 0, 1])
    with pytest.raises(ValueError, match="contiguous temporal order"):
        strict_past_summaries(rows)


def test_mask_forward_memory_removes_backward_and_inactive_gaps():
    base = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024)
    gaps = np.ones((2, 3, 1024), dtype=np.float32)
    availability = np.asarray([[1, 0, 1], [0, 1, 0]])
    masked_base, masked_gaps = mask_forward_memory(base, gaps, availability)
    assert masked_base.shape == (2, 512)
    assert masked_gaps.shape == (2, 3, 512)
    np.testing.assert_array_equal(masked_base, base[:, :512])
    assert np.count_nonzero(masked_gaps[0, 0]) == 0
    assert np.count_nonzero(masked_gaps[0, 1]) == 512
    assert np.count_nonzero(masked_gaps[0, 2]) == 0
    assert np.count_nonzero(masked_gaps[1, 0]) == 512
    assert np.count_nonzero(masked_gaps[1, 1]) == 0
    assert np.count_nonzero(masked_gaps[1, 2]) == 512


def test_donors_are_split_local_cross_conversation_and_label_free():
    rows = _rows()
    history = strict_past_summaries(rows)
    rows.update(history)
    first = match_donors(rows, seed=77)
    rows_with_changed_labels = dict(rows, labels=np.asarray([6, 5, 4, 3, 2]))
    second = match_donors(rows_with_changed_labels, seed=77)
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.included, second.included)
    for target, donor in enumerate(first.indices):
        if donor < 0:
            continue
        assert rows["conversation_ids"][target] != rows["conversation_ids"][donor]
        np.testing.assert_array_equal(rows["availability"][target], rows["availability"][donor])
        assert rows["all_present"][target] == rows["all_present"][donor]
        assert rows["same_speaker_present"][target] == rows["same_speaker_present"][donor]


def test_unmatched_donors_are_excluded_symmetrically():
    rows = _rows()
    rows.update(strict_past_summaries(rows))
    match = match_donors(rows, seed=66)
    assert match.total == 5
    assert match.included.sum() == match.matched
    assert np.all(match.indices[~match.included] == -1)


def test_cache_validation_rejects_inactive_gap_leakage():
    rows = _rows()
    rows["local"] = np.pad(rows["local"], ((0, 0), (0, 254)))
    rows.update(strict_past_summaries(rows))
    rows.update(
        labels=np.asarray([0, 1, 2, 3, 4]),
        base=np.zeros((5, 512), dtype=np.float32),
        gap=np.zeros((5, 3, 512), dtype=np.float32),
        prediction=np.zeros((5, 7), dtype=np.float32),
    )
    validate_cache(rows)
    rows["gap"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="inactive Gap leakage"):
        validate_cache(rows)
