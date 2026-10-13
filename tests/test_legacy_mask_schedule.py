import numpy as np
import pytest

from gcnet_modality_jepa.legacy_mask_schedule import LegacyFixedMaskSchedule


def schedule(rate, split="train"):
    return LegacyFixedMaskSchedule("MELD", split, 1, rate, 12345)


def test_complete_and_high_missing_match_gcnet_boundary():
    complete = schedule(0.0).generate("dia1", 9, "host", epoch=2)
    high = schedule(0.7).generate("dia1", 9, "host", epoch=2)
    assert np.all(complete.availability == 1)
    assert np.all(high.availability.sum(axis=1) == 1)
    assert high.realized_missing_rate == pytest.approx(2.0 / 3.0)


def test_count_matches_requested_observed_budget_and_padding_is_zero():
    generated = schedule(0.6).generate(
        "dia2", length=12, valid_length=10, side="host", epoch=4
    )
    assert int(generated.availability[:10].sum()) == round(30 * 0.4)
    assert np.all(generated.availability[:10].sum(axis=1) >= 1)
    assert np.all(generated.availability[10:] == 0)


def test_side_is_shared_and_training_epoch_changes_deterministically():
    item = schedule(0.4)
    host = item.generate("dia3", 20, "host", epoch=3).availability
    guest = item.generate("dia3", 20, "guest", epoch=3).availability
    again = item.generate("dia3", 20, "host", epoch=3).availability
    next_epoch = item.generate("dia3", 20, "host", epoch=4).availability
    assert np.array_equal(host, guest)
    assert np.array_equal(host, again)
    assert not np.array_equal(host, next_epoch)


def test_evaluation_is_frozen_at_epoch_zero():
    item = schedule(0.5, split="test")
    first = item.generate("dia4", 20, "host", epoch=0)
    later = item.generate("dia4", 20, "guest", epoch=49)
    assert first.epoch == later.epoch == 0
    assert first.schedule_hash == later.schedule_hash
    assert np.array_equal(first.availability, later.availability)


