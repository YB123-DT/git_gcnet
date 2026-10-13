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


def test_count_padding_side_and_evaluation_freeze():
    item = schedule(0.6)
    host = item.generate("dia2", 12, "host", epoch=4, valid_length=10)
    guest = item.generate("dia2", 12, "guest", epoch=4, valid_length=10)
    assert int(host.availability[:10].sum()) == round(30 * 0.4)
    assert np.all(host.availability[:10].sum(axis=1) >= 1)
    assert np.all(host.availability[10:] == 0)
    assert np.array_equal(host.availability, guest.availability)
    test = schedule(0.5, "test")
    assert np.array_equal(
        test.generate("dia3", 20, "host", 0).availability,
        test.generate("dia3", 20, "guest", 49).availability,
    )
