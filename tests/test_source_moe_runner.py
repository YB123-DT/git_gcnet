import pytest

from experiments.osram_source_moe_mosi_20261025.run import learning_rate, validate_mode


def test_finite_scope_and_schedule():
    assert learning_rate(0) == learning_rate(99) == 3e-4
    assert learning_rate(100) == learning_rate(149) == 3e-4 / 10
    for epoch in (-1, 150):
        with pytest.raises(ValueError):
            learning_rate(epoch)
    for mode in ('residual', 'replace'):
        validate_mode(mode)
    with pytest.raises(ValueError):
        validate_mode('flat')
