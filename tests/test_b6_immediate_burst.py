"""User-approved immediate launch consumes, rather than expands, the20 trials."""
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'experiments/osram_meld_b6_threshold_hpo_20261015/immediate_burst.py'
    spec = importlib.util.spec_from_file_location('b6_immediate', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_immediate_budget_and_asymmetric_card_capacity():
    m = module()
    assert m.burst_count(5) == 5
    assert m.burst_count(8) == 2
    assert m.burst_count(10) == 0
    assert m.burst_count(20) == 0
    assert m.CAPACITY == {0: 3, 3: 1, 5: 1}
    for n in (-1, 21):
        with pytest.raises(ValueError):
            m.burst_count(n)


def test_scheduler_never_launches_on_gpu2_or_bad_gpu4(tmp_path):
    m = module()
    from experiments.osram_meld_b6_threshold_hpo_20261015 import run as frozen
    s = m.make_scheduler(frozen, tmp_path / 'out', tmp_path / 'source', 1005.)
    assert s.gpus == (0, 3, 5)
    assert s.capacity == m.CAPACITY
    assert s.code_root == (tmp_path / 'source').resolve()
