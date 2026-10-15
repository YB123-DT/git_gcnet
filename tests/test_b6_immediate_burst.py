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
    assert m.burst_count(8) == 5
    assert m.burst_count(10) == 5
    assert m.burst_count(19) == 1
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


def test_existing_card_jobs_are_not_mistaken_for_new_burst_capacity(tmp_path):
    import json
    m = module()
    from experiments.osram_meld_b6_threshold_hpo_20261015 import run as frozen
    root = tmp_path / 'out/queue'
    root.mkdir(parents=True)
    (root / 'QUEUE.json').write_text(json.dumps({'jobs': {
        'old3a': {'status': 'running', 'gpu': 3},
        'old3b': {'status': 'running', 'gpu': 3},
        'old5a': {'status': 'running', 'gpu': 5},
        'old5b': {'status': 'running', 'gpu': 5}}, 'capacity': {'3': 1, '5': 1}}))
    s = m.make_scheduler(frozen, tmp_path / 'out', tmp_path / 'source', 1005.)
    assert s.capacity == {0: 3, 3: 3, 5: 3}


def test_rebalance_keeps_one_on_gpu3_and5_and_moves_excess_to_gpu2(tmp_path):
    m = module()
    from experiments.osram_meld_b6_threshold_hpo_20261015 import run as frozen
    s = m.make_scheduler(frozen, tmp_path / 'out', tmp_path / 'source', 1005., rebalance=True)
    assert s.gpus == (0, 2, 3)
    assert s.capacity == {0: 3, 2: 6, 3: 1, 5: 1}
    assert s.code_root == (tmp_path / 'source').resolve()


def test_one_active_plan_pauses_excess_then_resumes_next_owned_job():
    m = module()
    rows = [{'id': 'trial7', 'gpu': 5, 'started_at': '01', 'status': 'running'},
            {'id': 'trial10', 'gpu': 5, 'started_at': '02', 'status': 'running'},
            {'id': 'trial14', 'gpu': 5, 'started_at': '03', 'status': 'running'},
            {'id': 'othergpu', 'gpu': 0, 'started_at': '00', 'status': 'running'}]
    assert m.activity_plan(rows, {'trial7': 'R', 'trial10': 'R', 'trial14': 'S', 'othergpu': 'R'}) == [
        ('trial10', 'pause'), ('trial14', 'pause')]
    rows[0]['status'] = 'complete'
    assert m.activity_plan(rows, {'trial10': 'T', 'trial14': 'T'}) == [('trial10', 'resume')]
    assert m.activity_plan(rows, {}) == []  # no unverified PID can be signalled


def test_limiter_cleanup_is_safe_before_background_thread_started(tmp_path):
    m = module()
    from experiments.osram_meld_b6_threshold_hpo_20261015 import run as frozen
    s = m.make_scheduler(frozen, tmp_path / 'out', tmp_path / 'source', 1005.)
    limiter = m.ActivityLimiter(s, tmp_path / 'out', frozen.write_json)
    limiter.close()
