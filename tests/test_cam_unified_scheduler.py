import pytest

from experiments.cam_unified_20261030.scheduler import admissible_gpus, pending_variants


def test_bad_card_is_never_admitted_and_memory_is_reserved():
    cards = [
        {"index": 4, "uuid": "GPU-bad", "free_mib": 32000},
        {"index": 7, "uuid": "GPU-good", "free_mib": 25000},
        {"index": 2, "uuid": "GPU-busy", "free_mib": 3000},
    ]
    assert admissible_gpus(cards, reserve_mib=4096, job_mib=6000) == [
        {"index": 7, "uuid": "GPU-good", "slots": 3}
    ]


def test_queue_is_finite_and_never_restarts_completed_or_running_arms():
    assert pending_variants({"A0": "complete", "A1": "running"}) == ["A2", "A3"]
    assert pending_variants({"A0": "failed"}) == ["A1", "A2", "A3"]


def test_unknown_arm_or_unmeasured_cost_is_rejected():
    with pytest.raises(ValueError):
        pending_variants({"B0": "complete"})
    with pytest.raises(ValueError):
        admissible_gpus([], job_mib=0)
