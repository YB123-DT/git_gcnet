from collections import Counter

from experiments.osram_meld_nested_screen_20261012.dispatch import build_jobs


def test_meld_screen_contains_six_anchors_and_64_unique_grid_runs():
    jobs = build_jobs()
    assert len(jobs) == 70
    assert len({job["name"] for job in jobs}) == 70
    assert Counter(job["kind"] for job in jobs) == {"anchor": 6, "grid": 64}
    assert {(job["variant"], job["seed"]) for job in jobs if job["kind"] == "anchor"} == {
        (variant, seed) for variant in ("flat", "nested") for seed in (66, 67, 68)
    }


def test_meld_screen_is_exactly_five_balanced_waves():
    jobs = build_jobs()
    assert Counter(job["wave"] for job in jobs) == {wave: 14 for wave in range(5)}
    assert Counter((job["wave"], job["gpu"]) for job in jobs) == {
        (wave, gpu): 7 for wave in range(5) for gpu in (0, 1)
    }
