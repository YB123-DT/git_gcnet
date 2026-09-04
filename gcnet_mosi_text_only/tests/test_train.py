from pathlib import Path

import torch

from gcnet_mosi_text_only.run_mosi import SEEDS, build_jobs
from gcnet_mosi_text_only.train_mosi import select_best_epoch, select_test_oracle, text_batch


class ForbiddenFeature:
    def to(self, *_args, **_kwargs):
        raise AssertionError("audio/visual must not be moved or read")


def test_text_batch_moves_only_text_and_supervision():
    text = torch.randn(3, 2, 5)
    umask = torch.ones(2, 3)
    labels = torch.randn(2, 3)
    raw = [ForbiddenFeature(), text, ForbiddenFeature(), None, None, None, None, umask, labels, ["a", "b"]]
    view = text_batch(raw, torch.device("cpu"))
    assert view["text"] is text
    assert view["umask"] is umask
    assert view["labels"] is labels


def test_checkpoint_selection_uses_validation_weighted_f1():
    history = [
        {"epoch": 1, "validation": {"weighted_f1": 0.80, "loss": 0.5}},
        {"epoch": 2, "validation": {"weighted_f1": 0.85, "loss": 0.8}},
    ]
    assert select_best_epoch(history) == 2


def test_test_oracle_selects_test_weighted_f1_independently():
    history = [
        {"epoch": 1, "validation": {"weighted_f1": 0.90}, "test": {"weighted_f1": 0.80}},
        {"epoch": 2, "validation": {"weighted_f1": 0.85}, "test": {"weighted_f1": 0.88}},
    ]
    epoch, metrics = select_test_oracle(history)
    assert epoch == 2
    assert metrics["weighted_f1"] == 0.88


def test_runner_builds_exactly_five_text_only_jobs(tmp_path):
    jobs = build_jobs(tmp_path, Path("/python"), Path("/features"), (0, 1, 2, 3))
    assert tuple(job.seed for job in jobs) == SEEDS
    assert len(jobs) == 5
    for job in jobs:
        command = " ".join(job.command).lower()
        assert "gcnet_mosi_text_only.train_mosi" in command
        assert all(token not in command for token in ("jepa", "mmoe", "completion", "missing-rate"))
        assert job.gpu != 4
