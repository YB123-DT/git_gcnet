import pytest

from gcnet_missing_m3_soft_ordinal import train_gcnet as soft_train


def test_soft_ordinal_version_entry_injects_locked_task_mode(monkeypatch):
    captured = {}

    def shared_main(argv=None):
        captured["argv"] = list(argv)

    monkeypatch.setattr(soft_train.base_train, "main", shared_main)

    soft_train.main(["--audio-feature", "a"])

    assert captured["argv"][-2:] == [
        "--mosi-task-mode",
        "soft-ordinal",
    ]


def test_soft_ordinal_version_rejects_task_mode_override():
    with pytest.raises(ValueError, match="owns mosi_task_mode"):
        soft_train.main(["--mosi-task-mode", "regression"])
