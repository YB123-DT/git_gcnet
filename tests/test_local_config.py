from pathlib import Path

import config


def test_dataset_root_can_be_overridden(monkeypatch) -> None:
    repository = Path(config.__file__).resolve().parent
    assert Path(config.SAVED_ROOT).is_absolute()
    assert Path(config.PATH_TO_FEATURES["IEMOCAPSix"]).name == "features"
    assert repository.name == "modality-jepa"
