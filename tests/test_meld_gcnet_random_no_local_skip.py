import importlib.util
from pathlib import Path

import pytest


RUNNER = (
    Path(__file__).parents[1]
    / "experiments/osram_meld_gcnet_random_no_local_skip_20261014/run.py"
)
SPEC = importlib.util.spec_from_file_location("meld_gcnet_random_no_skip", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_corrected_settings_changes_only_local_skip() -> None:
    source = {
        "missing_mask_protocol": "gcnet-random-mask-v1",
        "train_missing_rates": [0.0, 0.1, 0.2],
        "osram_disable_local_skip": False,
        "seed": 66,
    }
    corrected = MODULE.corrected_settings(source)

    assert corrected == {
        "missing_mask_protocol": "gcnet-random-mask-v1",
        "train_missing_rates": (0.0, 0.1, 0.2),
        "osram_disable_local_skip": True,
        "seed": 66,
    }
    assert source["osram_disable_local_skip"] is False


@pytest.mark.parametrize(
    "protocol", ["conversation-mask-v2", "gcnet-legacy-fixed-v1", None]
)
def test_corrected_settings_rejects_noncanonical_masks(protocol: str | None) -> None:
    with pytest.raises(ValueError, match="canonical original-GCNet sampler"):
        MODULE.corrected_settings(
            {
                "missing_mask_protocol": protocol,
                "train_missing_rates": [0.0],
            }
        )
