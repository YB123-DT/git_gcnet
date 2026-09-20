import pytest

from gcnet_missing_m3.train_gcnet import _checkpoint_selection_metric


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("IEMOCAPFour", "accuracy"),
        ("IEMOCAPSix", "accuracy"),
        ("CMUMOSI", "weighted_f1"),
        ("CMUMOSEI", "weighted_f1"),
    ],
)
def test_checkpoint_selection_metric_matches_dataset_protocol(dataset, expected):
    assert _checkpoint_selection_metric(dataset) == expected
