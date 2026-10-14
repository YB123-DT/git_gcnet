import numpy as np
import torch

from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
    scatter_saved_availability,
)


def test_scatter_saved_availability_restores_valid_flattening_order() -> None:
    umask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    saved = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    restored, cursor = scatter_saved_availability(saved, 0, umask)

    assert cursor == 3
    torch.testing.assert_close(
        restored.transpose(0, 1)[umask.bool()], torch.from_numpy(saved)
    )
    assert restored[~umask.T.bool()].count_nonzero() == 0


def test_scatter_saved_availability_rejects_truncated_artifact() -> None:
    umask = torch.ones(1, 2)
    saved = np.ones((1, 3), dtype=np.float32)

    try:
        scatter_saved_availability(saved, 0, umask)
    except ValueError as error:
        assert "ended before" in str(error)
    else:
        raise AssertionError("truncated artifact was accepted")
