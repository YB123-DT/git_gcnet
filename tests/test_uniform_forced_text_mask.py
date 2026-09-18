import torch

from gcnet_missing_m3.train_gcnet import (
    TrainConfig,
    _protocol_rates,
    _uniform_forced_text_mask_tensors,
)


def _batch():
    length, batch = 5, 2
    umask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32)
    qmask = torch.tensor([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.long)
    return [None, None, None, None, None, None, qmask, umask, None, ["c0", "c1"]]


def test_uniform_forced_text_mode_uses_all_evaluation_rates():
    config = TrainConfig(
        dataset="CMUMOSI",
        train_rate_mode="uniform-forced-text",
        training_objective="emotion-only",
        backbone_type="osram",
        osram_bidirectional=False,
        osram_write_step=0.6,
        osram_readout_fusion="flat",
        fusion_type="mean",
    )
    assert _protocol_rates(config) == tuple(i / 10 for i in range(8))


def test_uniform_forced_text_is_deterministic_and_never_empty():
    config = TrainConfig(
        dataset="CMUMOSI",
        seed=66,
        train_rate_mode="uniform-forced-text",
        uniform_forced_text_probability=1.0,
    )
    first = _uniform_forced_text_mask_tensors(config, _batch(), epoch=3, batch_index=2)
    second = _uniform_forced_text_mask_tensors(config, _batch(), epoch=3, batch_index=2)
    for left, right in zip(first[:2], second[:2]):
        assert torch.equal(left, right)
        valid = _batch()[7].transpose(0, 1).bool()
        assert torch.all(left.sum(dim=-1)[valid] >= 1)
        assert torch.all(left[..., 1][valid] == 0)
    assert first[2]["mask_hash"] == second[2]["mask_hash"]
    assert first[2]["selected_forced_text"].sum().item() == 5
