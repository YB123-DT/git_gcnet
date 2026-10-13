import torch
from torch import nn

from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
    SOURCE_COMBINATIONS,
    materialize_legacy_masks,
    runtime_config_payload,
    replay_flat_hidden,
    select_flat_sources,
)
from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import (
    LegacyFixedMaskSchedule,
)


def test_select_flat_sources_zeros_exactly_the_disabled_groups():
    local = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    flat_input = torch.tensor(
        [[[1.0, 2.0, 10.0, 20.0, 30.0, 40.0]],
         [[3.0, 4.0, 11.0, 21.0, 31.0, 41.0]]]
    )
    umask = torch.tensor([[1.0, 0.0]])

    expected = {
        "L": [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        "B": [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        "G": [0.0, 0.0, 0.0, 20.0, 30.0, 40.0],
        "LB": [1.0, 2.0, 10.0, 0.0, 0.0, 0.0],
        "LG": [1.0, 2.0, 0.0, 20.0, 30.0, 40.0],
        "BG": [0.0, 0.0, 10.0, 20.0, 30.0, 40.0],
        "LBG": [1.0, 2.0, 10.0, 20.0, 30.0, 40.0],
    }
    assert tuple(SOURCE_COMBINATIONS) == tuple(expected)
    for name, wanted in expected.items():
        selected_local, selected_input = select_flat_sources(
            local, flat_input, umask, local_dim=2, context_dim=1,
            combination=name,
        )
        torch.testing.assert_close(selected_input[0, 0], torch.tensor(wanted))
        torch.testing.assert_close(
            selected_local[0, 0],
            local[0, 0] if "L" in name else torch.zeros_like(local[0, 0]),
        )
        assert torch.count_nonzero(selected_input[1]) == 0
        assert torch.count_nonzero(selected_local[1]) == 0


def test_full_combination_exactly_replays_original_flat_formula():
    torch.manual_seed(7)
    local = torch.randn(3, 2, 2)
    flat_input = torch.randn(3, 2, 6)
    umask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    local_skip = nn.Linear(2, 5)
    emotion_adapter = nn.Linear(6, 5)
    emotion_norm = nn.LayerNorm(5)
    valid = umask.T.bool().unsqueeze(-1)
    expected = emotion_norm(local_skip(local) + emotion_adapter(flat_input))
    expected = torch.where(valid, expected, torch.zeros_like(expected))

    actual = replay_flat_hidden(
        local_skip=local_skip,
        emotion_adapter=emotion_adapter,
        emotion_norm=emotion_norm,
        local=local,
        flat_input=flat_input,
        umask=umask,
        local_dim=2,
        context_dim=1,
        combination="LBG",
    )
    torch.testing.assert_close(actual, expected)


def test_legacy_checkpoint_protocol_is_only_sanitized_for_config_construction():
    source = {
        "missing_mask_protocol": "gcnet-legacy-fixed-v1",
        "dataset": "MELD",
    }
    runtime = runtime_config_payload(source)
    assert source["missing_mask_protocol"] == "gcnet-legacy-fixed-v1"
    assert runtime["missing_mask_protocol"] == "conversation-mask-v2"
    assert runtime["dataset"] == "MELD"


def test_frozen_legacy_schedule_preserves_observed_budget_and_never_goes_empty():
    schedule = LegacyFixedMaskSchedule("MELD", "test", 1, 0.6, 12345)
    generated = schedule.generate("dia2", length=12, valid_length=10, side="host")
    assert int(generated.availability[:10].sum()) == round(30 * 0.4)
    assert torch.as_tensor(generated.availability[:10]).sum(dim=1).min() >= 1
    assert generated.availability[10:].sum() == 0


def test_legacy_batch_materialization_matches_source_training_contract():
    schedule = LegacyFixedMaskSchedule("MELD", "test", 1, 0.5, 2468)
    umask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    host, guest = materialize_legacy_masks(
        schedule, ["dia1", "dia2"], umask, epoch=0
    )
    assert host.shape == guest.shape == (3, 2, 3)
    torch.testing.assert_close(host, guest)
    assert torch.count_nonzero(host[2, 0]) == 0
    assert host[:, 1].sum(dim=1).min() >= 1
