import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

import gcnet_missing_m3.model as missing_m3_model
import gcnet_missing_m3.train_gcnet as train_gcnet
import gcnet_modality_jepa.model as base_graph_model
from gcnet_missing_m3.loss import MissingM3Loss, missing_m3_loss
from gcnet_missing_m3.mixed_rate import (
    MISSING_RATES,
    BalancedBatchRateSchedule,
    mean_validation_weighted_f1,
    select_best_epoch,
)
from gcnet_missing_m3.model import (
    ContextualM3Predictor,
    DualGateTopKMMoE,
    LocalContextResidualFusion,
    MissingLatentResidualFusion,
    MissingM3GraphModel,
    ObservedSetEncoder,
    RawResidualObservedEncoder,
)
from gcnet_missing_m3.train_gcnet import (
    TrainConfig,
    _dataset_shape,
    _metrics,
    _task_loss,
    build_parser,
)


ASSERT_CLOSE = getattr(torch.testing, "assert_close", torch.testing.assert_allclose)


def test_graph_relation_ids_do_not_depend_on_python_hash_seed():
    script = (
        "import json,torch;"
        "from gcnet_modality_jepa.graph import batch_graphify;"
        "f=torch.zeros(2,1,1,3);q=torch.zeros(1,2);"
        "t=batch_graphify(f,q,[2],2,1,1,'temporal',True)[3];"
        "s=batch_graphify(f,q,[2],2,1,1,'speaker',True)[3];"
        "print(json.dumps({'temporal':t,'speaker':s},sort_keys=True))"
    )
    expected = {
        "temporal": {"past": 0, "now": 1, "future": 2},
        "speaker": {"00": 0, "01": 1, "10": 2, "11": 3},
    }
    for hash_seed in ("0", "1", "3", "6"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        environment["PYTHONPATH"] = os.pathsep.join(sys.path)
        output = subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=environment,
            text=True,
        )
        assert json.loads(output) == expected


def test_paper_faithful_mmoe_adds_only_branch_specific_parameters():
    legacy = DualGateTopKMMoE(4, num_experts=2, top_k=1, dropout=0.0)
    explicit_legacy = DualGateTopKMMoE(
        4, num_experts=2, top_k=1, dropout=0.0, variant="dual-gate"
    )
    paper = DualGateTopKMMoE(
        4, num_experts=2, top_k=1, dropout=0.0, variant="paper-faithful"
    )

    assert legacy.state_dict().keys() == explicit_legacy.state_dict().keys()
    assert "reg_task_embedding" not in legacy.state_dict()
    assert "cl_task_embedding" not in legacy.state_dict()
    assert paper.reg_gate is not paper.cl_gate
    assert "reg_task_embedding" in paper.state_dict()
    assert "cl_task_embedding" in paper.state_dict()
    assert "reg_norm.weight" in paper.state_dict()
    assert "cl_norm.weight" in paper.state_dict()


def test_paper_faithful_mmoe_preserves_each_task_input_with_residual():
    class ZeroExpert(torch.nn.Module):
        def forward(self, value):
            return torch.zeros_like(value)

    mmoe = DualGateTopKMMoE(
        3, num_experts=2, top_k=1, dropout=0.0, variant="paper-faithful"
    ).eval()
    mmoe.experts = torch.nn.ModuleList([ZeroExpert(), ZeroExpert()])
    with torch.no_grad():
        mmoe.source_embedding.weight.zero_()
        mmoe.target_embedding.weight.zero_()
        mmoe.reg_task_embedding.copy_(torch.tensor([0.1, 0.2, 0.3]))
        mmoe.cl_task_embedding.copy_(torch.tensor([-0.3, -0.2, -0.1]))
        for head in (*mmoe.reg_heads, *mmoe.cl_heads):
            head.weight.copy_(torch.eye(3))
            head.bias.zero_()

    value = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    reg, cl = mmoe(value, source_index=0, target_index=1)

    ASSERT_CLOSE(reg, value + mmoe.reg_task_embedding, rtol=0, atol=1e-7)
    ASSERT_CLOSE(cl, value + mmoe.cl_task_embedding, rtol=0, atol=1e-7)


def test_paper_faithful_routing_uses_full_softmax_and_exposes_statistics():
    mmoe = DualGateTopKMMoE(
        4, num_experts=2, top_k=1, dropout=0.0, variant="paper-faithful"
    )
    with torch.no_grad():
        mmoe.reg_gate.weight.zero_()
        mmoe.reg_gate.bias.copy_(torch.tensor([0.0, torch.log(torch.tensor(3.0))]))

    route = mmoe._route(
        torch.zeros(2, 4), mmoe.reg_gate, branch_index=0
    )
    expected = torch.tensor([[0.0, 0.75], [0.0, 0.75]])
    ASSERT_CLOSE(route, expected, rtol=0, atol=1e-7)

    statistics = mmoe.routing_statistics()
    ASSERT_CLOSE(
        statistics["selection_count"][0],
        torch.tensor([0.0, 2.0], dtype=torch.float64),
    )
    ASSERT_CLOSE(
        statistics["probability_mass"][0],
        torch.tensor([0.0, 1.5], dtype=torch.float64),
    )
    assert torch.isfinite(statistics["entropy"]).all()


def test_paper_faithful_mmoe_variant_is_exposed_by_cli():
    args = build_parser().parse_args(
        [
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--output-dir",
            "out",
            "--mmoe-variant",
            "paper-faithful",
        ]
    )

    assert args.mmoe_variant == "paper-faithful"


def test_missing_latent_residual_averages_only_real_missing_targets_and_zeros_padding():
    class Scale(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, latent):
            return latent * self.value

    fusion = MissingLatentResidualFusion(latent_dim=2, hidden_dim=2)
    fusion.target_projections = torch.nn.ModuleList(
        [Scale(1.0), Scale(2.0), Scale(3.0)]
    )
    predictions = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
            [[[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]]],
        ]
    )
    target_mask = torch.tensor(
        [[[0, 1, 1]], [[0, 0, 0]], [[1, 1, 0]]], dtype=torch.bool
    )
    umask = torch.tensor([[1.0, 1.0, 0.0]])

    residual = fusion(predictions, target_mask, umask)

    expected = (
        torch.tanh(torch.tensor([0.0, 2.0]))
        + torch.tanh(torch.tensor([3.0, 3.0]))
    ) / 2.0
    ASSERT_CLOSE(residual[0, 0], expected)
    assert torch.count_nonzero(residual[1]) == 0
    assert torch.count_nonzero(residual[2]) == 0


def test_classification_completion_executes_predictor_at_inference_without_returning_it(
    monkeypatch,
):
    model = MissingM3GraphModel(
        **_model_arguments(), classification_completion=True
    ).eval()
    inputs = _model_inputs()
    fixed_graph_hidden = torch.randn(3, 2, 10)
    monkeypatch.setattr(
        model, "encode_hidden", lambda *_args, **_kwargs: fixed_graph_hidden
    )
    calls = []
    original = model.missing_predictor.forward

    def capture(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.missing_predictor, "forward", capture)
    _, graph_hidden, _, returned_predictions = model(
        [inputs[0]], *inputs[1:], predict_missing=False
    )

    assert calls == [True]
    assert returned_predictions is None
    ASSERT_CLOSE(graph_hidden, fixed_graph_hidden, rtol=0, atol=0)


def test_classification_completion_is_opt_in_and_exposed_by_cli():
    default = MissingM3GraphModel(**_model_arguments())
    enabled = MissingM3GraphModel(
        **_model_arguments(), classification_completion=True
    )
    assert not hasattr(default, "missing_latent_fusion")
    assert any(
        name.startswith("missing_latent_fusion.")
        for name, _ in enabled.named_parameters()
    )

    args = build_parser().parse_args(
        [
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--output-dir",
            "out",
            "--classification-completion",
        ]
    )
    assert args.classification_completion is True


def _all_patterns():
    return torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    ).unsqueeze(1)


def test_observed_set_encoder_supports_seven_patterns_and_ignores_missing_values():
    torch.manual_seed(11)
    encoder = ObservedSetEncoder((2, 3, 4), latent_dim=8, dropout=0.0).eval()
    availability = _all_patterns()
    umask = torch.ones(1, 7)
    features = torch.randn(7, 1, 9)
    changed = features.clone()
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    changed[expanded == 0] += 10_000.0

    first_node, first_latents = encoder(features, availability, umask)
    second_node, second_latents = encoder(changed, availability, umask)

    assert first_node.shape == (7, 1, 8)
    assert set(first_latents) == {"audio", "text", "visual"}
    ASSERT_CLOSE(first_node, second_node, rtol=0, atol=0)
    for index, name in enumerate(("audio", "text", "visual")):
        ASSERT_CLOSE(first_latents[name], second_latents[name], rtol=0, atol=0)
        assert torch.count_nonzero(first_latents[name][~availability[..., index].bool()]) == 0


def test_observed_set_encoder_zeros_padding():
    encoder = ObservedSetEncoder((2, 3, 4), latent_dim=8, dropout=0.0).eval()
    availability = torch.tensor(
        [[[1, 0, 0]], [[1, 1, 1]], [[0, 0, 0]]], dtype=torch.float32
    )
    umask = torch.tensor([[1.0, 1.0, 0.0]])

    node, latents = encoder(torch.randn(3, 1, 9), availability, umask)

    assert torch.count_nonzero(node[2]) == 0
    assert all(torch.count_nonzero(value[2]) == 0 for value in latents.values())


def test_modality_track_encoder_supports_seven_patterns_without_missing_leakage():
    encoder_class = getattr(missing_m3_model, "ModalityTrackEncoder", None)
    assert encoder_class is not None, "ModalityTrackEncoder is not implemented"
    torch.manual_seed(101)
    encoder = encoder_class((2, 3, 4), latent_dim=8, dropout=0.0).eval()
    availability = _all_patterns()
    umask = torch.ones(1, 7)
    features = torch.randn(7, 1, 9)
    changed = features.clone()
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    changed[expanded == 0] += 10_000.0

    first_tracks, first_latents = encoder(features, availability, umask)
    second_tracks, second_latents = encoder(changed, availability, umask)

    assert set(first_tracks) == {"audio", "text", "visual"}
    assert set(first_latents) == {"audio", "text", "visual"}
    for index, name in enumerate(("audio", "text", "visual")):
        selected = availability[..., index].bool()
        assert first_tracks[name].shape == (7, 1, 8)
        ASSERT_CLOSE(first_tracks[name], second_tracks[name], rtol=0, atol=0)
        ASSERT_CLOSE(first_latents[name], second_latents[name], rtol=0, atol=0)
        assert torch.count_nonzero(first_tracks[name][~selected]) == 0
        assert torch.count_nonzero(first_latents[name][~selected]) == 0


def test_modality_track_encoder_zeros_every_track_at_padding():
    encoder_class = getattr(missing_m3_model, "ModalityTrackEncoder", None)
    assert encoder_class is not None, "ModalityTrackEncoder is not implemented"
    encoder = encoder_class((2, 3, 4), latent_dim=8, dropout=0.0).eval()
    availability = torch.tensor(
        [[[1, 0, 1]], [[1, 1, 1]], [[0, 0, 0]]], dtype=torch.float32
    )
    umask = torch.tensor([[1.0, 1.0, 0.0]])

    tracks, latents = encoder(torch.randn(3, 1, 9), availability, umask)

    assert all(torch.count_nonzero(value[2]) == 0 for value in tracks.values())
    assert all(torch.count_nonzero(value[2]) == 0 for value in latents.values())


def test_post_graph_track_fusion_excludes_missing_tracks_and_zeros_padding():
    fusion_class = getattr(missing_m3_model, "PostGraphTrackFusion", None)
    assert fusion_class is not None, "PostGraphTrackFusion is not implemented"
    torch.manual_seed(103)
    fusion = fusion_class(hidden_dim=6, dropout=0.0).eval()
    availability = torch.tensor(
        [
            [[1, 0, 1], [0, 1, 0]],
            [[1, 1, 0], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    umask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    tracks = {
        name: torch.randn(2, 2, 6)
        for name in ("audio", "text", "visual")
    }
    changed = {name: value.clone() for name, value in tracks.items()}
    for index, name in enumerate(("audio", "text", "visual")):
        changed[name][~availability[..., index].bool()] += 10_000.0

    first = fusion(tracks, availability, umask)
    second = fusion(changed, availability, umask)

    assert first.shape == (2, 2, 6)
    ASSERT_CLOSE(first, second, rtol=0, atol=0)
    assert torch.count_nonzero(first[~umask.T.bool()]) == 0


def test_pcir_initially_returns_zero_for_seven_patterns_and_padding():
    residual_class = getattr(
        missing_m3_model, "PatternConditionedInteractionResidual", None
    )
    assert residual_class is not None, "PCIR is not implemented"
    module = residual_class(
        latent_dim=8,
        pair_embedding_dim=4,
        pair_rank=5,
        residual_hidden_dim=6,
    ).eval()
    availability = torch.cat(
        [_all_patterns(), torch.zeros(1, 1, 3)], dim=0
    )
    umask = torch.tensor([[1.0] * 7 + [0.0]])
    latents = {
        name: torch.randn(8, 1, 8)
        for name in ("audio", "text", "visual")
    }

    residual = module(latents, availability, umask)

    assert residual.shape == (8, 1, 8)
    ASSERT_CLOSE(residual, torch.zeros_like(residual), rtol=0, atol=0)


def test_pcir_ignores_missing_latents_after_residual_is_enabled():
    residual_class = getattr(
        missing_m3_model, "PatternConditionedInteractionResidual", None
    )
    assert residual_class is not None, "PCIR is not implemented"
    torch.manual_seed(131)
    module = residual_class(
        latent_dim=8,
        pair_embedding_dim=4,
        pair_rank=5,
        residual_hidden_dim=6,
    ).eval()
    with torch.no_grad():
        torch.nn.init.normal_(module.residual_mlp[-1].weight)
        torch.nn.init.normal_(module.residual_mlp[-1].bias)
    availability = _all_patterns()
    umask = torch.ones(1, 7)
    latents = {
        name: torch.randn(7, 1, 8)
        for name in ("audio", "text", "visual")
    }
    changed = {name: value.clone() for name, value in latents.items()}
    for index, name in enumerate(("audio", "text", "visual")):
        changed[name][~availability[..., index].bool()] += 10_000.0

    first = module(latents, availability, umask)
    second = module(changed, availability, umask)

    ASSERT_CLOSE(first, second, rtol=0, atol=0)


def test_pcir_pair_mask_activates_only_observed_pairs():
    residual_class = getattr(
        missing_m3_model, "PatternConditionedInteractionResidual", None
    )
    assert residual_class is not None, "PCIR is not implemented"
    availability = _all_patterns()

    pair_mask = residual_class.active_pair_mask(availability)

    assert pair_mask.shape == (7, 1, 3)
    assert pair_mask.squeeze(1).tolist() == [
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [True, True, True],
    ]


def test_slot_fusion_supports_seven_patterns_without_missing_value_leakage():
    torch.manual_seed(13)
    encoder = ObservedSetEncoder(
        (2, 3, 4), latent_dim=8, dropout=0.0, fusion_type="slot"
    ).eval()
    availability = _all_patterns()
    umask = torch.ones(1, 7)
    features = torch.randn(7, 1, 9)
    changed = features.clone()
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    changed[expanded == 0] += 10_000.0

    first, _ = encoder(features, availability, umask)
    second, _ = encoder(changed, availability, umask)

    assert first.shape == (7, 1, 8)
    ASSERT_CLOSE(first, second, rtol=0, atol=0)


def test_slot_fusion_keeps_audio_text_and_visual_in_fixed_distinct_slots():
    torch.manual_seed(19)
    encoder = ObservedSetEncoder(
        (1, 1, 1), latent_dim=4, dropout=0.0, fusion_type="slot"
    ).eval()
    availability = torch.tensor(
        [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]], dtype=torch.float32
    )
    captured = []
    handle = encoder.fusion[0].register_forward_pre_hook(
        lambda _module, values: captured.append(values[0].detach().clone())
    )

    encoder(torch.ones(3, 1, 3), availability, torch.ones(1, 3))
    handle.remove()

    fusion_input = captured[0]
    assert fusion_input.shape == (3, 16)
    for row, active_slot in enumerate((0, 1, 2)):
        modality_slots = fusion_input[row, :12].reshape(3, 4)
        assert torch.count_nonzero(modality_slots[active_slot]) > 0
        inactive = [index for index in range(3) if index != active_slot]
        assert torch.count_nonzero(modality_slots[inactive]) == 0


def test_default_mean_fusion_is_exactly_backward_compatible():
    torch.manual_seed(23)
    default = ObservedSetEncoder((2, 3, 4), latent_dim=8, dropout=0.0).eval()
    torch.manual_seed(23)
    explicit = ObservedSetEncoder(
        (2, 3, 4), latent_dim=8, dropout=0.0, fusion_type="mean"
    ).eval()
    availability = _all_patterns()
    features = torch.randn(7, 1, 9)
    umask = torch.ones(1, 7)

    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        ASSERT_CLOSE(value, explicit.state_dict()[key], rtol=0, atol=0)
    first, _ = default(features, availability, umask)
    second, _ = explicit(features, availability, umask)
    ASSERT_CLOSE(first, second, rtol=0, atol=0)


def test_fusion_type_is_validated_and_exposed_by_cli():
    with pytest.raises(ValueError, match="fusion_type"):
        ObservedSetEncoder((2, 3, 4), latent_dim=8, fusion_type="attention")

    args = build_parser().parse_args(
        [
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--output-dir",
            "out",
            "--fusion-type",
            "slot",
        ]
    )
    assert args.fusion_type == "slot"


def test_raw_residual_encoder_starts_as_exact_masked_raw_input():
    torch.manual_seed(29)
    encoder = RawResidualObservedEncoder(
        (2, 3, 4), latent_dim=8, dropout=0.0
    ).eval()
    availability = _all_patterns()
    umask = torch.ones(1, 7)
    features = torch.randn(7, 1, 9)
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )

    output, latents = encoder(features, availability, umask)

    ASSERT_CLOSE(output, features * expanded, rtol=0, atol=0)
    assert set(latents) == {"audio", "text", "visual"}
    assert all(value.shape == (7, 1, 8) for value in latents.values())


def test_raw_residual_encoder_ignores_missing_values_and_zeros_padding():
    torch.manual_seed(31)
    encoder = RawResidualObservedEncoder(
        (2, 3, 4), latent_dim=8, dropout=0.0
    ).eval()
    availability = torch.cat(
        [_all_patterns(), torch.zeros(1, 1, 3)], dim=0
    )
    umask = torch.tensor([[1.0] * 7 + [0.0]])
    features = torch.randn(8, 1, 9)
    changed = features.clone()
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    changed[expanded == 0] += 10_000.0

    first_output, first_latents = encoder(features, availability, umask)
    second_output, second_latents = encoder(changed, availability, umask)

    ASSERT_CLOSE(first_output, second_output, rtol=0, atol=0)
    for name in ("audio", "text", "visual"):
        ASSERT_CLOSE(first_latents[name], second_latents[name], rtol=0, atol=0)
        assert torch.count_nonzero(first_latents[name][-1]) == 0
    assert torch.count_nonzero(first_output[-1]) == 0


def test_contextual_m3_selects_true_missing_targets_and_averages_two_sources(monkeypatch):
    predictor = ContextualM3Predictor(
        latent_dim=4,
        context_dim=5,
        num_experts=2,
        top_k=1,
        dropout=0.0,
    )
    latents = {
        "audio": torch.full((2, 1, 4), 1.0),
        "text": torch.full((2, 1, 4), 3.0),
        "visual": torch.full((2, 1, 4), 5.0),
    }
    hidden = torch.zeros(2, 1, 5)
    availability = torch.tensor(
        [[[1, 1, 0]], [[1, 0, 0]]], dtype=torch.float32
    )
    umask = torch.ones(1, 2)

    def fake_direction(source, context, source_index, target_index):
        value = source + float(source_index * 10 + target_index)
        return value, value + 100.0

    monkeypatch.setattr(predictor, "direction_forward", fake_direction)
    output = predictor(latents, hidden, availability, umask)

    assert output.target_mask[:, 0].tolist() == [
        [False, False, True],
        [False, True, True],
    ]
    assert output.source_counts[:, 0].tolist() == [
        [0, 0, 2],
        [0, 1, 1],
    ]
    # AT -> V averages A->V (1 + 2) and T->V (3 + 12).
    ASSERT_CLOSE(output.reg_predictions[0, 0, 2], torch.full((4,), 9.0))
    # A -> T is a single direction.
    ASSERT_CLOSE(output.reg_predictions[1, 0, 1], torch.full((4,), 2.0))


def test_missing_m3_loss_is_zero_for_complete_atv_and_finite_for_missing_targets():
    predictor = ContextualM3Predictor(4, 5, num_experts=2, top_k=1, dropout=0.0)
    latents = {name: torch.randn(3, 2, 4) for name in ("audio", "text", "visual")}
    hidden = torch.randn(3, 2, 5, requires_grad=True)
    umask = torch.ones(2, 3)
    complete = torch.ones(3, 2, 3)
    teacher = {name: torch.randn(3, 2, 4) for name in latents}

    complete_predictions = predictor(latents, hidden, complete, umask)
    complete_loss = missing_m3_loss(complete_predictions, teacher, temperature=0.1)

    assert complete_loss.target_count == 0
    assert complete_loss.total.item() == 0.0

    missing = complete.clone()
    missing[:, :, 2] = 0
    predictions = predictor(latents, hidden, missing, umask)
    result = missing_m3_loss(predictions, teacher, temperature=0.1)
    result.total.backward()

    assert result.target_count == 6
    assert torch.isfinite(result.total)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def _synthetic_missing_predictions(target_mask, values):
    target_mask = torch.tensor(target_mask, dtype=torch.bool).view(-1, 1, 3)
    reg_predictions = torch.tensor(values, dtype=torch.float32).view(
        target_mask.shape[0], 1, 3, 1
    )
    reg_predictions.requires_grad_()
    return missing_m3_model.MissingM3Predictions(
        reg_predictions=reg_predictions,
        cl_predictions=reg_predictions,
        target_mask=target_mask,
        source_counts=target_mask.long(),
    )


def test_utterance_balanced_regression_differs_only_when_target_counts_differ():
    predictions = _synthetic_missing_predictions(
        [[1, 0, 0], [0, 1, 1]],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    )
    teacher = {
        name: torch.zeros(2, 1, 1)
        for name in ("audio", "text", "visual")
    }

    legacy = missing_m3_loss(
        predictions,
        teacher,
        temperature=0.1,
        regression_aggregation="target",
    )
    balanced = missing_m3_loss(
        predictions,
        teacher,
        temperature=0.1,
        regression_aggregation="utterance",
    )

    ASSERT_CLOSE(legacy.regression, torch.tensor(1.0 / 3.0))
    ASSERT_CLOSE(balanced.regression, torch.tensor(0.25))


def test_utterance_balanced_regression_is_invariant_to_duplicate_target_error():
    one_target = _synthetic_missing_predictions(
        [[1, 0, 0]],
        [1.0, 0.0, 0.0],
    )
    duplicated_target = _synthetic_missing_predictions(
        [[1, 1, 0]],
        [1.0, 1.0, 0.0],
    )
    teacher = {
        name: torch.zeros(1, 1, 1)
        for name in ("audio", "text", "visual")
    }

    first = missing_m3_loss(
        one_target,
        teacher,
        regression_aggregation="utterance",
    )
    second = missing_m3_loss(
        duplicated_target,
        teacher,
        regression_aggregation="utterance",
    )

    ASSERT_CLOSE(first.regression, second.regression, rtol=0, atol=0)


def test_target_aggregation_default_remains_exactly_legacy():
    predictions = _synthetic_missing_predictions(
        [[1, 0, 0], [0, 1, 1]],
        [0.2, 0.0, 0.0, 0.0, 0.5, -0.7],
    )
    teacher = {
        name: torch.zeros(2, 1, 1)
        for name in ("audio", "text", "visual")
    }

    implicit = missing_m3_loss(predictions, teacher)
    explicit = missing_m3_loss(
        predictions,
        teacher,
        regression_aggregation="target",
    )

    ASSERT_CLOSE(implicit.total, explicit.total, rtol=0, atol=0)
    ASSERT_CLOSE(implicit.regression, explicit.regression, rtol=0, atol=0)
    ASSERT_CLOSE(implicit.contrastive, explicit.contrastive, rtol=0, atol=0)


def test_utterance_balanced_zero_target_loss_has_finite_backward():
    predictions = _synthetic_missing_predictions(
        [[0, 0, 0]],
        [0.0, 0.0, 0.0],
    )
    teacher = {
        name: torch.zeros(1, 1, 1)
        for name in ("audio", "text", "visual")
    }

    result = missing_m3_loss(
        predictions,
        teacher,
        regression_aggregation="utterance",
    )
    result.total.backward()

    assert result.target_count == 0
    assert result.total.item() == 0.0
    assert torch.isfinite(predictions.reg_predictions.grad).all()


def _model_arguments():
    return dict(
        base_model="LSTM",
        adim=2,
        tdim=3,
        vdim=4,
        D_e=4,
        graph_hidden_size=2,
        n_speakers=2,
        window_past=1,
        window_future=1,
        n_classes=6,
        dropout=0.0,
        time_attn=False,
        no_cuda=True,
        latent_dim=8,
        num_experts=2,
        top_k=1,
        predictor_dropout=0.0,
    )


def _model_inputs():
    features = torch.randn(3, 2, 9)
    availability = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 1]],
            [[1, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    qmask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    return features, availability, qmask, umask, [3, 2]


def _single_conversation_inputs(valid_features, total_length):
    valid_length = valid_features.shape[0]
    padding = valid_features.new_zeros(
        total_length - valid_length, 1, valid_features.shape[-1]
    )
    features = torch.cat([valid_features, padding], dim=0)
    availability = features.new_zeros(total_length, 1, 3)
    availability[:valid_length] = 1
    qmask = features.new_zeros(1, total_length)
    umask = features.new_zeros(1, total_length)
    umask[:, :valid_length] = 1
    return features, availability, qmask, umask, [valid_length]


def test_packed_recurrent_preserves_parameters_rng_and_equal_length_outputs():
    torch.manual_seed(1559)
    default = MissingM3GraphModel(**_model_arguments()).eval()
    default_rng = torch.get_rng_state()
    torch.manual_seed(1559)
    legacy = MissingM3GraphModel(
        **_model_arguments(), recurrent_padding_mode="legacy"
    ).eval()
    legacy_rng = torch.get_rng_state()
    torch.manual_seed(1559)
    packed = MissingM3GraphModel(
        **_model_arguments(), recurrent_padding_mode="packed"
    ).eval()
    packed_rng = torch.get_rng_state()

    assert default.state_dict().keys() == legacy.state_dict().keys()
    assert legacy.state_dict().keys() == packed.state_dict().keys()
    for name, value in default.state_dict().items():
        ASSERT_CLOSE(value, legacy.state_dict()[name], rtol=0, atol=0)
        ASSERT_CLOSE(value, packed.state_dict()[name], rtol=0, atol=0)
    ASSERT_CLOSE(default_rng, legacy_rng, rtol=0, atol=0)
    ASSERT_CLOSE(legacy_rng, packed_rng, rtol=0, atol=0)
    legacy.load_state_dict(default.state_dict(), strict=True)

    valid_features = torch.randn(3, 1, 9)
    inputs = _single_conversation_inputs(valid_features, total_length=3)
    legacy_outputs = legacy([inputs[0]], *inputs[1:])
    packed_outputs = packed([inputs[0]], *inputs[1:])
    ASSERT_CLOSE(legacy_outputs[0], packed_outputs[0], rtol=1e-6, atol=1e-6)
    ASSERT_CLOSE(legacy_outputs[1], packed_outputs[1], rtol=1e-6, atol=1e-6)


def test_packed_mode_reaches_pregraph_and_both_postgraph_recurrents(monkeypatch):
    calls = []
    original = base_graph_model._run_recurrent

    def recording_run(recurrent, *args, **kwargs):
        calls.append(recurrent)
        return original(recurrent, *args, **kwargs)

    monkeypatch.setattr(base_graph_model, "_run_recurrent", recording_run)
    model = MissingM3GraphModel(
        **_model_arguments(), recurrent_padding_mode="packed"
    ).eval()
    valid_features = torch.randn(3, 1, 9)
    inputs = _single_conversation_inputs(valid_features, total_length=3)

    model([inputs[0]], *inputs[1:])

    assert calls == [
        model.lstm,
        model.graph_net_temporal.grufusion,
        model.graph_net_speaker.grufusion,
    ]


def test_packed_mode_rejects_nonprefix_or_inconsistent_validity_mask():
    model = MissingM3GraphModel(
        **_model_arguments(), recurrent_padding_mode="packed"
    ).eval()
    valid_features = torch.randn(3, 1, 9)
    inputs = list(_single_conversation_inputs(valid_features, total_length=3))
    inputs[3] = torch.tensor([[1.0, 0.0, 1.0]])
    inputs[1][1] = 0

    with pytest.raises(ValueError, match="same contiguous prefix"):
        model([inputs[0]], *inputs[1:])


def test_packed_recurrent_valid_outputs_ignore_suffix_padding_and_backward_is_finite():
    torch.manual_seed(1561)
    model = MissingM3GraphModel(
        **_model_arguments(), recurrent_padding_mode="packed"
    ).eval()
    valid_features = torch.randn(2, 1, 9)
    short = _single_conversation_inputs(valid_features, total_length=2)
    padded_short = _single_conversation_inputs(valid_features, total_length=7)
    companion_features = torch.randn(7, 1, 9)
    companion = _single_conversation_inputs(
        companion_features, total_length=7
    )
    mixed = (
        torch.cat([padded_short[0], companion[0]], dim=1),
        torch.cat([padded_short[1], companion[1]], dim=1),
        torch.cat([padded_short[2], companion[2]], dim=0),
        torch.cat([padded_short[3], companion[3]], dim=0),
        [2, 7],
    )

    short_outputs = model([short[0]], *short[1:])
    mixed_outputs = model([mixed[0]], *mixed[1:])

    ASSERT_CLOSE(
        short_outputs[0][:, 0], mixed_outputs[0][:2, 0], rtol=1e-6, atol=1e-6
    )
    ASSERT_CLOSE(
        short_outputs[1][:, 0], mixed_outputs[1][:2, 0], rtol=1e-6, atol=1e-6
    )
    ASSERT_CLOSE(
        mixed_outputs[1][2:, 0],
        torch.zeros_like(mixed_outputs[1][2:, 0]),
        rtol=0,
        atol=0,
    )

    model.train()
    logits, hidden, _, _ = model([mixed[0]], *mixed[1:])
    (logits[:2, 0].sum() + hidden[:2, 0].square().mean()).backward()
    recurrent_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "lstm" in name or "grufusion" in name
    ]
    assert any(
        gradient is not None and bool(torch.count_nonzero(gradient))
        for gradient in recurrent_gradients
    )
    assert all(
        gradient is None or bool(torch.isfinite(gradient).all())
        for gradient in recurrent_gradients
    )


def test_packed_recurrent_is_an_explicit_cli_switch_with_legacy_default():
    parser = build_parser()
    required = [
        "--audio-feature",
        "a",
        "--text-feature",
        "t",
        "--video-feature",
        "v",
        "--output-dir",
        "out",
    ]

    assert parser.parse_args(required).recurrent_padding_mode == "legacy"
    assert (
        parser.parse_args(
            [*required, "--recurrent-padding-mode", "packed"]
        ).recurrent_padding_mode
        == "packed"
    )
    assert TrainConfig().recurrent_padding_mode == "legacy"


def test_postgraph_sequence_default_and_explicit_independent_are_exact():
    torch.manual_seed(1601)
    default = MissingM3GraphModel(**_model_arguments()).eval()
    default_rng = torch.get_rng_state()
    torch.manual_seed(1601)
    explicit = MissingM3GraphModel(
        **_model_arguments(), postgraph_sequence_mode="independent"
    ).eval()
    explicit_rng = torch.get_rng_state()

    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name, value in default.state_dict().items():
        ASSERT_CLOSE(value, explicit.state_dict()[name], rtol=0, atol=0)
    ASSERT_CLOSE(default_rng, explicit_rng, rtol=0, atol=0)

    inputs = _model_inputs()
    default_outputs = default([inputs[0]], *inputs[1:])
    explicit_outputs = explicit([inputs[0]], *inputs[1:])
    for default_value, explicit_value in zip(
        default_outputs[:3], explicit_outputs[:3]
    ):
        if isinstance(default_value, dict):
            for key in default_value:
                ASSERT_CLOSE(
                    default_value[key], explicit_value[key], rtol=0, atol=0
                )
        else:
            ASSERT_CLOSE(default_value, explicit_value, rtol=0, atol=0)


def test_independent_postgraph_preserves_legacy_speaker_call_signature(monkeypatch):
    model = MissingM3GraphModel(**_model_arguments()).eval()
    original_forward = model.graph_net_speaker.forward
    calls = []

    def legacy_forward(features, edge_index, edge_type, seq_lengths, umask):
        calls.append(1)
        return original_forward(
            features, edge_index, edge_type, seq_lengths, umask
        )

    monkeypatch.setattr(model.graph_net_speaker, "forward", legacy_forward)
    inputs = _model_inputs()
    model([inputs[0]], *inputs[1:])

    assert calls == [1]


def test_shared_postgraph_bilstm_reuses_temporal_recurrent_for_both_branches(
    monkeypatch,
):
    torch.manual_seed(1603)
    control = MissingM3GraphModel(**_model_arguments()).eval()
    control_state = control.state_dict()
    control_parameter_count = sum(
        parameter.numel() for parameter in control.parameters()
    )
    control_trainable_count = sum(
        parameter.numel()
        for parameter in control.parameters()
        if parameter.requires_grad
    )

    torch.manual_seed(1603)
    model = MissingM3GraphModel(
        **_model_arguments(), postgraph_sequence_mode="shared-bilstm"
    ).train()
    model_rng = torch.get_rng_state()
    assert model.state_dict().keys() == control_state.keys()
    model.load_state_dict(control_state, strict=True)
    assert model.graph_net_temporal.conv1 is not model.graph_net_speaker.conv1
    assert model.graph_net_temporal.conv2 is not model.graph_net_speaker.conv2
    assert model.graph_net_temporal.linear is not model.graph_net_speaker.linear
    assert model.graph_net_temporal.grufusion is not model.graph_net_speaker.grufusion
    assert all(
        not parameter.requires_grad
        for parameter in model.graph_net_speaker.grufusion.parameters()
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        control_parameter_count
    )
    frozen_count = sum(
        parameter.numel()
        for parameter in model.graph_net_speaker.grufusion.parameters()
    )
    assert sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) == control_trainable_count - frozen_count
    ASSERT_CLOSE(model_rng, torch.get_rng_state(), rtol=0, atol=0)

    recurrent_outputs = []
    original_forward = model.graph_net_temporal.grufusion.forward

    def recording_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        result[0].retain_grad()
        recurrent_outputs.append(result[0])
        return result

    monkeypatch.setattr(
        model.graph_net_temporal.grufusion,
        "forward",
        recording_forward,
    )
    inputs = _model_inputs()
    logits, hidden, _, _ = model([inputs[0]], *inputs[1:])
    (logits.square().mean() + hidden.square().mean()).backward()

    assert len(recurrent_outputs) == 2
    assert all(
        output.grad is not None
        and torch.isfinite(output.grad).all()
        and torch.count_nonzero(output.grad) > 0
        for output in recurrent_outputs
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.graph_net_temporal.grufusion.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.graph_net_speaker.grufusion.parameters()
    )


def test_shared_postgraph_bilstm_ignores_frozen_copy_but_keeps_speaker_linear():
    torch.manual_seed(1605)
    model = MissingM3GraphModel(
        **_model_arguments(), postgraph_sequence_mode="shared-bilstm"
    ).eval()
    inputs = _model_inputs()
    before = model([inputs[0]], *inputs[1:])[1]

    with torch.no_grad():
        for parameter in model.graph_net_speaker.grufusion.parameters():
            parameter.add_(100.0)
    after_frozen_change = model([inputs[0]], *inputs[1:])[1]
    ASSERT_CLOSE(before, after_frozen_change, rtol=0, atol=0)

    with torch.no_grad():
        model.graph_net_speaker.linear.bias.add_(10.0)
    after_linear_change = model([inputs[0]], *inputs[1:])[1]
    assert not torch.equal(before, after_linear_change)


def test_shared_postgraph_bilstm_preserves_forward_rng_budget_and_formal_count():
    arguments = {**_model_arguments(), "dropout": 0.2}
    torch.manual_seed(1607)
    control = MissingM3GraphModel(**arguments).train()
    torch.manual_seed(1607)
    candidate = MissingM3GraphModel(
        **arguments, postgraph_sequence_mode="shared-bilstm"
    ).train()
    candidate.load_state_dict(control.state_dict(), strict=True)
    inputs = _model_inputs()

    torch.manual_seed(1609)
    control([inputs[0]], *inputs[1:])
    control_rng = torch.get_rng_state()
    torch.manual_seed(1609)
    candidate([inputs[0]], *inputs[1:])
    candidate_rng = torch.get_rng_state()

    ASSERT_CLOSE(control_rng, candidate_rng, rtol=0, atol=0)

    formal = MissingM3GraphModel(
        base_model="LSTM",
        adim=512,
        tdim=1024,
        vdim=1024,
        D_e=100,
        graph_hidden_size=50,
        n_speakers=1,
        window_past=1,
        window_future=1,
        n_classes=1,
        dropout=0.5,
        time_attn=False,
        no_cuda=True,
        latent_dim=256,
        fusion_type="slot",
        postgraph_sequence_mode="shared-bilstm",
    )
    assert sum(
        parameter.numel()
        for parameter in formal.graph_net_speaker.grufusion.parameters()
    ) == 2_508_000


def test_shared_postgraph_bilstm_requires_both_graph_branches():
    with pytest.raises(ValueError, match="requires graph_branch_mode='both'"):
        MissingM3GraphModel(
            **_model_arguments(),
            graph_branch_mode="temporal-only",
            postgraph_sequence_mode="shared-bilstm",
        )


def test_graph_message_calibration_default_is_exactly_none():
    torch.manual_seed(131)
    default = MissingM3GraphModel(**_model_arguments()).eval()
    torch.manual_seed(131)
    explicit = MissingM3GraphModel(
        **_model_arguments(), graph_message_calibration="none"
    ).eval()

    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        ASSERT_CLOSE(value, explicit.state_dict()[key], rtol=0, atol=0)
    inputs = _model_inputs()
    ASSERT_CLOSE(
        default(*inputs)[:3],
        explicit(*inputs)[:3],
        rtol=0,
        atol=0,
    )


def test_branch_graph_message_calibration_matches_the_bounded_formula():
    network = base_graph_model.GraphNetwork(
        num_features=4,
        num_relations=3,
        time_attn=False,
        hidden_size=3,
        dropout=0.0,
        no_cuda=True,
        graph_message_calibration="branch-layernorm-residual",
    )
    message = torch.tensor(
        [[1.0, 2.0, 4.0], [-2.0, 1.0, 3.0]], requires_grad=True
    )

    initial = network._calibrate_graph_message(message)
    ASSERT_CLOSE(initial, message, rtol=0, atol=0)

    with torch.no_grad():
        network.message_calibration_alpha.copy_(
            torch.tensor([0.2, -0.3, 0.4])
        )
    normalized = torch.nn.functional.layer_norm(message, (3,))
    expected = message + torch.tanh(network.message_calibration_alpha) * (
        normalized - message
    )
    actual = network._calibrate_graph_message(message)
    ASSERT_CLOSE(actual, expected)

    probe = torch.tensor([[1.0, -0.5, 0.25], [-0.7, 0.3, 1.1]])
    (actual * probe).sum().backward()
    gradient = network.message_calibration_alpha.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == gradient.numel()


def test_graph_message_calibration_is_branch_specific_and_adds_only_2dg():
    torch.manual_seed(137)
    control = MissingM3GraphModel(**_model_arguments())
    torch.manual_seed(137)
    treatment = MissingM3GraphModel(
        **_model_arguments(),
        graph_message_calibration="branch-layernorm-residual",
    )

    extra_keys = set(treatment.state_dict()) - set(control.state_dict())
    assert extra_keys == {
        "graph_net_temporal.message_calibration_alpha",
        "graph_net_speaker.message_calibration_alpha",
    }
    for key, value in control.state_dict().items():
        ASSERT_CLOSE(value, treatment.state_dict()[key], rtol=0, atol=0)
    temporal = treatment.graph_net_temporal.message_calibration_alpha
    speaker = treatment.graph_net_speaker.message_calibration_alpha
    assert temporal is not speaker
    assert temporal.data_ptr() != speaker.data_ptr()
    parameter_delta = sum(p.numel() for p in treatment.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    assert parameter_delta == 2 * treatment.graph_net_temporal.hidden_size


def test_postgraph_sequence_mode_is_an_explicit_cli_switch():
    parser = build_parser()
    required = [
        "--audio-feature",
        "a",
        "--text-feature",
        "t",
        "--video-feature",
        "v",
        "--output-dir",
        "out",
    ]

    assert parser.parse_args(required).postgraph_sequence_mode == "independent"
    assert (
        parser.parse_args(
            [*required, "--postgraph-sequence-mode", "shared-bilstm"]
        ).postgraph_sequence_mode
        == "shared-bilstm"
    )
    assert TrainConfig().postgraph_sequence_mode == "independent"


def test_graph_message_calibration_is_an_explicit_cli_switch():
    required = [
        "--audio-feature",
        "a",
        "--text-feature",
        "t",
        "--video-feature",
        "v",
        "--output-dir",
        "out",
    ]
    parser = build_parser()
    defaults = parser.parse_args(required)
    candidate = parser.parse_args(
        [
            *required,
            "--graph-message-calibration",
            "branch-layernorm-residual",
        ]
    )

    assert defaults.graph_message_calibration == "none"
    assert TrainConfig().graph_message_calibration == "none"
    assert candidate.graph_message_calibration == "branch-layernorm-residual"


def test_track_representation_reuses_one_graph_core_for_three_modalities(
    monkeypatch,
):
    model = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        representation_type="track",
    ).eval()
    calls = []

    def fake_encode_hidden(inputfeats, qmask, umask, seq_lengths):
        calls.append(inputfeats[0].detach().clone())
        padding = inputfeats[0].new_zeros(*inputfeats[0].shape[:2], 2)
        return torch.cat([inputfeats[0], padding], dim=-1)

    monkeypatch.setattr(model, "encode_hidden", fake_encode_hidden)
    features, availability, qmask, umask, lengths = _model_inputs()

    logits, hidden, latents, _ = model(
        [features], availability, qmask, umask, lengths
    )

    assert len(calls) == 3
    assert logits.shape == (3, 2, 6)
    assert hidden.shape == (3, 2, 10)
    assert set(latents) == {"audio", "text", "visual"}
    graph_prefixes = {
        name.split(".", 1)[0]
        for name, _ in model.named_parameters()
        if name.startswith("graph_net_")
    }
    assert graph_prefixes == {"graph_net_temporal", "graph_net_speaker"}


def test_default_representation_preserves_state_keys_and_single_graph_call(
    monkeypatch,
):
    torch.manual_seed(107)
    default = MissingM3GraphModel(
        **_model_arguments(), fusion_type="slot"
    ).eval()
    torch.manual_seed(107)
    explicit = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        representation_type="slot",
    ).eval()
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        ASSERT_CLOSE(value, explicit.state_dict()[key], rtol=0, atol=0)

    calls = []
    original = explicit.encode_hidden

    def counted_encode_hidden(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(explicit, "encode_hidden", counted_encode_hidden)
    explicit([_model_inputs()[0]], *_model_inputs()[1:])
    assert len(calls) == 1


def test_track_representation_pairs_shared_initialization_and_fusion_dropout():
    torch.manual_seed(113)
    control = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        projector_dropout=0.17,
        representation_type="slot",
    )
    torch.manual_seed(113)
    treatment = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        projector_dropout=0.17,
        representation_type="track",
    )

    control_state = control.state_dict()
    treatment_state = treatment.state_dict()
    shared_keys = set(control_state).intersection(treatment_state)
    assert shared_keys
    for key in shared_keys:
        ASSERT_CLOSE(control_state[key], treatment_state[key], rtol=0, atol=0)
    assert treatment.track_fusion.fusion[-1].p == pytest.approx(0.17)


def test_track_representation_rejects_conflicting_fusion_paths():
    with pytest.raises(ValueError, match="track.*fusion_type='slot'"):
        MissingM3GraphModel(
            **_model_arguments(),
            fusion_type="mean",
            representation_type="track",
        )
    with pytest.raises(ValueError, match="track.*local_context_residual"):
        MissingM3GraphModel(
            **_model_arguments(),
            fusion_type="slot",
            representation_type="track",
            local_context_residual=True,
        )
    with pytest.raises(ValueError, match="track.*classification_completion"):
        MissingM3GraphModel(
            **_model_arguments(),
            fusion_type="slot",
            representation_type="track",
            classification_completion=True,
        )


def test_representation_type_is_exposed_by_cli_and_config():
    args = build_parser().parse_args(
        [
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--output-dir",
            "out",
            "--fusion-type",
            "slot",
            "--representation-type",
            "track",
        ]
    )
    config = TrainConfig(
        fusion_type=args.fusion_type,
        representation_type=args.representation_type,
    )

    assert args.representation_type == "track"
    assert asdict(config)["representation_type"] == "track"


def test_node_interaction_residual_is_exposed_by_cli_and_config():
    args = build_parser().parse_args(
        [
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--output-dir",
            "out",
            "--fusion-type",
            "slot",
            "--node-interaction-residual",
        ]
    )
    config = TrainConfig(
        fusion_type=args.fusion_type,
        node_interaction_residual=args.node_interaction_residual,
    )

    assert args.node_interaction_residual is True
    assert asdict(config)["node_interaction_residual"] is True


def test_track_representation_backward_reaches_projectors_graph_fusion_and_predictor():
    torch.manual_seed(109)
    model = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        representation_type="track",
    ).train()
    features, availability, qmask, umask, lengths = _model_inputs()

    logits, _, _, predictions = model(
        [features],
        availability,
        qmask,
        umask,
        lengths,
        predict_missing=True,
    )
    assert predictions is not None
    (logits.square().mean() + predictions.reg_predictions.square().mean()).backward()

    groups = {
        "projectors": model.observed_set.projectors.parameters(),
        "modality_embedding": model.observed_set.modality_embedding.parameters(),
        "temporal_graph": model.graph_net_temporal.parameters(),
        "speaker_graph": model.graph_net_speaker.parameters(),
        "track_fusion": model.track_fusion.parameters(),
        "predictor": model.missing_predictor.parameters(),
    }
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters]
        assert any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient) > 0
            for gradient in gradients
        ), f"no finite non-zero gradient reached {name}"


def test_track_representation_logits_ignore_raw_values_in_missing_blocks():
    torch.manual_seed(127)
    model = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        representation_type="track",
    ).eval()
    features, availability, qmask, umask, lengths = _model_inputs()
    changed = features.clone()
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    changed[expanded == 0] += 10_000.0

    first = model([features], availability, qmask, umask, lengths)
    second = model([changed], availability, qmask, umask, lengths)

    ASSERT_CLOSE(first[0], second[0], rtol=0, atol=0)
    ASSERT_CLOSE(first[1], second[1], rtol=0, atol=0)
    for name in first[2]:
        ASSERT_CLOSE(first[2][name], second[2][name], rtol=0, atol=0)


def test_node_interaction_residual_preserves_shared_initialization_and_initial_output():
    torch.manual_seed(137)
    control = MissingM3GraphModel(
        **_model_arguments(), fusion_type="slot"
    ).eval()
    torch.manual_seed(137)
    treatment = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        node_interaction_residual=True,
    ).eval()

    control_state = control.state_dict()
    treatment_state = treatment.state_dict()
    assert set(control_state).issubset(treatment_state)
    for key, value in control_state.items():
        ASSERT_CLOSE(value, treatment_state[key], rtol=0, atol=0)
    assert any(key.startswith("node_interaction.") for key in treatment_state)

    inputs = _model_inputs()
    control_outputs = control([inputs[0]], *inputs[1:], predict_missing=True)
    treatment_outputs = treatment(
        [inputs[0]], *inputs[1:], predict_missing=True
    )
    for control_value, treatment_value in zip(
        control_outputs[:3], treatment_outputs[:3]
    ):
        if isinstance(control_value, dict):
            for name in control_value:
                ASSERT_CLOSE(
                    control_value[name], treatment_value[name], rtol=0, atol=0
                )
        else:
            ASSERT_CLOSE(control_value, treatment_value, rtol=0, atol=0)
    for field in (
        "reg_predictions",
        "cl_predictions",
        "target_mask",
        "source_counts",
    ):
        ASSERT_CLOSE(
            getattr(control_outputs[3], field),
            getattr(treatment_outputs[3], field),
            rtol=0,
            atol=0,
        )


def test_node_interaction_residual_backward_reaches_new_and_shared_modules():
    torch.manual_seed(139)
    model = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        node_interaction_residual=True,
    ).train()
    with torch.no_grad():
        torch.nn.init.normal_(model.node_interaction.residual_mlp[-1].weight)
    features, availability, qmask, umask, lengths = _model_inputs()

    logits, _, _, predictions = model(
        [features],
        availability,
        qmask,
        umask,
        lengths,
        predict_missing=True,
    )
    assert predictions is not None
    (logits.square().mean() + predictions.reg_predictions.square().mean()).backward()

    groups = {
        "scale_shift": model.node_interaction.scale_shift.parameters(),
        "pair_mlp": model.node_interaction.pair_mlp.parameters(),
        "residual_mlp": model.node_interaction.residual_mlp.parameters(),
        "projectors": model.observed_set.projectors.parameters(),
        "temporal_graph": model.graph_net_temporal.parameters(),
        "speaker_graph": model.graph_net_speaker.parameters(),
    }
    for name, parameters in groups.items():
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad) > 0
            for parameter in parameters
        ), f"no finite non-zero gradient reached {name}"


def test_node_interaction_residual_rejects_conflicting_paths():
    with pytest.raises(ValueError, match="node_interaction_residual.*slot"):
        MissingM3GraphModel(
            **_model_arguments(),
            fusion_type="mean",
            node_interaction_residual=True,
        )
    with pytest.raises(ValueError, match="node_interaction_residual.*track"):
        MissingM3GraphModel(
            **_model_arguments(),
            fusion_type="slot",
            representation_type="track",
            node_interaction_residual=True,
        )


@pytest.mark.parametrize(
    "mode,active_name,inactive_name",
    [
        ("temporal-only", "graph_net_temporal", "graph_net_speaker"),
        ("speaker-only", "graph_net_speaker", "graph_net_temporal"),
    ],
)
def test_graph_branch_mode_missing_m3_routes_gradients_only_to_active_branch(
    mode, active_name, inactive_name
):
    torch.manual_seed(23)
    model = MissingM3GraphModel(
        **_model_arguments(), graph_branch_mode=mode
    ).train()
    features, availability, qmask, umask, lengths = _model_inputs()

    logits, _, _, _ = model(
        [features], availability, qmask, umask, lengths, predict_missing=False
    )
    logits.square().mean().backward()

    active_gradients = [
        parameter.grad for parameter in getattr(model, active_name).parameters()
    ]
    inactive_gradients = [
        parameter.grad for parameter in getattr(model, inactive_name).parameters()
    ]
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0
        for gradient in active_gradients
    )
    assert all(gradient is None for gradient in inactive_gradients)


def test_graph_branch_mode_missing_m3_defaults_to_both():
    torch.manual_seed(29)
    default = MissingM3GraphModel(**_model_arguments()).eval()
    torch.manual_seed(29)
    explicit = MissingM3GraphModel(
        **_model_arguments(), graph_branch_mode="both"
    ).eval()
    explicit.load_state_dict(default.state_dict(), strict=True)
    inputs = _model_inputs()

    default_output = default(
        [inputs[0]], *inputs[1:4], inputs[4], predict_missing=False
    )[0]
    explicit_output = explicit(
        [inputs[0]], *inputs[1:4], inputs[4], predict_missing=False
    )[0]

    ASSERT_CLOSE(default_output, explicit_output, rtol=0, atol=0)


def test_local_context_residual_has_expected_shape_zero_padding_and_no_missing_leakage():
    torch.manual_seed(41)
    fusion = LocalContextResidualFusion(
        latent_dim=4, context_dim=6, hidden_dim=7, dropout=0.0
    ).eval()
    with torch.no_grad():
        torch.nn.init.normal_(fusion.fusion[-1].weight)
        torch.nn.init.normal_(fusion.fusion[-1].bias)
    availability = torch.tensor(
        [
            [[1, 0, 1], [0, 1, 0]],
            [[1, 1, 0], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    umask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    latents = {
        name: torch.randn(2, 2, 4)
        for name in ("audio", "text", "visual")
    }
    changed = {name: value.clone() for name, value in latents.items()}
    for index, name in enumerate(("audio", "text", "visual")):
        changed[name][~availability[..., index].bool()] += 10_000.0

    first = fusion(latents, availability, umask)
    second = fusion(changed, availability, umask)

    assert first.shape == (2, 2, 6)
    ASSERT_CLOSE(first, second, rtol=0, atol=0)
    assert torch.count_nonzero(first[~umask.T.bool()]) == 0


def test_local_context_residual_preserves_same_seed_shared_state_and_zero_init_outputs():
    torch.manual_seed(43)
    base = MissingM3GraphModel(
        **_model_arguments(), fusion_type="slot"
    ).eval()
    torch.manual_seed(43)
    local = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        local_context_residual=True,
        local_fusion_hidden_dim=12,
        local_fusion_dropout=0.0,
    ).eval()

    base_state = base.state_dict()
    local_state = local.state_dict()
    assert set(base_state).issubset(local_state)
    for key, value in base_state.items():
        ASSERT_CLOSE(value, local_state[key], rtol=0, atol=0)

    torch.manual_seed(47)
    inputs = _model_inputs()
    base_outputs = base([inputs[0]], *inputs[1:], predict_missing=True)
    local_outputs = local([inputs[0]], *inputs[1:], predict_missing=True)

    for base_value, local_value in zip(base_outputs[:3], local_outputs[:3]):
        if isinstance(base_value, dict):
            for name in base_value:
                ASSERT_CLOSE(base_value[name], local_value[name], rtol=0, atol=0)
        else:
            ASSERT_CLOSE(base_value, local_value, rtol=0, atol=0)
    assert base_outputs[3] is not None and local_outputs[3] is not None
    for field in ("reg_predictions", "cl_predictions", "target_mask", "source_counts"):
        ASSERT_CLOSE(
            getattr(base_outputs[3], field),
            getattr(local_outputs[3], field),
            rtol=0,
            atol=0,
        )


def test_local_context_residual_preserves_train_mode_rng_and_zero_init_outputs():
    arguments = _model_arguments()
    arguments["predictor_dropout"] = 0.4
    torch.manual_seed(67)
    base = MissingM3GraphModel(
        **arguments, fusion_type="slot"
    ).train()
    torch.manual_seed(67)
    local = MissingM3GraphModel(
        **arguments,
        fusion_type="slot",
        local_context_residual=True,
        local_fusion_hidden_dim=12,
        local_fusion_dropout=0.5,
    ).train()
    torch.manual_seed(71)
    inputs = _model_inputs()
    forward_rng = torch.random.get_rng_state()

    torch.random.set_rng_state(forward_rng)
    base_outputs = base([inputs[0]], *inputs[1:], predict_missing=True)
    torch.random.set_rng_state(forward_rng)
    local_outputs = local([inputs[0]], *inputs[1:], predict_missing=True)

    ASSERT_CLOSE(base_outputs[0], local_outputs[0], rtol=0, atol=0)
    ASSERT_CLOSE(base_outputs[1], local_outputs[1], rtol=0, atol=0)
    assert base_outputs[3] is not None and local_outputs[3] is not None
    for field in ("reg_predictions", "cl_predictions", "target_mask", "source_counts"):
        ASSERT_CLOSE(
            getattr(base_outputs[3], field),
            getattr(local_outputs[3], field),
            rtol=0,
            atol=0,
        )


def test_local_context_residual_keeps_missing_predictor_on_graph_hidden(monkeypatch):
    torch.manual_seed(53)
    model = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        local_context_residual=True,
        local_fusion_hidden_dim=12,
        local_fusion_dropout=0.0,
    ).eval()
    with torch.no_grad():
        model.local_context_fusion.fusion[-1].bias.fill_(1.0)
    captured = {}

    def capture_predictor(latents, hidden, availability, umask):
        captured["hidden"] = hidden.detach().clone()
        return None

    monkeypatch.setattr(model.missing_predictor, "forward", capture_predictor)
    torch.manual_seed(59)
    inputs = _model_inputs()
    _, classification_hidden, _, _ = model(
        [inputs[0]], *inputs[1:], predict_missing=True
    )

    valid = inputs[3].T.bool()
    ASSERT_CLOSE(
        captured["hidden"][valid] + 1.0,
        classification_hidden.detach()[valid],
        rtol=0,
        atol=0,
    )


def test_local_context_residual_disabled_has_no_parameters_and_rejects_non_slot():
    disabled = MissingM3GraphModel(
        **_model_arguments(), fusion_type="slot", local_context_residual=False
    )

    assert not any(
        name.startswith("local_context_fusion.")
        for name, _ in disabled.named_parameters()
    )
    assert not hasattr(disabled, "local_context_fusion")
    for fusion_type in ("mean", "raw-residual"):
        with pytest.raises(ValueError, match="local_context_residual.*slot"):
            MissingM3GraphModel(
                **_model_arguments(),
                fusion_type=fusion_type,
                local_context_residual=True,
            )


def test_local_context_residual_cli_and_config_route_all_options():
    args = build_parser().parse_args(
        [
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--output-dir",
            "out",
            "--fusion-type",
            "slot",
            "--local-context-residual",
            "--local-fusion-hidden-dim",
            "96",
            "--local-fusion-dropout",
            "0.15",
        ]
    )
    config = TrainConfig(
        fusion_type=args.fusion_type,
        local_context_residual=args.local_context_residual,
        local_fusion_hidden_dim=args.local_fusion_hidden_dim,
        local_fusion_dropout=args.local_fusion_dropout,
    )

    assert {
        key: asdict(config)[key]
        for key in (
            "local_context_residual",
            "local_fusion_hidden_dim",
            "local_fusion_dropout",
        )
    } == {
        "local_context_residual": True,
        "local_fusion_hidden_dim": 96,
        "local_fusion_dropout": 0.15,
    }


def test_local_context_residual_persists_in_config_json_and_checkpoint(tmp_path):
    config = TrainConfig(
        fusion_type="slot",
        local_context_residual=True,
        local_fusion_hidden_dim=96,
        local_fusion_dropout=0.15,
    )
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "best.pt"

    train_gcnet._write_run_config(config_path, config)
    train_gcnet._save_best_checkpoint(
        checkpoint_path,
        model_state={"weight": torch.tensor([1.0])},
        config_value=config,
        epoch=3,
        validation_mean_weighted_f1=0.75,
    )

    json_config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_config = torch.load(checkpoint_path, map_location="cpu")["config"]
    expected = {
        "local_context_residual": True,
        "local_fusion_hidden_dim": 96,
        "local_fusion_dropout": 0.15,
    }
    assert {key: json_config[key] for key in expected} == expected
    assert {key: checkpoint_config[key] for key in expected} == expected


def test_local_context_residual_backward_reaches_all_training_paths():
    torch.manual_seed(61)
    model = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        local_context_residual=True,
        local_fusion_hidden_dim=12,
        local_fusion_dropout=0.0,
    ).train()
    inputs = _model_inputs()

    logits, _, _, predictions = model(
        [inputs[0]], *inputs[1:], predict_missing=True
    )
    assert predictions is not None and bool(predictions.target_mask.any())
    predicted = predictions.reg_predictions[predictions.target_mask]
    (logits.square().mean() + predicted.square().mean()).backward()

    parameter_groups = {
        "student": model.observed_set.projectors.parameters(),
        "local": model.local_context_fusion.parameters(),
        "graph": model.graph_net_temporal.parameters(),
        "predictor": model.missing_predictor.parameters(),
    }
    for name, parameters in parameter_groups.items():
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None
        ]
        assert gradients, name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients), name


def test_raw_residual_model_keeps_original_recurrent_width_and_cli():
    raw = MissingM3GraphModel(
        **_model_arguments(), fusion_type="raw-residual"
    )
    slot = MissingM3GraphModel(**_model_arguments(), fusion_type="slot")

    assert raw.lstm.input_size == 9
    assert slot.lstm.input_size == 8

    args = build_parser().parse_args(
        [
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--output-dir",
            "out",
            "--fusion-type",
            "raw-residual",
        ]
    )
    assert args.fusion_type == "raw-residual"


def test_raw_residual_backward_reaches_online_encoder_graph_and_predictor():
    torch.manual_seed(37)
    model = MissingM3GraphModel(
        **_model_arguments(), fusion_type="raw-residual"
    ).train()
    features = torch.randn(3, 2, 9)
    availability = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 1]],
            [[1, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    qmask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    logits, _, _, predictions = model(
        [features], availability, qmask, umask, [3, 2], predict_missing=True
    )
    assert predictions is not None and bool(predictions.target_mask.any())
    predicted = predictions.reg_predictions[predictions.target_mask]
    loss = logits.square().mean() + predicted.square().mean()
    loss.backward()

    parameter_groups = {
        "student": model.observed_set.projectors.parameters(),
        "adapter": model.observed_set.adapters.parameters(),
        "graph": model.graph_net_temporal.parameters(),
        "predictor": model.missing_predictor.parameters(),
    }
    for name, parameters in parameter_groups.items():
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None
        ]
        assert gradients, name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients), name


def test_missing_m3_graph_model_uses_one_gcnet_forward_and_updates_teacher_exactly():
    torch.manual_seed(17)
    model = MissingM3GraphModel(**_model_arguments()).eval()
    features = torch.randn(3, 2, 9)
    availability = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 1]],
            [[1, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    qmask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    lengths = [3, 2]

    logits, hidden, latents, predictions = model(
        [features], availability, qmask, umask, lengths, predict_missing=True
    )

    assert logits.shape == (3, 2, 6)
    assert hidden.shape == (3, 2, 10)
    assert predictions is not None
    assert not hasattr(model, "linear_rec")
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())

    before = copy.deepcopy(model.teacher.state_dict())
    with torch.no_grad():
        for parameter in model.observed_set.projectors.parameters():
            parameter.add_(1.0)
    students = model.observed_set.projectors.state_dict()
    model.update_teacher(0.9)
    for key, value in model.teacher.state_dict().items():
        ASSERT_CLOSE(value, before[key] * 0.9 + students[key] * 0.1)


def test_balanced_rate_schedule_covers_all_rates_and_rotates_by_epoch():
    schedule = BalancedBatchRateSchedule(MISSING_RATES)

    first = tuple(schedule.rate_for(epoch=0, batch_index=index) for index in range(8))
    second = tuple(schedule.rate_for(epoch=1, batch_index=index) for index in range(8))

    assert first == MISSING_RATES
    assert second == MISSING_RATES[1:] + MISSING_RATES[:1]
    assert set(first) == set(MISSING_RATES)


class _CountingOptimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.zero_grad_calls = 0
        self.step_gradients = []

    def zero_grad(self, set_to_none=False):
        self.zero_grad_calls += 1
        if set_to_none:
            self.parameter.grad = None
        elif self.parameter.grad is not None:
            self.parameter.grad.zero_()

    def step(self):
        self.step_gradients.append(self.parameter.grad.detach().clone())


class _LifecycleModel(torch.nn.Module):
    def __init__(self, events=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_rates = []
        self.teacher_calls = 0
        self.ema_calls = 0
        self.events = events

    def forward(
        self,
        features,
        availability,
        qmask,
        umask,
        lengths,
        predict_missing,
    ):
        del features, qmask, umask, lengths
        assert predict_missing is True
        rate_index = int(availability.item())
        rate = MISSING_RATES[rate_index]
        self.forward_rates.append(rate)
        if self.events is not None:
            self.events.append(("forward", rate))
        logits = (self.weight * (rate_index + 1)).reshape(1, 1, 1)
        return logits, None, None, logits

    def encode_teacher_targets(self, complete):
        self.teacher_calls += 1
        return {"complete": complete[0]}

    def update_teacher(self, tau):
        assert tau == pytest.approx(0.996)
        self.ema_calls += 1


def _install_train_epoch_lifecycle_fakes(monkeypatch):
    prepared_rates = []
    clipped_gradients = []
    events = []

    def prepare_view(data, schedule, epoch, dimensions):
        del data, epoch, dimensions
        rate_index = MISSING_RATES.index(schedule)
        prepared_rates.append(schedule)
        events.append(("prepare", schedule))
        return {
            "complete": torch.ones(1, 1, 1),
            "incomplete": torch.ones(1, 1, 1),
            "availability": torch.tensor(float(rate_index)),
            "qmask": torch.ones(1, 1),
            "umask": torch.ones(1, 1),
            "labels": torch.zeros(1, 1),
            "lengths": [1],
        }

    def jepa_loss(predictions, teacher, temperature):
        assert teacher.keys() == {"complete"}
        assert temperature == pytest.approx(0.03)
        rate_index = int(predictions.detach().item()) - 1
        zero = predictions.sum() * 0.0
        return MissingM3Loss(
            total=zero,
            regression=zero,
            contrastive=zero,
            target_count=int(rate_index > 0),
        )

    def clip_grad_norm(parameters, max_norm):
        assert max_norm == pytest.approx(1.0)
        gradients = [
            parameter.grad.detach().clone()
            for parameter in parameters
            if parameter.grad is not None
        ]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        clipped_gradients.append(gradients)

    monkeypatch.setattr(train_gcnet, "_move_batch", lambda raw, device: raw)
    monkeypatch.setattr(train_gcnet, "_prepare_view", prepare_view)
    monkeypatch.setattr(
        train_gcnet,
        "_task_loss",
        lambda dataset, logits, labels, umask, mosi_task_mode, *args: logits.sum(),
    )
    monkeypatch.setattr(train_gcnet, "missing_m3_loss", jepa_loss)
    monkeypatch.setattr(
        train_gcnet,
        "_collect_predictions",
        lambda dataset, logits, labels, umask, mosi_task_mode: (
            train_gcnet.np.array([0.0]),
            train_gcnet.np.array([0.0]),
            train_gcnet.np.array([0.0]),
        ),
    )
    monkeypatch.setattr(
        train_gcnet,
        "_metrics",
        lambda dataset, labels, predictions, mosi_task_mode: {"weighted_f1": 1.0},
    )
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_grad_norm)
    return prepared_rates, clipped_gradients, events


def test_all_train_rate_mode_averages_eight_views_with_one_update_per_batch(monkeypatch):
    prepared_rates, clipped_gradients, events = _install_train_epoch_lifecycle_fakes(
        monkeypatch
    )
    model = _LifecycleModel(events)
    optimizer = _CountingOptimizer(model.weight)
    config = TrainConfig(train_rate_mode="all")

    metrics = train_gcnet.train_epoch(
        model=model,
        loader=[["first"], ["second"]],
        optimizer=optimizer,
        config=config,
        schedules={rate: rate for rate in MISSING_RATES},
        epoch=0,
        dimensions=(1, 1, 1),
        device=torch.device("cpu"),
    )

    assert prepared_rates == list(MISSING_RATES) * 2
    assert model.forward_rates == list(MISSING_RATES) * 2
    assert events == [
        event
        for _ in range(2)
        for rate in MISSING_RATES
        for event in (("prepare", rate), ("forward", rate))
    ]
    assert model.teacher_calls == 2
    assert optimizer.zero_grad_calls == 2
    assert len(clipped_gradients) == 2
    assert len(optimizer.step_gradients) == 2
    assert all(torch.isfinite(value) for value in optimizer.step_gradients)
    assert all(value.item() == pytest.approx(4.5) for value in optimizer.step_gradients)
    assert model.ema_calls == 2
    assert metrics["loss"] == pytest.approx(4.5)
    assert metrics["classification_loss"] == pytest.approx(4.5)
    assert metrics["jepa_target_count"] == 14
    assert metrics["optimizer_steps"] == 2
    assert metrics["rate_batch_counts"] == {str(rate): 2 for rate in MISSING_RATES}


def test_fixed_rate_mode_uses_only_the_registered_rate_for_every_batch(monkeypatch):
    prepared_rates, clipped_gradients, events = _install_train_epoch_lifecycle_fakes(
        monkeypatch
    )
    model = _LifecycleModel(events)
    optimizer = _CountingOptimizer(model.weight)
    config = TrainConfig(train_rate_mode="fixed", fixed_missing_rate=0.5)

    metrics = train_gcnet.train_epoch(
        model=model,
        loader=[["first"], ["second"]],
        optimizer=optimizer,
        config=config,
        schedules={rate: rate for rate in MISSING_RATES},
        epoch=7,
        dimensions=(1, 1, 1),
        device=torch.device("cpu"),
    )

    assert prepared_rates == [0.5, 0.5]
    assert model.forward_rates == [0.5, 0.5]
    assert events == [
        ("prepare", 0.5),
        ("forward", 0.5),
        ("prepare", 0.5),
        ("forward", 0.5),
    ]
    assert model.teacher_calls == 2
    assert optimizer.zero_grad_calls == 2
    assert len(clipped_gradients) == 2
    assert len(optimizer.step_gradients) == 2
    assert all(value.item() == pytest.approx(6.0) for value in optimizer.step_gradients)
    assert model.ema_calls == 2
    assert metrics["optimizer_steps"] == 2
    assert metrics["rate_batch_counts"]["0.5"] == 2
    assert sum(metrics["rate_batch_counts"].values()) == 2


@pytest.mark.parametrize(
    "mode,rate,match",
    [
        ("fixed", None, "fixed_missing_rate"),
        (
            "fixed",
            0.25,
            "official missing rates",
        ),
        (
            "all",
            0.5,
            "only valid when train_rate_mode='fixed'",
        ),
    ],
)
def test_fixed_rate_configuration_rejects_ambiguous_or_invalid_contracts(
    mode, rate, match
):
    config = TrainConfig(train_rate_mode=mode, fixed_missing_rate=rate)
    with pytest.raises(ValueError, match=match):
        train_gcnet._fixed_missing_rate(config)


def test_sparsity_jepa_weights_preserve_the_active_rate_budget():
    uniform = [
        train_gcnet._jepa_rate_weight(rate, "uniform")
        for rate in MISSING_RATES
    ]
    weighted = [
        train_gcnet._jepa_rate_weight(rate, "sparsity-budget")
        for rate in MISSING_RATES
    ]

    assert uniform == pytest.approx([1.0] * len(MISSING_RATES))
    assert weighted == sorted(weighted)
    assert sum(weighted[1:]) / len(weighted[1:]) == pytest.approx(1.0)
    assert weighted[1] < 1.0 < weighted[-1]
    with pytest.raises(ValueError, match="jepa_rate_weighting"):
        train_gcnet._jepa_rate_weight(0.5, "unknown")


def test_sparsity_jepa_weighting_changes_only_the_jepa_gradient(monkeypatch):
    _install_train_epoch_lifecycle_fakes(monkeypatch)

    def jepa_loss(predictions, teacher, temperature):
        del teacher
        assert temperature == pytest.approx(0.03)
        rate_index = int(predictions.detach().item()) - 1
        active = float(rate_index > 0)
        total = predictions.sum() * active
        zero = predictions.sum() * 0.0
        return MissingM3Loss(
            total=total,
            regression=total,
            contrastive=zero,
            target_count=int(active),
        )

    monkeypatch.setattr(train_gcnet, "missing_m3_loss", jepa_loss)
    model = _LifecycleModel()
    optimizer = _CountingOptimizer(model.weight)
    config = TrainConfig(
        train_rate_mode="all",
        jepa_weight=0.1,
        jepa_rate_weighting="sparsity-budget",
    )

    train_gcnet.train_epoch(
        model=model,
        loader=[["only"]],
        optimizer=optimizer,
        config=config,
        schedules={rate: rate for rate in MISSING_RATES},
        epoch=0,
        dimensions=(1, 1, 1),
        device=torch.device("cpu"),
    )

    expected = sum(
        (index + 1)
        * (
            1.0
            + (
                0.1 * train_gcnet._jepa_rate_weight(rate, "sparsity-budget")
                if rate > 0
                else 0.0
            )
        )
        for index, rate in enumerate(MISSING_RATES)
    ) / len(MISSING_RATES)
    assert optimizer.step_gradients[0].item() == pytest.approx(expected)


def test_default_train_rate_mode_preserves_single_cyclic_view(monkeypatch):
    prepared_rates, clipped_gradients, events = _install_train_epoch_lifecycle_fakes(
        monkeypatch
    )
    model = _LifecycleModel(events)
    optimizer = _CountingOptimizer(model.weight)
    config = TrainConfig()

    metrics = train_gcnet.train_epoch(
        model=model,
        loader=[["only"]],
        optimizer=optimizer,
        config=config,
        schedules={rate: rate for rate in MISSING_RATES},
        epoch=2,
        dimensions=(1, 1, 1),
        device=torch.device("cpu"),
    )

    assert config.train_rate_mode == "cyclic"
    assert prepared_rates == [0.2]
    assert model.forward_rates == [0.2]
    assert events == [("prepare", 0.2), ("forward", 0.2)]
    assert model.teacher_calls == 1
    assert optimizer.zero_grad_calls == 1
    assert len(clipped_gradients) == 1
    assert len(optimizer.step_gradients) == 1
    assert optimizer.step_gradients[0].item() == pytest.approx(3.0)
    assert model.ema_calls == 1
    assert metrics["loss"] == pytest.approx(3.0)
    assert metrics["optimizer_steps"] == 1
    assert sum(metrics["rate_batch_counts"].values()) == 1
    assert metrics["rate_batch_counts"]["0.2"] == 1


def test_train_rate_mode_cli_defaults_and_persists_in_run_artifacts(tmp_path):
    required = [
        "--audio-feature",
        "a",
        "--text-feature",
        "t",
        "--video-feature",
        "v",
        "--output-dir",
        "out",
    ]
    parser = build_parser()
    assert parser.parse_args(required).train_rate_mode == "cyclic"
    args = parser.parse_args(required + ["--train-rate-mode", "all"])
    config = TrainConfig(train_rate_mode=args.train_rate_mode)
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "best.pt"

    train_gcnet._write_run_config(config_path, config)
    train_gcnet._save_best_checkpoint(
        checkpoint_path,
        model_state={"weight": torch.tensor([1.0])},
        config_value=config,
        epoch=3,
        validation_mean_weighted_f1=0.75,
    )

    assert json.loads(config_path.read_text(encoding="utf-8"))["train_rate_mode"] == "all"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["config"]["train_rate_mode"] == "all"
    with pytest.raises(SystemExit):
        parser.parse_args(required + ["--train-rate-mode", "invalid"])


def test_fixed_rate_cli_exposes_and_persists_the_selected_missing_rate(tmp_path):
    required = [
        "--audio-feature",
        "a",
        "--text-feature",
        "t",
        "--video-feature",
        "v",
        "--output-dir",
        "out",
    ]
    args = build_parser().parse_args(
        required
        + [
            "--train-rate-mode",
            "fixed",
            "--train-missing-rate",
            "0.5",
        ]
    )
    config = TrainConfig(
        train_rate_mode=args.train_rate_mode,
        fixed_missing_rate=args.train_missing_rate,
    )
    config_path = tmp_path / "config.json"

    train_gcnet._write_run_config(config_path, config)

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert args.train_rate_mode == "fixed"
    assert args.train_missing_rate == pytest.approx(0.5)
    assert saved["train_rate_mode"] == "fixed"
    assert saved["fixed_missing_rate"] == pytest.approx(0.5)


def test_fixed_rate_protocol_selects_and_tests_only_the_training_rate(
    monkeypatch, tmp_path
):
    evaluated = []

    class FixedRateLifecycleModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            del args, kwargs
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.ema_step = 200
            self.readout_type = "shared"
            self.readout_rank = 8

    loaders = ([["train"]], [["validation"]], [["test"]], 1, 1, 1)
    monkeypatch.setattr(train_gcnet, "get_loaders", lambda **_kwargs: loaders)
    monkeypatch.setattr(train_gcnet, "MissingM3GraphModel", FixedRateLifecycleModel)
    monkeypatch.setattr(
        train_gcnet,
        "_schedules",
        lambda config, split: {rate: (split, rate) for rate in MISSING_RATES},
    )
    monkeypatch.setattr(
        train_gcnet,
        "train_epoch",
        lambda *_args, **_kwargs: {
            "weighted_f1": 0.5,
            "classification_loss": 1.0,
            "jepa_loss": 0.5,
        },
    )

    def evaluate_rate(model, loader, schedule, dataset, dimensions, device, collect, **kwargs):
        del model, loader, dataset, dimensions, device, kwargs
        split, rate = schedule
        evaluated.append((split, rate, collect))
        metrics = {"weighted_f1": 0.8 + rate / 100}
        if not collect:
            return metrics, None
        metrics["mask_sha256"] = "mask-{}".format(rate)
        return metrics, {
            "predictions": train_gcnet.np.array([1.0]),
            "labels": train_gcnet.np.array([1.0]),
            "availability": train_gcnet.np.ones((1, 3)),
        }

    monkeypatch.setattr(train_gcnet, "evaluate_rate", evaluate_rate)

    result = train_gcnet.run_experiment(
        TrainConfig(
            dataset="CMUMOSI",
            fold=1,
            epochs=2,
            device="cpu",
            train_rate_mode="fixed",
            fixed_missing_rate=0.5,
        ),
        "audio",
        "text",
        "visual",
        tmp_path,
    )

    assert evaluated == [
        ("validation", 0.5, False),
        ("validation", 0.5, False),
        ("test", 0.5, True),
    ]
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [set(record["validation"]) for record in history] == [{"0.5"}, {"0.5"}]
    assert result["best_epoch"] == 1
    assert result["best_validation_mean_weighted_f1"] == pytest.approx(0.805)
    assert set(result["test"]) == {"0.5"}
    assert result["train_missing_rate"] == pytest.approx(0.5)
    assert result["selection_missing_rates"] == [0.5]
    assert (tmp_path / "predictions_miss_0p5.npz").is_file()
    assert not (tmp_path / "predictions_miss_0p0.npz").exists()


def test_protocol_rates_preserve_eight_rate_lifecycle_for_existing_modes():
    assert train_gcnet._protocol_rates(TrainConfig(train_rate_mode="all")) == MISSING_RATES
    assert train_gcnet._protocol_rates(TrainConfig(train_rate_mode="cyclic")) == MISSING_RATES
    assert train_gcnet._protocol_rates(
        TrainConfig(train_rate_mode="fixed", fixed_missing_rate=0.7)
    ) == (0.7,)


def test_mosi_task_mode_cli_defaults_to_regression_and_accepts_binary_for_mosi():
    required = [
        "--audio-feature",
        "a",
        "--text-feature",
        "t",
        "--video-feature",
        "v",
        "--output-dir",
        "out",
    ]
    parser = build_parser()

    assert parser.parse_args(required).mosi_task_mode == "regression"
    args = parser.parse_args(
        required + ["--dataset", "CMUMOSI", "--mosi-task-mode", "binary"]
    )
    assert args.dataset == "CMUMOSI"
    assert args.mosi_task_mode == "binary"


def test_mosi_task_mode_binary_contract_is_restricted_to_cmumosi():
    with pytest.raises(ValueError, match="CMUMOSI"):
        train_gcnet._resolve_task_contract("IEMOCAPSix", "binary")


def test_mosi_task_mode_contract_selects_regression_or_binary_shape():
    regression = train_gcnet._resolve_task_contract("CMUMOSI", "regression")
    binary = train_gcnet._resolve_task_contract("CMUMOSI", "binary")

    assert regression["task"] == "regression"
    assert regression["num_classes"] == 1
    assert binary["task"] == "binary"
    assert binary["num_classes"] == 2


def test_mosi_task_mode_contract_rejects_invalid_mode():
    with pytest.raises(ValueError, match="unsupported MOSI task mode"):
        train_gcnet._resolve_task_contract("CMUMOSI", "multiclass")


def test_mosi_task_mode_main_passes_binary_to_run_experiment(monkeypatch):
    captured = {}

    def run_experiment(config_value, *roots, output_dir):
        captured["config"] = config_value
        captured["roots"] = roots
        captured["output_dir"] = output_dir

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_gcnet.py",
            "--dataset",
            "CMUMOSI",
            "--mosi-task-mode",
            "binary",
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--feature-root",
            "features",
            "--output-dir",
            "out",
        ],
    )
    monkeypatch.setattr(train_gcnet.torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(train_gcnet.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(train_gcnet, "run_experiment", run_experiment)

    train_gcnet.main()

    assert captured["config"].mosi_task_mode == "binary"
    assert captured["roots"] == (
        "features/a",
        "features/t",
        "features/v",
    )
    assert captured["output_dir"] == "out"


def test_graph_branch_mode_cli_defaults_and_main_passthrough(monkeypatch):
    required = [
        "--audio-feature",
        "a",
        "--text-feature",
        "t",
        "--video-feature",
        "v",
        "--output-dir",
        "out",
    ]
    parser = build_parser()
    assert parser.parse_args(required).graph_branch_mode == "both"
    assert parser.parse_args(
        required + ["--graph-branch-mode", "temporal-only"]
    ).graph_branch_mode == "temporal-only"
    assert parser.parse_args(
        required + ["--graph-branch-mode", "speaker-only"]
    ).graph_branch_mode == "speaker-only"

    captured = {}

    def run_experiment(config_value, *roots, output_dir):
        del roots, output_dir
        captured["config"] = config_value

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_gcnet.py",
            *required,
            "--feature-root",
            "features",
            "--graph-branch-mode",
            "speaker-only",
        ],
    )
    monkeypatch.setattr(train_gcnet.torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(train_gcnet.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(train_gcnet, "run_experiment", run_experiment)

    train_gcnet.main()

    assert captured["config"].graph_branch_mode == "speaker-only"


def test_mosi_task_mode_run_experiment_builds_one_or_two_output_classes(
    monkeypatch, tmp_path
):
    captured_classes = []

    class CapturingModel(torch.nn.Module):
        def __init__(self, *args, n_classes, **kwargs):
            super().__init__()
            del args, kwargs
            captured_classes.append(n_classes)
            self.dummy = torch.nn.Parameter(torch.zeros(()))

    empty_loaders = ([[]], [[]], [[]], 1, 1, 1)
    monkeypatch.setattr(train_gcnet, "MissingM3GraphModel", CapturingModel)
    monkeypatch.setattr(train_gcnet, "get_loaders", lambda **_kwargs: empty_loaders)
    monkeypatch.setattr(train_gcnet, "_schedules", lambda *_args: {})

    for mode in ("regression", "binary"):
        with pytest.raises(RuntimeError, match="no best checkpoint"):
            train_gcnet.run_experiment(
                TrainConfig(
                    dataset="CMUMOSI",
                    fold=1,
                    epochs=0,
                    device="cpu",
                    mosi_task_mode=mode,
                ),
                "audio",
                "text",
                "visual",
                tmp_path / mode,
            )

    assert captured_classes == [1, 2]


def test_mosi_task_mode_binary_collection_aligns_multiple_batches_and_steps():
    logits = torch.tensor(
        [
            [[4.0, -1.0], [-3.0, 2.0]],
            [[-2.0, 3.0], [5.0, -4.0]],
        ]
    )
    labels = torch.tensor([[-1.0, 2.0], [3.0, 0.0]])
    umask = torch.ones(2, 2)

    predicted, binary, continuous = train_gcnet._collect_predictions(
        "CMUMOSI", logits, labels, umask, mosi_task_mode="binary"
    )

    assert predicted.tolist() == [0, 1, 1]
    assert binary.tolist() == [0, 1, 1]
    assert continuous.tolist() == [-1.0, 2.0, 3.0]


def test_train_rate_mode_preserves_legacy_positional_config_order():
    legacy_field_names = (
        "dataset",
        "fold",
        "seed",
        "base_model",
        "window_past",
        "window_future",
        "hidden",
        "dropout",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "latent_dim",
        "num_experts",
        "top_k",
        "projector_dropout",
        "predictor_dropout",
        "fusion_type",
        "local_context_residual",
        "local_fusion_hidden_dim",
        "local_fusion_dropout",
        "jepa_weight",
        "temperature",
        "ema_tau",
        "gradient_clip_norm",
        "time_attention",
        "evaluation_protocol",
        "validation_fraction",
        "device",
    )
    legacy_values = (
        "CMUMOSI",
        1,
        77,
        "GRU",
        -1,
        3,
        128,
        0.2,
        16,
        12,
        2e-3,
        3e-5,
        64,
        3,
        1,
        0.05,
        0.06,
        "slot",
        True,
        96,
        0.15,
        0.2,
        0.07,
        0.99,
        0.5,
        True,
        "strict",
        0.2,
        "cpu",
    )

    config = TrainConfig(*legacy_values)

    serialized = asdict(config)
    assert tuple(serialized[name] for name in legacy_field_names) == legacy_values
    assert config.train_rate_mode == "cyclic"


def test_checkpoint_score_is_equal_mean_of_eight_validation_rates():
    metrics = {
        rate: {"weighted_f1": float(index)}
        for index, rate in enumerate(MISSING_RATES)
    }

    assert mean_validation_weighted_f1(metrics) == pytest.approx(3.5)
    with pytest.raises(ValueError, match="all eight"):
        mean_validation_weighted_f1(dict(list(metrics.items())[:-1]))


def test_best_epoch_uses_mixed_validation_mean_not_any_single_rate_peak():
    first = {
        rate: {"weighted_f1": 0.8 if rate == 0.0 else 0.4}
        for rate in MISSING_RATES
    }
    second = {
        rate: {"weighted_f1": 0.5}
        for rate in MISSING_RATES
    }

    assert select_best_epoch([
        {"epoch": 1, "validation": first},
        {"epoch": 2, "validation": second},
    ]) == 2


@pytest.mark.parametrize(
    "dataset,num_folds,num_classes,num_speakers,task",
    [
        ("IEMOCAPFour", 5, 4, 2, "classification"),
        ("IEMOCAPSix", 5, 6, 2, "classification"),
        ("CMUMOSI", 1, 1, 1, "regression"),
        ("CMUMOSEI", 1, 1, 1, "regression"),
    ],
)
def test_dataset_shape_matches_gcnet_task_contract(
    dataset, num_folds, num_classes, num_speakers, task
):
    actual = _dataset_shape(dataset)
    assert actual == {
        "num_folds": num_folds,
        "num_classes": num_classes,
        "num_speakers": num_speakers,
        "task": task,
    }


def test_task_loss_uses_cross_entropy_for_erc_and_mse_for_sentiment():
    logits = torch.tensor([[[2.0, -1.0]], [[-1.0, 2.0]]])
    labels = torch.tensor([[0, 1]])
    umask = torch.ones(1, 2)
    expected_ce = torch.nn.functional.cross_entropy(
        logits.transpose(0, 1).reshape(-1, 2), labels.reshape(-1)
    )
    ASSERT_CLOSE(_task_loss("IEMOCAPFour", logits, labels, umask), expected_ce)

    prediction = torch.tensor([[[0.5]], [[-1.0]]])
    target = torch.tensor([[1.5, 1.0]])
    expected_mse = torch.tensor(((0.5 - 1.5) ** 2 + (-1.0 - 1.0) ** 2) / 2)
    ASSERT_CLOSE(_task_loss("CMUMOSI", prediction, target, umask), expected_mse)


def test_mosi_binary_task_loss_excludes_zero_labels_and_padding():
    logits = torch.tensor(
        [
            [[3.0, -1.0]],
            [[-4.0, 4.0]],
            [[-2.0, 2.0]],
            [[5.0, -5.0]],
        ]
    )
    labels = torch.tensor([[-2.0, 0.0, 1.5, -3.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    expected = torch.nn.functional.cross_entropy(
        logits[[0, 2], 0], torch.tensor([0, 1])
    )

    actual = _task_loss(
        "CMUMOSI", logits, labels, umask, mosi_task_mode="binary"
    )

    ASSERT_CLOSE(actual, expected)


def test_mosi_binary_empty_selection_returns_connected_zero_loss():
    logits = torch.randn(3, 1, 2, requires_grad=True)
    labels = torch.tensor([[0.0, 0.0, -1.0]])
    umask = torch.tensor([[1.0, 1.0, 0.0]])

    loss = _task_loss(
        "CMUMOSI", logits, labels, umask, mosi_task_mode="binary"
    )
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_mosi_binary_connected_zero_loss_allows_auxiliary_gradient():
    logits = torch.randn(3, 1, 2, requires_grad=True)
    labels = torch.zeros(1, 3)
    umask = torch.ones(1, 3)
    classification = _task_loss(
        "CMUMOSI", logits, labels, umask, mosi_task_mode="binary"
    )
    auxiliary = logits.square().mean()

    total = classification + auxiliary
    total.backward()

    assert torch.isfinite(total)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_mosi_binary_prediction_collection_excludes_zero_and_padding():
    logits = torch.tensor(
        [
            [[3.0, -1.0]],
            [[-4.0, 4.0]],
            [[-2.0, 2.0]],
            [[5.0, -5.0]],
        ]
    )
    labels = torch.tensor([[-2.0, 0.0, 1.5, -3.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

    predictions, metric_labels, continuous_labels = (
        train_gcnet._collect_predictions(
            "CMUMOSI", logits, labels, umask, mosi_task_mode="binary"
        )
    )

    assert predictions.tolist() == [0, 1]
    assert metric_labels.tolist() == [0, 1]
    assert continuous_labels.tolist() == [-2.0, 1.5]


def test_mosi_binary_metrics_use_direct_class_arrays_without_regression_fields():
    labels = torch.tensor([0, 0, 1, 1]).numpy()
    predictions = torch.tensor([0, 1, 1, 1]).numpy()

    result = _metrics(
        "CMUMOSI", labels, predictions, mosi_task_mode="binary"
    )

    assert set(result) == {"accuracy", "weighted_f1", "macro_f1"}
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["weighted_f1"] == pytest.approx(0.7333333333333334)
    assert result["macro_f1"] == pytest.approx(0.7333333333333334)


def test_default_and_explicit_regression_helpers_are_equivalent():
    logits = torch.tensor([[[0.5]], [[-1.0]], [[2.0]]])
    labels = torch.tensor([[1.5, 0.0, -2.0]])
    umask = torch.tensor([[1.0, 1.0, 0.0]])

    ASSERT_CLOSE(
        _task_loss("CMUMOSI", logits, labels, umask),
        _task_loss(
            "CMUMOSI", logits, labels, umask, mosi_task_mode="regression"
        ),
        rtol=0,
        atol=0,
    )
    default_collection = train_gcnet._collect_predictions(
        "CMUMOSI", logits, labels, umask
    )
    explicit_collection = train_gcnet._collect_predictions(
        "CMUMOSI", logits, labels, umask, mosi_task_mode="regression"
    )
    assert len(default_collection) == len(explicit_collection) == 3
    for default, explicit in zip(default_collection, explicit_collection):
        assert default.tolist() == explicit.tolist()
    default_metrics = _metrics(
        "CMUMOSI", default_collection[1], default_collection[0]
    )
    explicit_metrics = _metrics(
        "CMUMOSI",
        explicit_collection[1],
        explicit_collection[0],
        mosi_task_mode="regression",
    )
    assert default_metrics == explicit_metrics


def test_task_smooth_l1_is_robust_regression_and_keeps_zero_labels():
    logits = torch.tensor([[[3.0]], [[2.0]], [[-4.0]], [[99.0]]])
    labels = torch.tensor([[0.0, 1.5, -1.0, -99.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

    actual = _task_loss(
        "CMUMOSI",
        logits,
        labels,
        umask,
        task_regression_loss="smooth-l1",
        task_smooth_l1_beta=1.0,
    )
    expected = torch.nn.functional.smooth_l1_loss(
        torch.tensor([3.0, 2.0, -4.0]),
        torch.tensor([0.0, 1.5, -1.0]),
        beta=1.0,
    )

    ASSERT_CLOSE(actual, expected, rtol=0, atol=0)
    assert actual < _task_loss("CMUMOSI", logits, labels, umask)


def test_task_regression_loss_defaults_to_exact_mse_and_validates_beta():
    logits = torch.tensor([[[0.5]], [[-1.0]]])
    labels = torch.tensor([[1.5, 1.0]])
    umask = torch.ones(1, 2)

    default = _task_loss("CMUMOSI", logits, labels, umask)
    explicit = _task_loss(
        "CMUMOSI",
        logits,
        labels,
        umask,
        task_regression_loss="mse",
        task_smooth_l1_beta=1.0,
    )
    ASSERT_CLOSE(default, explicit, rtol=0, atol=0)

    with pytest.raises(ValueError, match="task_regression_loss"):
        _task_loss(
            "CMUMOSI",
            logits,
            labels,
            umask,
            task_regression_loss="unknown",
        )
    with pytest.raises(ValueError, match="beta"):
        _task_loss(
            "CMUMOSI",
            logits,
            labels,
            umask,
            task_regression_loss="smooth-l1",
            task_smooth_l1_beta=0.0,
        )


def test_task_regression_loss_preserves_mse_gradients_and_bounds_smooth_l1():
    default_logits = torch.tensor([[[0.0]], [[3.0]]], requires_grad=True)
    explicit_logits = default_logits.detach().clone().requires_grad_(True)
    robust_logits = default_logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([[0.5, -2.0]])
    umask = torch.ones(1, 2)

    default = _task_loss("CMUMOSI", default_logits, labels, umask)
    explicit = _task_loss(
        "CMUMOSI",
        explicit_logits,
        labels,
        umask,
        task_regression_loss="mse",
    )
    robust = _task_loss(
        "CMUMOSI",
        robust_logits,
        labels,
        umask,
        task_regression_loss="smooth-l1",
        task_smooth_l1_beta=1.0,
    )
    default.backward()
    explicit.backward()
    robust.backward()

    ASSERT_CLOSE(default, explicit, rtol=0, atol=0)
    ASSERT_CLOSE(default_logits.grad, explicit_logits.grad, rtol=0, atol=0)
    assert torch.isfinite(robust_logits.grad).all()
    assert robust_logits.grad.abs().max() <= 0.5


def test_classification_rejects_irrelevant_smooth_l1_task_configuration():
    logits = torch.tensor([[[2.0, -1.0]]])
    labels = torch.tensor([[0]])
    umask = torch.ones(1, 1)

    with pytest.raises(ValueError, match="continuous regression"):
        _task_loss(
            "IEMOCAPFour",
            logits,
            labels,
            umask,
            task_regression_loss="smooth-l1",
        )


def test_task_regression_loss_cli_and_config_defaults_are_explicit():
    parser = build_parser()
    required = [
        "--audio-feature",
        "a",
        "--text-feature",
        "t",
        "--video-feature",
        "v",
        "--output-dir",
        "out",
    ]

    defaults = parser.parse_args(required)
    candidate = parser.parse_args(
        [
            *required,
            "--task-regression-loss",
            "smooth-l1",
            "--task-smooth-l1-beta",
            "1.0",
        ]
    )

    assert defaults.task_regression_loss == "mse"
    assert defaults.task_smooth_l1_beta == pytest.approx(1.0)
    assert TrainConfig().task_regression_loss == "mse"
    assert TrainConfig().task_smooth_l1_beta == pytest.approx(1.0)
    assert candidate.task_regression_loss == "smooth-l1"
    assert candidate.task_smooth_l1_beta == pytest.approx(1.0)


def test_jepa_rate_weighting_cli_and_config_defaults_are_explicit():
    parser = build_parser()
    required = [
        "--audio-feature",
        "a",
        "--text-feature",
        "t",
        "--video-feature",
        "v",
        "--output-dir",
        "out",
    ]

    defaults = parser.parse_args(required)
    candidate = parser.parse_args(
        [*required, "--jepa-rate-weighting", "sparsity-budget"]
    )

    assert defaults.jepa_rate_weighting == "uniform"
    assert TrainConfig().jepa_rate_weighting == "uniform"
    assert candidate.jepa_rate_weighting == "sparsity-budget"


def test_binary_evaluation_filters_artifacts_but_hashes_full_valid_mask(monkeypatch):
    availability = torch.tensor(
        [
            [[1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0]],
            [[1.0, 1.0, 0.0]],
        ]
    )
    labels = torch.tensor([[-1.0, 0.0, 2.0]])
    umask = torch.ones(1, 3)
    view = {
        "complete": torch.zeros(3, 1, 1),
        "incomplete": torch.zeros(3, 1, 1),
        "availability": availability,
        "qmask": torch.ones(3, 1, 1),
        "umask": umask,
        "labels": labels,
        "lengths": [3],
    }

    class BinaryEvaluationModel:
        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            del args, kwargs
            logits = torch.tensor(
                [[[4.0, -1.0]], [[-2.0, 2.0]], [[-3.0, 3.0]]]
            )
            return logits, None, None, None

    monkeypatch.setattr(train_gcnet, "_move_batch", lambda raw, device: raw)
    monkeypatch.setattr(
        train_gcnet,
        "_prepare_view",
        lambda data, schedule, epoch, dimensions: view,
    )

    metrics, artifacts = train_gcnet.evaluate_rate(
        model=BinaryEvaluationModel(),
        loader=[["batch"]],
        schedule=object(),
        dataset="CMUMOSI",
        dimensions=(1, 1, 1),
        device=torch.device("cpu"),
        collect=True,
        mosi_task_mode="binary",
    )

    assert artifacts is not None
    assert artifacts["predictions"].tolist() == [0, 1]
    assert artifacts["labels"].tolist() == [0, 1]
    assert artifacts["continuous_labels"].tolist() == [-1.0, 2.0]
    assert artifacts["availability"].tolist() == [
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
    assert metrics["mask_sha256"] == train_gcnet._sha256_tensor(availability)


def test_regression_evaluation_preserves_legacy_artifact_key_set(monkeypatch):
    availability = torch.tensor(
        [[[1.0, 0.0, 1.0]], [[0.0, 1.0, 1.0]]]
    )
    view = {
        "complete": torch.zeros(2, 1, 1),
        "incomplete": torch.zeros(2, 1, 1),
        "availability": availability,
        "qmask": torch.ones(2, 1, 1),
        "umask": torch.ones(1, 2),
        "labels": torch.tensor([[-1.0, 0.0]]),
        "lengths": [2],
    }

    class RegressionEvaluationModel:
        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            del args, kwargs
            return torch.tensor([[[-0.5]], [[0.25]]]), None, None, None

    monkeypatch.setattr(train_gcnet, "_move_batch", lambda raw, device: raw)
    monkeypatch.setattr(
        train_gcnet,
        "_prepare_view",
        lambda data, schedule, epoch, dimensions: view,
    )

    _, artifacts = train_gcnet.evaluate_rate(
        model=RegressionEvaluationModel(),
        loader=[["batch"]],
        schedule=object(),
        dataset="CMUMOSI",
        dimensions=(1, 1, 1),
        device=torch.device("cpu"),
        collect=True,
    )

    assert artifacts is not None
    assert set(artifacts) == {"predictions", "labels", "availability"}


def test_sentiment_metrics_match_gcnet_binary_nonzero_protocol():
    labels = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).numpy()
    predictions = torch.tensor([-1.0, 0.5, -3.0, 0.1, 2.0]).numpy()

    result = _metrics("CMUMOSI", labels, predictions)

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["weighted_f1"] == pytest.approx(0.7333333333333334)
    assert result["mae"] == pytest.approx(1.28)
