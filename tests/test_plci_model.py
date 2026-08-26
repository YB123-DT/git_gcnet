from unittest import mock

import pytest
import torch

from gcnet_modality_jepa.graph import batch_graphify
from gcnet_modality_jepa.model import GraphModel
from gcnet_plci_jepa.model import PLCIJEPAGraphModel
from gcnet_plci_jepa.modules import MODALITIES, normalize_latent


ASSERT_CLOSE = getattr(torch.testing, "assert_close", torch.testing.assert_allclose)


def model_arguments():
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
    )


def plci_arguments():
    return dict(
        **model_arguments(),
        latent_dim=5,
        source_dim=4,
        context_rank=3,
        innovation_rank=3,
        context_cap=0.2,
        innovation_cap=0.3,
        pattern_embedding_dim=4,
        predictor_embedding_dim=3,
    )


def inputs():
    torch.manual_seed(19)
    features = torch.randn(3, 2, 9)
    qmask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    lengths = [3, 2]
    return [features], qmask, umask, lengths


def legacy_hidden(model, inputfeats, qmask, umask, lengths):
    outputs, _ = model.lstm(inputfeats[0])
    outputs = outputs.unsqueeze(2)
    features, edge_index, edge_type, mapping = batch_graphify(
        outputs, qmask, lengths, model.n_speakers, model.window_past,
        model.window_future, "temporal", model.no_cuda
    )
    assert len(mapping) == 3
    hidden1 = model.graph_net_temporal(
        features, edge_index, edge_type, lengths, umask
    )
    features, edge_index, edge_type, mapping = batch_graphify(
        outputs, qmask, lengths, model.n_speakers, model.window_past,
        model.window_future, "speaker", model.no_cuda
    )
    assert len(mapping) == model.n_speakers ** 2
    hidden2 = model.graph_net_speaker(
        features, edge_index, edge_type, lengths, umask
    )
    return hidden1 + hidden2


def availability_atv(umask):
    return umask.T.unsqueeze(-1).expand(-1, -1, 3).clone()


def mixed_availability(umask):
    availability = availability_atv(umask)
    availability[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    availability[1, 0] = torch.tensor([1.0, 1.0, 0.0])
    availability[0, 1] = torch.tensor([0.0, 1.0, 1.0])
    return availability


def auxiliary_availability(umask):
    availability = mixed_availability(umask)
    availability[1, 1] = torch.tensor([1.0, 0.0, 1.0])
    availability[2, 0] = torch.tensor([0.0, 0.0, 1.0])
    return availability


def mask_features(features, availability, dimensions=(2, 3, 4)):
    mask = torch.repeat_interleave(
        availability, torch.tensor(dimensions), dim=-1
    )
    return features * mask


def test_original_forward_is_unchanged_after_encode_hidden_extraction():
    torch.manual_seed(7)
    model = GraphModel(**model_arguments()).eval()
    inputfeats, qmask, umask, lengths = inputs()
    expected_hidden = legacy_hidden(model, inputfeats, qmask, umask, lengths)
    expected = (
        model.smax_fc(expected_hidden),
        [model.linear_rec(expected_hidden)],
        expected_hidden,
    )

    actual = model(inputfeats, qmask, umask, lengths)

    for left, right in zip(
        (expected[0], expected[1][0], expected[2]),
        (actual[0], actual[1][0], actual[2]),
    ):
        ASSERT_CLOSE(left, right, rtol=0, atol=0)


def test_encode_hidden_checks_and_adds_post_rnn_residual():
    model = GraphModel(**model_arguments()).eval()
    inputfeats, qmask, umask, lengths = inputs()
    residual = torch.randn(3, 2, 8)
    seen = []
    handle = model.graph_net_temporal.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    try:
        model.encode_hidden(inputfeats, qmask, umask, lengths, residual)
    finally:
        handle.remove()
    recurrent, _ = model.lstm(inputfeats[0])
    expected, _, _, _ = batch_graphify(
        (recurrent + residual).unsqueeze(2), qmask, lengths,
        model.n_speakers, model.window_past, model.window_future,
        "temporal", model.no_cuda,
    )
    ASSERT_CLOSE(seen[0], expected)
    with pytest.raises(ValueError, match="pre_graph_residual"):
        model.encode_hidden(
            inputfeats, qmask, umask, lengths, torch.zeros(3, 2, 7)
        )


def test_pattern_residual_uses_model_dtype_for_uint8_mask_bank():
    model = PLCIJEPAGraphModel(**plci_arguments()).eval()
    _, _, umask, _ = inputs()
    availability = mixed_availability(umask).to(torch.uint8)

    residual = model.pattern_residual(availability, umask, allow_atv=True)

    assert residual.dtype == model.pattern_projection.weight.dtype


def test_atv_natural_path_is_exact_and_bypasses_student_and_pattern_modules():
    torch.manual_seed(31)
    model = PLCIJEPAGraphModel(**plci_arguments()).eval()
    inputfeats, qmask, umask, lengths = inputs()
    availability = availability_atv(umask)
    expected = GraphModel.forward(model, inputfeats, qmask, umask, lengths)

    with mock.patch.object(
        model.student_adapter, "forward", wraps=model.student_adapter.forward
    ) as adapter, mock.patch.object(
        model.pattern_projection, "forward", wraps=model.pattern_projection.forward
    ) as pattern:
        log_prob, reconstruction, hidden, latents = model.forward_natural(
            inputfeats, availability, qmask, umask, lengths
        )

    adapter.assert_not_called()
    pattern.assert_not_called()
    for left, right in zip(
        (expected[0], expected[1][0], expected[2]),
        (log_prob, reconstruction[0], hidden),
    ):
        ASSERT_CLOSE(left, right, rtol=0, atol=0)
    assert all(torch.count_nonzero(latents[name]) == 0 for name in MODALITIES)


def test_mixed_natural_masks_missing_blocks_and_adds_pattern_after_rnn():
    model = PLCIJEPAGraphModel(**plci_arguments()).eval()
    inputfeats, qmask, umask, lengths = inputs()
    availability = mixed_availability(umask)
    seen_adapter = []
    seen_rnn = []
    seen_graph = []
    handles = [
        model.student_adapter.register_forward_pre_hook(
            lambda _module, args: seen_adapter.append(args[0].detach().clone())
        ),
        model.lstm.register_forward_hook(
            lambda _module, _args, output: seen_rnn.append(output[0].detach().clone())
        ),
        model.graph_net_temporal.register_forward_pre_hook(
            lambda _module, args: seen_graph.append(args[0].detach().clone())
        ),
    ]
    try:
        model.forward_natural(inputfeats, availability, qmask, umask, lengths)
    finally:
        for handle in handles:
            handle.remove()

    assert torch.equal(seen_adapter[0], inputfeats[0])
    adapted, _ = model.student_adapter(inputfeats[0], availability)
    expected_rnn, _ = model.lstm(adapted)
    ASSERT_CLOSE(seen_rnn[0], expected_rnn)
    residual = model.pattern_residual(availability, umask, allow_atv=True)
    expected_nodes, _, _, _ = batch_graphify(
        (expected_rnn + residual).unsqueeze(2), qmask, lengths,
        model.n_speakers, model.window_past, model.window_future,
        "temporal", model.no_cuda,
    )
    ASSERT_CLOSE(seen_graph[0], expected_nodes)


def test_auxiliary_runs_one_backbone_and_skips_classifier_and_reconstruction():
    model = PLCIJEPAGraphModel(**plci_arguments()).eval()
    inputfeats, qmask, umask, lengths = inputs()
    availability = auxiliary_availability(umask)
    source = mask_features(inputfeats[0], availability)

    with mock.patch.object(
        model, "encode_hidden", wraps=model.encode_hidden
    ) as encode, mock.patch.object(
        model.smax_fc, "forward", wraps=model.smax_fc.forward
    ) as classifier, mock.patch.object(
        model.linear_rec, "forward", wraps=model.linear_rec.forward
    ) as reconstruction:
        predictions, hidden, latents = model.forward_auxiliary(
            source, availability, qmask, umask, lengths
        )

    assert predictions.targets
    assert hidden.shape == (3, 2, 10)
    assert set(latents) == set(MODALITIES)
    assert encode.call_count == 1
    classifier.assert_not_called()
    reconstruction.assert_not_called()


def test_auxiliary_rejects_nonzero_missing_source_and_never_accepts_teacher():
    model = PLCIJEPAGraphModel(**plci_arguments()).eval()
    inputfeats, qmask, umask, lengths = inputs()
    availability = auxiliary_availability(umask)
    leaking = inputfeats[0].clone()
    with pytest.raises(ValueError, match="missing modality blocks must be zero"):
        model.forward_auxiliary(leaking, availability, qmask, umask, lengths)
    with pytest.raises(TypeError):
        model.forward_auxiliary(
            mask_features(leaking, availability), availability, qmask, umask,
            lengths, teacher_features=leaking,
        )


def test_teacher_is_separate_normalized_and_does_not_affect_auxiliary_predictions():
    model = PLCIJEPAGraphModel(**plci_arguments()).eval()
    inputfeats, qmask, umask, lengths = inputs()
    availability = auxiliary_availability(umask)
    source = mask_features(inputfeats[0], availability)
    first, _, _ = model.forward_auxiliary(
        source, availability, qmask, umask, lengths
    )
    before = [record.paths.detach().clone() for record in first.targets]
    changed_teacher_features = inputfeats[0].clone()
    changed_teacher_features[..., 2:5].add_(1000.0)
    teacher = model.encode_teacher_targets(changed_teacher_features)
    second, _, _ = model.forward_auxiliary(
        source, availability, qmask, umask, lengths
    )

    assert all(not value.requires_grad for value in teacher.values())
    for name in MODALITIES:
        ASSERT_CLOSE(teacher[name], normalize_latent(teacher[name]))
    for old, record in zip(before, second.targets):
        ASSERT_CLOSE(old, record.paths, rtol=0, atol=0)


def test_gradients_reach_student_pattern_gcnet_and_predictor_but_not_teacher():
    model = PLCIJEPAGraphModel(**plci_arguments()).train()
    inputfeats, qmask, umask, lengths = inputs()
    availability = auxiliary_availability(umask)
    source = mask_features(inputfeats[0], availability)
    log_prob, reconstruction, hidden, _ = model.forward_natural(
        [source], availability, qmask, umask, lengths
    )
    predictions, auxiliary_hidden, _ = model.forward_auxiliary(
        source, availability, qmask, umask, lengths
    )
    loss = log_prob.sum() + reconstruction[0].sum() + auxiliary_hidden.sum()
    loss = loss + sum(record.paths.sum() for record in predictions.targets)
    loss.backward()

    groups = (
        model.student_adapter.parameters(),
        model.pattern_projection.parameters(),
        model.graph_net_temporal.parameters(),
        model.predictor.parameters(),
    )
    assert all(any(p.grad is not None for p in parameters) for parameters in groups)
    assert all(p.grad is None for p in model.teacher.parameters())
    assert not model.teacher.training
    model.train()
    assert not model.teacher.training


def test_update_teacher_tracks_tau_and_step_and_pattern_validation_is_strict():
    model = PLCIJEPAGraphModel(**plci_arguments())
    assert model.ema_step == 0
    model.update_teacher(0.9)
    assert model.ema_step == 1
    assert model.last_teacher_tau == 0.9
    _, qmask, umask, lengths = inputs()
    invalid = availability_atv(umask)
    invalid[0, 0] = 0
    with pytest.raises(ValueError, match="pattern"):
        model.pattern_residual(invalid, umask, allow_atv=True)
    padding = availability_atv(umask)
    residual = model.pattern_residual(padding, umask, allow_atv=True)
    assert torch.count_nonzero(residual[~umask.T.bool()]) == 0
