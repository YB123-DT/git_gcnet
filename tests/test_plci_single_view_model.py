import pytest
import torch

from gcnet_plci_jepa.loss import plci_jepa_loss
from gcnet_plci_jepa.modules import MODALITIES
from gcnet_plci_single_view.model import SingleViewPLCIJEPAGraphModel


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
    features = torch.randn(4, 1, 9)
    qmask = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    umask = torch.ones(1, 4)
    lengths = [4]
    availability = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0]],
            [[0.0, 1.0, 1.0]],
        ]
    )
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    return [features], [features * expanded], availability, qmask, umask, lengths


def test_natural_prediction_skips_atv_targets_without_removing_graph_nodes():
    model = SingleViewPLCIJEPAGraphModel(**model_arguments()).eval()
    _, masked, availability, qmask, umask, lengths = inputs()

    log_prob, reconstruction, hidden, latents = model.forward_natural(
        masked, availability, qmask, umask, lengths
    )
    predictions = model.predict_natural(latents, hidden, availability, umask)

    assert log_prob.shape[:2] == (4, 1)
    assert reconstruction[0].shape[:2] == (4, 1)
    assert hidden.shape[:2] == (4, 1)
    assert {record.utterance_index for record in predictions.targets} == {0, 1, 3}
    assert all(record.utterance_index != 2 for record in predictions.targets)
    assert model.last_prediction_umask.shape == umask.shape
    assert model.last_prediction_umask[0, 2].item() == 0


def test_all_atv_has_zero_plci_targets_and_loss():
    model = SingleViewPLCIJEPAGraphModel(**model_arguments()).eval()
    full, _, availability, qmask, umask, lengths = inputs()
    availability = umask.T.unsqueeze(-1).expand(-1, -1, 3).clone()

    _, _, hidden, latents = model.forward_natural(
        full, availability, qmask, umask, lengths
    )
    predictions = model.predict_natural(latents, hidden, availability, umask)
    teacher_targets = model.encode_teacher_targets(full[0])
    loss, counts = plci_jepa_loss(predictions, teacher_targets)

    assert not predictions.targets
    assert loss.item() == 0.0
    assert counts["utterances"] == 0
    assert counts["targets"] == 0
    assert counts["paths"] == 0
    assert all(counts[name + "_targets"] == 0 for name in MODALITIES)


def test_natural_prediction_rejects_zero_pattern_at_valid_utterance():
    model = SingleViewPLCIJEPAGraphModel(**model_arguments()).eval()
    _, masked, availability, qmask, umask, lengths = inputs()
    availability[0, 0] = 0

    with pytest.raises(ValueError, match="pattern"):
        model.forward_natural(masked, availability, qmask, umask, lengths)


def test_natural_missing_target_never_enters_student_hidden_or_prediction():
    model = SingleViewPLCIJEPAGraphModel(**model_arguments()).eval()
    full, masked, availability, qmask, umask, lengths = inputs()
    _, _, hidden_1, latents_1 = model.forward_natural(
        masked, availability, qmask, umask, lengths
    )
    predictions_1 = model.predict_natural(
        latents_1, hidden_1, availability, umask
    )
    teacher_1 = model.encode_teacher_targets(full[0])

    changed_full = full[0].clone()
    changed_full[0, 0, 2:5].add_(1000.0)
    _, _, hidden_2, latents_2 = model.forward_natural(
        masked, availability, qmask, umask, lengths
    )
    predictions_2 = model.predict_natural(
        latents_2, hidden_2, availability, umask
    )
    teacher_2 = model.encode_teacher_targets(changed_full)

    torch.testing.assert_close(hidden_1, hidden_2, rtol=0, atol=0)
    for first, second in zip(predictions_1.targets, predictions_2.targets):
        torch.testing.assert_close(first.paths, second.paths, rtol=0, atol=0)
    assert not torch.equal(teacher_1["text"][0, 0], teacher_2["text"][0, 0])


def test_single_view_objective_reaches_student_predictor_and_gcnet_not_teacher():
    model = SingleViewPLCIJEPAGraphModel(**model_arguments()).train()
    full, masked, availability, qmask, umask, lengths = inputs()
    log_prob, reconstruction, hidden, latents = model.forward_natural(
        masked, availability, qmask, umask, lengths
    )
    predictions = model.predict_natural(latents, hidden, availability, umask)
    teacher_targets = model.encode_teacher_targets(full[0])
    jepa_loss, _ = plci_jepa_loss(predictions, teacher_targets)

    (log_prob.sum() + reconstruction[0].sum() + jepa_loss).backward()

    groups = (
        model.student_adapter.projectors.parameters(),
        model.predictor.parameters(),
        model.graph_net_temporal.parameters(),
    )
    for parameters in groups:
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
