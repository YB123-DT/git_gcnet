import sys
from pathlib import Path

import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.comp import build_model, stage_loss


def example(causal=True):
    torch.manual_seed(31)
    model = build_model(dict(hidden=8, depth=2, heads=2, dropout=0,
                             seq_len=5, temporal_causal=causal), (3, 4, 2), 7, 9)
    inputs = [torch.randn(2, 4, d) for d in (3, 4, 2)]
    availability = torch.ones(2, 4, 3, dtype=torch.bool)
    availability[0, 0, 0] = False
    availability[1, :, 2] = False
    padding = torch.tensor([[True, True, True, True], [True, True, True, False]])
    labels = torch.randint(0, 7, (2, 4))
    return model, inputs, availability, padding, labels


def _check_both_stage_forward_backward_finite(causal, first_stage):
    model, inputs, availability, padding, labels = example(causal)
    output = model(*inputs, availability, padding, first_stage=first_stage)
    assert output['logits'].shape == (2, 4, 7)
    loss = stage_loss(output, labels, padding, first_stage)
    assert torch.isfinite(loss)
    loss.backward()
    report = model.after_backward(output, labels, padding, first_stage)
    assert report['matched_parameters'] == 0
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)
    assert (model.model.nlp_head.weight.grad is None) == first_stage
    parameter = (model.model.shared_enc_a.fc1.weight if first_stage else
                 model.model.prop_former_a_1.proj_n.fc1.weight)
    before = parameter.detach().clone()
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    assert not torch.equal(before, parameter)


def _check_unavailable_and_padding_features_do_not_reach_model(first_stage):
    model, inputs, availability, padding, _ = example()
    model.eval()
    modified = [x.clone() for x in inputs]
    for i, x in enumerate(modified):
        x[~(availability[..., i] & padding)] = 10000
    with torch.no_grad():
        original = model(*inputs, availability, padding, first_stage=first_stage)['logits']
        changed = model(*modified, availability, padding, first_stage=first_stage)['logits']
    torch.testing.assert_close(original, changed)


def _check_causal_future_feature_and_mask_perturbation(first_stage):
    model, inputs, availability, padding, _ = example()
    model.eval()
    changed = [x.clone() for x in inputs]
    for x in changed:
        x[:, 2:] += 500
    new_availability = availability.clone()
    new_availability[:, 2:, :2] = ~new_availability[:, 2:, :2]
    new_availability[:, 2:, 0] = True
    with torch.no_grad():
        before = model(*inputs, availability, padding, first_stage=first_stage)['logits']
        after = model(*changed, new_availability, padding, first_stage=first_stage)['logits']
    torch.testing.assert_close(before[:, :2], after[:, :2], atol=1e-6, rtol=1e-5)


def _check_wrapper_does_not_accept_prediction_labels():
    model, inputs, availability, padding, labels = example()
    model.eval()
    with torch.no_grad():
        output = model(*inputs, availability, padding)
        first = output['logits'].clone()
        stage_loss(output, labels, padding, False)
        stage_loss(output, (labels + 1) % 7, padding, False)
    torch.testing.assert_close(first, model(*inputs, availability, padding)['logits'])


def _check_original_still_reads_future_and_has_batch_prompt_mixing():
    model, inputs, availability, padding, _ = example(False)
    model.eval()
    with torch.no_grad():
        before = model(*inputs, availability, padding)['logits']
        changed = [x.clone() for x in inputs]
        for x in changed:
            x[:, 2:] += 100
        after = model(*changed, availability, padding)['logits']
        batch_changed = [x.clone() for x in inputs]
        for x in batch_changed:
            x[1] += 100
        other = model(*batch_changed, availability, padding)['logits']
    assert not torch.allclose(before[:, :2], after[:, :2])
    assert not torch.allclose(before[0], other[0])


def _check_existing_causal_prefix_equivalence():
    model, inputs, availability, padding, _ = example()
    model.eval()
    with torch.no_grad():
        for first_stage in (True, False):
            full = model(*inputs, availability, padding, first_stage=first_stage)['logits']
            prefix = model(*[x[:, :2] for x in inputs], availability[:, :2],
                           padding[:, :2], first_stage=first_stage)['logits']
            torch.testing.assert_close(full[:, :2], prefix, atol=1e-6, rtol=1e-5)


class TestComPWrapper(unittest.TestCase):
    def test_forward_backward(self):
        for causal in (True, False):
            for stage in (True, False):
                with self.subTest(causal=causal, first_stage=stage):
                    _check_both_stage_forward_backward_finite(causal, stage)

    def test_hidden_features(self):
        for stage in (True, False):
            with self.subTest(first_stage=stage):
                _check_unavailable_and_padding_features_do_not_reach_model(stage)

    def test_causality(self):
        for stage in (True, False):
            with self.subTest(first_stage=stage):
                _check_causal_future_feature_and_mask_perturbation(stage)

    def test_labels(self):
        _check_wrapper_does_not_accept_prediction_labels()

    def test_original(self):
        _check_original_still_reads_future_and_has_batch_prompt_mixing()

    def test_prefix(self):
        _check_existing_causal_prefix_equivalence()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
