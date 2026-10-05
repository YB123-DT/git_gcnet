import unittest
import torch

from gcnet_missing_m3.core20_inference import (
    DEQReadout, NRIReadout, PrototypeClassifier, UniversalACTReadout, fixed_point,
)


def inputs():
    torch.manual_seed(2)
    local = torch.randn(4, 2, 8)
    base = torch.randn(4, 2, 12)
    gap = torch.randn(4, 2, 3, 12)
    availability = torch.ones(4, 2, 3)
    availability[1:, :, 0] = 0
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    return local, base, gap, availability, mask


def check_causal_mask_and_real_gradients(kind):
    model = kind(8, 12, 20).eval()
    args = inputs()
    output = model(*args)
    changed = [a.clone() for a in args]
    changed[0][3] = 999
    changed[1][3] = 999
    changed[2][3] = 999
    assert torch.allclose(output[:3], model(*changed)[:3], atol=1e-6)
    contaminated = [a.clone() for a in args]
    contaminated[1][0] = torch.nan  # no history before first utterance
    contaminated[2][0] = torch.nan
    contaminated[2][:, :, 1:] = torch.nan  # inactive observed gaps
    contaminated[0][2:, 1] = torch.nan
    contaminated[1][2:, 1] = torch.nan
    contaminated[2][2:, 1] = torch.nan
    assert torch.allclose(output, model(*contaminated), atol=1e-6)
    assert torch.equal(output[2:, 1], torch.zeros_like(output[2:, 1]))
    output.square().sum().backward()
    assert model.inject.weight.grad is not None
    assert torch.isfinite(model.inject.weight.grad).all()
    assert model.inject.weight.grad.abs().sum() > 0


def test_deq_implicit_gradient_matches_long_unroll():
    model = DEQReadout(8, 12, 20, tolerance=1e-8).double()
    args = tuple(a.double() for a in inputs())
    output = model(*args)
    output.square().sum().backward()
    actual = model.recurrent.grad.clone()
    from gcnet_missing_m3.core20_inference import _inputs
    model.zero_grad()
    prepared, valid = _inputs(*args)
    injected = model.inject(prepared[valid])
    state = torch.zeros_like(injected)
    for _ in range(100):
        state = model.cell(state, injected)
    model.output(state).square().sum().backward()
    assert torch.allclose(actual, model.recurrent.grad, atol=2e-6, rtol=2e-5)
    assert model.last_diagnostics["forward"]["converged"].all()
    assert model.last_diagnostics["backward"]["converged"].all()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "converge"):
        fixed_point(lambda x: x + 1, torch.zeros(2, 3), max_steps=3)


def test_act_halt_and_ponder_gradient():
    model = UniversalACTReadout(8, 12, 20, max_steps=4)
    with torch.no_grad():
        model.halt.weight.zero_()
        model.halt.bias.fill_(-10)
    output = model(*inputs())
    assert all(torch.equal(r["updates"], torch.full_like(r["updates"], 4))
               for r in model.last_diagnostics["act"])
    assert all(torch.allclose(r["halting_probability"], torch.ones_like(r["halting_probability"]))
               for r in model.last_diagnostics["act"])
    (output.square().mean() + model.ponder_cost).backward()
    assert model.halt.bias.grad.abs().sum() > 0
    with torch.no_grad():
        model.halt.bias.fill_(10)
    model(*inputs())
    assert all((r["updates"] == 1).all() for r in model.last_diagnostics["act"])


def test_nri_categorical_edges_temporal_loss_and_decoder_training():
    model = NRIReadout(8, 12, 20).train()
    output = model(*inputs())
    edges = model.last_diagnostics["edges"]
    assert edges and all(r["posterior"].shape[-1] == 3 for r in edges)
    assert all(torch.allclose(r["posterior"].sum(-1), torch.ones_like(r["posterior"][:, 0])) for r in edges)
    assert all((r["nodes"] <= r["time"]).all() for r in edges)
    assert model.last_diagnostics["prediction_loss"] > 0
    assert model.last_diagnostics["kl"] >= -1e-7
    (output.square().mean() + model.auxiliary_loss).backward()
    for parameter in (model.posterior[-1].weight, model.messages[0][0].weight,
                      model.predict.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_act_halted_attention_keys_are_frozen():
    class PerPositionHalt(torch.nn.Module):
        def forward(self, state):
            logits = state.new_full((*state.shape[:-1], 1), -10.)
            logits[:, 0] = 10.
            return logits
    model = UniversalACTReadout(8, 12, 20, max_steps=3).eval()
    model.halt = PerPositionHalt()
    keys, transformed = [], []
    model.attention.register_forward_pre_hook(lambda _, args: keys.append(args[1].detach().clone()))
    model.norm2.register_forward_hook(lambda _, args, output: transformed.append(output.detach().clone()))
    args = inputs()
    model(*(a[:, :1] if i < 4 else a[:1] for i, a in enumerate(args)))
    assert len(keys) == 3
    assert torch.equal(keys[1][:, 0], transformed[0][:, 0])
    assert torch.equal(keys[2][:, 0], transformed[0][:, 0])


def test_prototypes_train_projection_classification_and_regularization():
    torch.manual_seed(3)
    model = PrototypeClassifier(10, 6, 2, 2)
    samples, labels = torch.randn(8, 10), torch.arange(8) % 2
    logits, distances = model(samples)
    model.loss(logits, distances, labels).backward()
    assert model.prototype_vectors.grad.abs().sum() > 0
    assert model.features[0].weight.grad.abs().sum() > 0
    with unittest.TestCase().assertRaisesRegex(ValueError, "TRAIN"):
        model.project_train_exemplars([(samples, labels, list(range(8)))], split="test")
    records = model.project_train_exemplars([(samples, labels, list(range(8)))], split="train")
    encoded = model.features(samples).detach()
    for p, record in enumerate(records):
        row = int(record["exemplar_id"])
        assert labels[row] == model.prototype_class[p]
        assert torch.equal(model.prototype_vectors[p], encoded[row])
    assert model.projected.all()


class CoreInferenceTests(unittest.TestCase):
    def test_deq_mask(self):
        check_causal_mask_and_real_gradients(DEQReadout)

    def test_act_mask(self):
        check_causal_mask_and_real_gradients(UniversalACTReadout)

    def test_nri_mask(self):
        check_causal_mask_and_real_gradients(NRIReadout)

    test_deq_gradient = staticmethod(test_deq_implicit_gradient_matches_long_unroll)
    test_act = staticmethod(test_act_halt_and_ponder_gradient)
    test_act_frozen = staticmethod(test_act_halted_attention_keys_are_frozen)
    test_nri = staticmethod(test_nri_categorical_edges_temporal_loss_and_decoder_training)
    test_prototypes = staticmethod(test_prototypes_train_projection_classification_and_regularization)


if __name__ == "__main__":
    unittest.main()
