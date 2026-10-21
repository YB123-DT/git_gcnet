"""CPU contracts for source relation attention, independent of the graph model."""

import importlib
import io
from pathlib import Path
import unittest

import torch
from torch import nn


class SourceRelationAttentionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'gcnet_missing_m3/source_relation.py'
        self.assertTrue(path.exists(), 'SourceRelationAttention is not implemented')
        self.cls = importlib.import_module('gcnet_missing_m3.source_relation').SourceRelationAttention
        torch.manual_seed(31)

    def inputs(self, t=3, b=2):
        trajectory = torch.ones(t, b, 3, 4)
        trajectory[..., 0] = 3.
        trajectory[..., 1] = .6
        trajectory[..., 2] = 2.
        return [torch.randn(t, b, 5), torch.randn(t, b, 3, 7),
                torch.randn(t, b, 3, 3, 7), trajectory,
                torch.zeros(t, b, 3), torch.ones(b, t)]

    def model(self):
        return self.cls(local_dim=5, forward_dim=7, output_dim=11)

    def activate(self, model):
        with torch.no_grad():
            model.output_projection.weight.normal_(std=.08)
            model.output_projection.bias.fill_(.3)

    def test_default_shape_and_exact_zero_initialization(self):
        model = self.cls()
        args = [torch.randn(2, 1, 256), torch.randn(2, 1, 3, 512),
                torch.randn(2, 1, 3, 3, 512), torch.ones(2, 1, 3, 4),
                torch.zeros(2, 1, 3), torch.ones(1, 2)]
        residual = model(*args)
        self.assertEqual(residual.shape, (2, 1, 1600))
        self.assertEqual(torch.count_nonzero(residual).item(), 0)
        self.assertEqual(torch.count_nonzero(model.output_projection.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(model.output_projection.bias).item(), 0)

    def test_architecture_uses_four_shared_role_projections_and_plain_attention(self):
        model = self.model()
        projections = [model.base_projection, *model.gap_projections]
        self.assertEqual(len(projections), 4)
        self.assertEqual(len({id(p.weight) for p in projections}), 4)
        for projection in projections:
            self.assertEqual(projection.in_features, 7)
            self.assertEqual(projection.out_features, 64)
            self.assertIsNone(projection.bias)
        self.assertEqual([type(layer) for layer in model.relation_mlp],
                         [nn.LayerNorm, nn.Linear, nn.GELU, nn.Linear])
        self.assertEqual(model.relation_mlp[0].normalized_shape, (263,))
        self.assertEqual(model.source_embedding.shape, (3, 128))
        for attention in (model.source_self_attention, model.local_cross_attention):
            self.assertEqual(attention.embed_dim, 128)
            self.assertEqual(attention.num_heads, 4)
            self.assertEqual(attention.dropout, 0.)
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in model.modules()))

    def test_gap_mean_preserves_role_projections_and_metadata_layout(self):
        model = self.model()
        args = self.inputs(t=1, b=1)
        args[4][:] = torch.tensor([1., 0., 0.])
        args[2][:, :, 0] = float('nan')
        with torch.no_grad():
            model.base_projection.weight.fill_(.1)
            for index, projection in enumerate(model.gap_projections):
                projection.weight.fill_(index + 1.)
        captured = []
        handle = model.relation_mlp.register_forward_pre_hook(
            lambda _, values: captured.append(values[0].detach().clone()))
        model(*args)
        handle.remove()
        features = captured[0].reshape(1, 1, 3, 263)
        base = model.base_projection(args[1])
        gap = (model.gap_projections[1](args[2][:, :, 1]) +
               model.gap_projections[2](args[2][:, :, 2])) / 2
        torch.testing.assert_close(features[..., :64], base)
        torch.testing.assert_close(features[..., 64:128], gap)
        torch.testing.assert_close(features[..., 128:192], base * gap)
        torch.testing.assert_close(features[..., 192:256], (base - gap).abs())
        torch.testing.assert_close(features[..., 256:259],
                                   args[4].unsqueeze(2).expand(1, 1, 3, 3))
        expected_trajectory = args[3].clone()
        expected_trajectory[..., 0] = expected_trajectory[..., 0].log1p()
        expected_trajectory[..., 2] = expected_trajectory[..., 2].log1p()
        torch.testing.assert_close(features[..., 259:], expected_trajectory)

    def test_cross_attention_keys_and_values_are_self_attention_output(self):
        model = self.model()
        self_outputs, cross_inputs = [], []
        first = model.source_self_attention.register_forward_hook(
            lambda _, values, result: self_outputs.append(result[0].detach().clone()))
        second = model.local_cross_attention.register_forward_pre_hook(
            lambda _, values: cross_inputs.append(tuple(v.detach().clone() for v in values[:3])))
        model(*self.inputs())
        first.remove()
        second.remove()
        query, key, value = cross_inputs[0]
        self.assertEqual(query.shape[-2:], (1, 128))
        torch.testing.assert_close(key, self_outputs[0], rtol=0, atol=0)
        torch.testing.assert_close(value, self_outputs[0], rtol=0, atol=0)

    def test_no_active_gap_has_exact_zero_gap_features(self):
        model = self.model()
        self.activate(model)
        args = self.inputs()
        args[4].fill_(1.)
        args[2].fill_(float('nan'))
        captured = []
        handle = model.relation_mlp.register_forward_pre_hook(
            lambda _, values: captured.append(values[0].detach().clone()))
        residual = model(*args)
        handle.remove()
        self.assertEqual(torch.count_nonzero(captured[0][..., 64:128]).item(), 0)
        self.assertTrue(torch.isfinite(residual).all())
        self.assertGreater(residual.abs().sum().item(), 0)

    def test_no_history_and_padding_stay_zero_after_output_bias(self):
        model = self.model()
        self.activate(model)
        args = self.inputs()
        args[3][0, :, :, 3] = 0
        args[5][1, 2] = 0
        residual = model(*args)
        self.assertEqual(torch.count_nonzero(residual[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(residual[2, 1]).item(), 0)
        self.assertGreater(residual[1].abs().sum().item(), 0)
        args[3][..., 3] = 0
        for value in args[:3]:
            value.fill_(float('nan'))
        residual = model(*args)
        self.assertTrue(torch.isfinite(residual).all())
        self.assertEqual(torch.count_nonzero(residual).item(), 0)
        self.assertEqual(torch.count_nonzero(
            model.last_diagnostics['cross_attention_weights']).item(), 0)

    def test_masked_nans_cannot_change_outputs_or_poison_gradients(self):
        model = self.model()
        self.activate(model)
        args = self.inputs()
        args[3][0, :, :, 3] = 0
        args[3][:, :, 1, 3] = 0
        args[4][:, :, 0] = 1
        args[5][1, 2] = 0
        expected = model(*args).detach()
        args[2][:, :, 0] = float('nan')
        args[1][:, :, 1] = float('nan')
        args[2][:, :, :, 1] = float('nan')
        args[3][:, :, 1, :3] = float('nan')
        for value in args[:3]:
            value[0] = float('nan')
        args[3][0, :, :, :3] = float('nan')
        for value in args[:5]:
            value[2, 1] = float('nan')
        for value in args[:4]:
            value.requires_grad_()
        residual = model(*args)
        torch.testing.assert_close(residual, expected, rtol=0, atol=0)
        residual.square().sum().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for value in args[:4]:
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_unseen_identity_tokens_are_zero_and_attention_weights_are_masked(self):
        model = self.model()
        self.activate(model)
        args = self.inputs()
        args[3][:, :, 1, 3] = 0
        captured = []
        handle = model.source_self_attention.register_forward_pre_hook(
            lambda _, values: captured.append(values[0].detach().clone()))
        expected = model(*args).detach()
        handle.remove()
        self.assertEqual(torch.count_nonzero(captured[0][:, 1]).item(), 0)
        weights = model.last_diagnostics['cross_attention_weights']
        self.assertEqual(weights.shape, (3, 2, 3))
        self.assertEqual(torch.count_nonzero(weights[..., 1]).item(), 0)
        torch.testing.assert_close(weights.sum(-1), torch.ones(3, 2))
        with torch.no_grad():
            model.source_embedding[1].fill_(1.e6)
        torch.testing.assert_close(model(*args), expected, rtol=0, atol=0)

    def test_zero_init_learns_output_then_upstream_parameters_over_three_steps(self):
        model = self.model()
        args = self.inputs()
        target = torch.randn(3, 2, 11)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        upstream = [model.base_projection.weight,
                    *(p.weight for p in model.gap_projections),
                    model.local_projection.weight,
                    model.source_self_attention.in_proj_weight,
                    model.local_cross_attention.in_proj_weight,
                    model.source_embedding]
        for step in range(3):
            optimizer.zero_grad()
            residual = model(*args)
            (residual - target).square().mean().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for parameter in upstream:
                if step == 0:
                    self.assertEqual(parameter.grad.abs().sum().item(), 0)
                else:
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertGreater(model.output_projection.weight.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_state_dict_reload_preserves_exact_outputs(self):
        model = self.model().eval()
        self.activate(model)
        args = self.inputs()
        expected = model(*args)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        reloaded = self.model().eval()
        reloaded.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(reloaded(*args), expected, rtol=0, atol=0)

    def test_diagnostics_are_finite_detached_and_descriptive(self):
        model = self.model()
        model(*self.inputs())
        self.assertEqual(set(model.last_diagnostics),
                         {'residual_norm', 'source_token_norm', 'active_source_mean',
                          'cross_attention_weights'})
        for value in model.last_diagnostics.values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
        self.assertEqual(model.last_diagnostics['active_source_mean'].item(), 3.)

    def test_invalid_dimensions_and_input_shapes_fail_clearly(self):
        for kwargs in ({'local_dim': 0}, {'forward_dim': -1}, {'output_dim': 0},
                       {'local_dim': True}, {'forward_dim': 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.cls(**kwargs)
        model = self.model()
        for index, shape in enumerate([(3, 2, 6), (3, 2, 4, 7),
                                       (3, 2, 3, 7), (3, 2, 3, 3),
                                       (3, 2, 4), (2, 4)]):
            args = self.inputs()
            args[index] = torch.zeros(shape)
            with self.subTest(index=index), self.assertRaises(ValueError):
                model(*args)


if __name__ == '__main__':
    unittest.main()
