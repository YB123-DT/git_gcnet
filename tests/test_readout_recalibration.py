"""Current-utterance recalibration: masking, independence, and paper operators."""
import importlib.util
import math
import unittest

import torch
from torch import nn


METHODS = ('film', 'se', 'eca', 'cbam', 'gct', 'simam', 'sk')


class RecalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(311)

    def build(self, method, dim=16):
        self.assertIsNotNone(importlib.util.find_spec('gcnet_missing_m3.readout_candidates_recalibration'),
                             'recalibration implementation is missing')
        from gcnet_missing_m3.readout_candidates_recalibration import build_recalibration
        return build_recalibration(method, dim)

    def test_masked_nan_zero_rows_and_gradients(self):
        active = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        for method in METHODS:
            with self.subTest(method=method):
                module = self.build(method)
                clean = torch.randn(3, 5, 16)
                clean[~active] = 0
                expected = module(clean, active)
                poisoned = clean.clone()
                poisoned[~active] = float('nan')
                poisoned.requires_grad_()
                actual = module(poisoned, active)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.shape, (3, 16))
                self.assertEqual(actual[2].count_nonzero(), 0)
                actual.square().sum().backward()
                self.assertTrue(torch.isfinite(poisoned.grad).all())
                self.assertEqual(poisoned.grad[~active].count_nonzero(), 0)
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_no_batch_mixing_or_forward_randomness(self):
        x = torch.randn(3, 5, 16)
        active = torch.ones(3, 5, dtype=torch.bool)
        for method in METHODS:
            with self.subTest(method=method):
                module = self.build(method)
                self.assertFalse(any(isinstance(m, (nn.Dropout, nn.modules.batchnorm._BatchNorm))
                                     for m in module.modules()))
                state = torch.get_rng_state().clone()
                full = module.train()(x, active)
                torch.testing.assert_close(full, module.eval()(x, active), rtol=0, atol=0)
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                separate = torch.cat([module(x[i:i+1], active[i:i+1]) for i in range(3)])
                torch.testing.assert_close(full, separate, rtol=1e-5, atol=1e-6)
                zero = module(torch.zeros_like(x), active)
                self.assertTrue(torch.isfinite(zero).all())

    def test_film_affine_memory_only(self):
        module = self.build('film', 4)
        with torch.no_grad():
            for p in module.parameters():
                p.zero_()
            module.conditioner[-1].bias[:4] = 1
            module.conditioner[-1].bias[4:] = 3
        x = torch.arange(20.).reshape(1, 5, 4)
        active = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.bool)
        expected = (x[:, 0] + 2*x[:, 1] + 3 + 2*x[:, 3] + 3) / 3
        torch.testing.assert_close(module(x, active), expected)

    def test_se_bottleneck_excitation(self):
        module = self.build('se', 4)
        with torch.no_grad():
            module.excitation[0].weight.fill_(1)
            module.excitation[0].bias.zero_()
            module.excitation[2].weight.copy_(torch.arange(1., 5.).reshape(4, 1))
            module.excitation[2].bias.zero_()
        x = torch.full((1, 5, 4), .1)
        active = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(module(x, active), .1*torch.sigmoid(.4*torch.arange(1., 5.))[None])

    def test_eca_filter_is_along_features(self):
        module = self.build('eca', 16)
        with torch.no_grad():
            module.channel_filter.weight.zero_()
            module.channel_filter.weight[0, 0, module.kernel_size//2-1] = 1
        z = torch.arange(1., 17.)[None]
        x = z[:, None, :].expand(-1, 5, -1)
        active = torch.ones(1, 5, dtype=torch.bool)
        torch.testing.assert_close(module(x, active), z*torch.sigmoid(torch.arange(16.)[None]))

    def test_cbam_masked_max_and_two_stage_gate(self):
        module = self.build('cbam', 4)
        with torch.no_grad():
            module.excitation[0].weight.fill_(-1)
            module.excitation[0].bias.zero_()
            module.excitation[2].weight.fill_(.1)
            module.excitation[2].bias.zero_()
            module.slot_filter.weight.zero_()
        x = torch.zeros(1, 5, 4)
        x[:, 0] = -1
        x[:, 1] = -3
        active = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
        # Channel descriptors are mean=-2 and max=-1, not max=0 from absent slots.
        expected = torch.full((1, 4), -2.) * torch.sigmoid(torch.tensor(1.2)) * .5
        torch.testing.assert_close(module(x, active), expected)

    def test_gct_identity_and_normalized_l2_formula(self):
        module = self.build('gct', 4)
        x = torch.arange(1., 21.).reshape(1, 5, 4)
        active = torch.ones(1, 5, dtype=torch.bool)
        torch.testing.assert_close(module(x, active), x.mean(1), rtol=0, atol=0)
        with torch.no_grad():
            module.gamma.fill_(.4)
            module.beta.fill_(.1)
        embedding = (x.square().sum(1) + 1e-5).sqrt()
        scale = 1 + torch.tanh(.4*embedding/(embedding.square().mean(-1, keepdim=True)+1e-5).sqrt()+.1)
        torch.testing.assert_close(module(x, active), x.mean(1)*scale)

    def test_simam_unbiased_slot_variance_and_singleton(self):
        module = self.build('simam', 4)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 0)
        x = torch.zeros(1, 5, 4)
        x[:, 0] = 1
        x[:, 1] = 3
        active = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
        expected = torch.full((1, 4), 2.) * torch.sigmoid(torch.tensor(.5+1/(4*(2+1e-4))))
        torch.testing.assert_close(module(x, active), expected)
        active[:, 1] = False
        torch.testing.assert_close(module(x, active), x[:, 0], rtol=0, atol=0)

    def test_sk_softmax_selects_transformed_branches_per_channel(self):
        module = self.build('sk', 4)
        with torch.no_grad():
            for index, branch in enumerate(module.branches):
                branch.weight.copy_((index+1)*torch.eye(4))
                branch.bias.zero_()
            for selector in module.selectors:
                selector.weight.zero_()
                selector.bias.zero_()
            module.selectors[0].bias.fill_(math.log(3))
        x = torch.arange(1., 21.).reshape(1, 5, 4)
        active = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(module(x, active), 1.25*x[:, :2].mean(1))

    def test_factory_and_shape_guards(self):
        with self.assertRaises(ValueError):
            self.build('unknown')
        with self.assertRaises(ValueError):
            self.build('se', 0)
        module = self.build('se')
        with self.assertRaises(ValueError):
            module(torch.ones(2, 4, 16), torch.ones(2, 4, dtype=torch.bool))
        with self.assertRaises(ValueError):
            module(torch.ones(2, 5, 16), torch.ones(2, 5))


if __name__ == '__main__':
    unittest.main()
