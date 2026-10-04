"""Small CPU equation, gradient and authoritative-mask contracts for B group."""
import unittest

import torch

from gcnet_missing_m3.priority40_bilinear import METHODS, _normalize, build


class PriorityBilinearTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.active = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 0, 0],
                                    [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        self.tokens = torch.randn(4, 5, 64)

    def test_shapes_mask_nan_and_finite_gradients(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = build(method)
                clean = torch.where(self.active[..., None], self.tokens, 0)
                poisoned = torch.where(self.active[..., None], self.tokens, float('nan')).requires_grad_()
                output = model(poisoned, self.active)
                self.assertEqual(output.shape, (4, 64))
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output, model(clean, self.active))
                self.assertTrue((output[-1] == 0).all())
                if method not in METHODS[:2]:
                    self.assertTrue((output[2] == 0).all())
                output.square().sum().backward()
                self.assertTrue(torch.isfinite(poisoned.grad).all())
                self.assertTrue((poisoned.grad[~self.active] == 0).all())
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertEqual(model(self.tokens[:0], self.active[:0]).shape, (0, 64))

    def test_tfn_full_outer_and_neutral_role(self):
        model = build('m06_tfn')
        clean = torch.where(self.active[..., None], self.tokens, 0)
        roles = model.augmented_roles(clean, self.active)
        torch.testing.assert_close(roles[~self.active], torch.tensor([1., 0., 0., 0., 0.]).expand(
            (~self.active).sum(), -1))
        outer = torch.einsum('ni,nj,nk,nl,nm->nijklm', *roles.unbind(1)).flatten(1)
        self.assertEqual(outer.shape[-1], 3125)
        expected = model.output(outer)
        expected[-1] = 0
        torch.testing.assert_close(model(self.tokens, self.active), expected)

    def test_lmf_rank_four_exact_equation(self):
        model = build('m07_lmf')
        expected = torch.ones(4, 4, 64)
        for role, layer in enumerate(model.factors):
            projected = layer(self.tokens[:, role]).reshape(4, 4, 64)
            expected *= torch.where(self.active[:, role, None, None], projected, 1)
        expected = expected.sum(1)
        expected[-1] = 0
        torch.testing.assert_close(model(self.tokens, self.active), expected)

    def test_mfb_tucker_and_block_keep_ordered_concatenation(self):
        for method in METHODS[2:]:
            model = build(method)
            self.assertEqual(model.history.in_features, 256)
            full = torch.ones_like(self.active)
            swapped = self.tokens[:, [0, 2, 1, 3, 4]]
            self.assertFalse(torch.allclose(model(self.tokens, full), model(swapped, full)))
        model = build('m09_tucker')
        self.assertEqual(model.core.shape, (32, 32, 64))
        left, right = model.local(self.tokens[:, 0]), model.history(self.tokens[:, 1:].flatten(1))
        expected = (left[:, :, None, None] * model.core[None] * right[:, None, :, None]).sum((1, 2))
        torch.testing.assert_close(model(self.tokens, full), expected)
        block = build('m10_block')
        self.assertEqual(len(block.left_blocks), 4)
        self.assertEqual(block.left_blocks[0].out_features, 16 * 4)

    def test_normalization_zero_has_finite_gradient(self):
        value = torch.zeros(2, 64, requires_grad=True)
        _normalize(value).sum().backward()
        self.assertTrue(torch.isfinite(value.grad).all())


if __name__ == '__main__':
    unittest.main()
