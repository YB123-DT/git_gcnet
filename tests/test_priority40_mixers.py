"""Minimal CPU checks; no training, CUDA, datasets or new dependencies."""
import unittest

import torch

from gcnet_missing_m3.priority40_mixers import METHODS, FNet, build


class PriorityMixersTest(unittest.TestCase):
    def test_contract_mask_backward_and_row_isolation(self):
        torch.manual_seed(41)
        active = torch.tensor([[True, True, False, True, False], [False] * 5, [True] * 5])
        x = torch.randn(3, 5, 64)
        clean = torch.where(active[..., None], x, torch.zeros_like(x))
        dirty = x.masked_fill(~active[..., None], float("nan"))
        for method in METHODS:
            with self.subTest(method=method):
                model = build(method).eval()
                reference = model(clean, active)
                output = model(dirty, active)
                self.assertEqual(output.shape, (3, 64))
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.equal(output[1], torch.zeros(64)))
                modified = dirty.clone()
                modified[2] *= 11
                torch.testing.assert_close(model(modified, active)[0], output[0], rtol=0, atol=0)
                output.square().sum().backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_mixer_spec_and_sgu_initialization(self):
        mixer = build(METHODS[0])
        self.assertEqual((mixer.token_mlp[0].in_features, mixer.token_mlp[0].out_features,
                          mixer.token_mlp[2].out_features), (5, 10, 5))
        self.assertEqual((mixer.channel_mlp[0].in_features, mixer.channel_mlp[0].out_features,
                          mixer.channel_mlp[2].out_features), (64, 128, 64))
        self.assertTrue(torch.equal(build(METHODS[1]).spatial.bias, torch.ones(5)))

    def test_fourier_real_2d_not_batch(self):
        x = torch.randn(2, 5, 64)
        expected = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        torch.testing.assert_close(FNet.fourier(x), expected)
        torch.testing.assert_close(FNet.fourier(x)[0], FNet.fourier(x[:1])[0])

    def test_grn_active_energy_and_zero_initialization(self):
        model = build(METHODS[3])
        active = torch.tensor([[True, False, True, False, False]])
        x = torch.randn(1, 5, 256)
        clean = torch.where(active[..., None], x, torch.zeros_like(x))
        dirty = x.masked_fill(~active[..., None], float("nan"))
        torch.testing.assert_close(model.response_norm(dirty, active), clean, rtol=0, atol=0)
        with torch.no_grad():
            model.gamma.fill_(1.)
        g = torch.linalg.vector_norm(clean, dim=1, keepdim=True)
        expected = clean + clean * g / (g.mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(model.response_norm(dirty, active), expected)

    def test_dynamic_weights_two_segments_source_axis(self):
        model = build(METHODS[4])
        active = torch.tensor([[True, False, True, True, False]])
        x = torch.randn(1, 5, 64)
        weights = model.mixing_weights(x, active)
        self.assertEqual(weights.shape, (1, 2, 5, 5))
        expected_sums = active[:, None, :].expand(-1, 2, -1).float()
        torch.testing.assert_close(weights.sum(-2), expected_sums)
        self.assertFalse(torch.equal(weights, model.mixing_weights(x * 2, active)))
        self.assertTrue(torch.equal(weights[:, :, ~active[0]], torch.zeros_like(weights[:, :, ~active[0]])))


if __name__ == "__main__":
    unittest.main()
