"""Shared masking/initialization contract for the twenty adapted mechanisms."""
import unittest

import torch

from gcnet_missing_m3.readout_candidates import CANDIDATE_METHODS, ExternalReadoutResidual


class CandidateWrapperTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(17)
        local = torch.randn(4, 2, 12)
        base = torch.randn(4, 2, 32)
        gap = torch.randn(4, 2, 3, 32)
        available = torch.tensor([[[1, 0, 1], [1, 1, 1]]] * 4).float()
        umask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]])
        anchor = torch.randn(4, 2, 20)
        return local, base, gap, available, umask, anchor

    def test_twenty_unique_methods(self):
        self.assertEqual(len(CANDIDATE_METHODS), 20)
        self.assertEqual(len(set(CANDIDATE_METHODS)), 20)
        with self.assertRaises(ValueError):
            ExternalReadoutResidual(12, 32, 20, 'invented')

    def test_zero_initialization_and_no_forward_rng_consumption(self):
        for name in CANDIDATE_METHODS:
            with self.subTest(name=name):
                model = ExternalReadoutResidual(12, 32, 20, name)
                inputs = self.inputs()
                before = torch.get_rng_state().clone()
                output = model(*inputs)
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))
                self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_learned_masks_first_valid_forward_half_and_nan_safety(self):
        for name in CANDIDATE_METHODS:
            with self.subTest(name=name):
                model = ExternalReadoutResidual(12, 32, 20, name).eval()
                torch.nn.init.normal_(model.output.weight, std=.01)
                torch.nn.init.constant_(model.output.bias, .2)
                inputs = list(self.inputs())
                expected = model(*inputs)
                valid = inputs[4].T.bool()
                first = valid & (valid.long().cumsum(0) == 1)
                self.assertEqual(expected[~valid | first].count_nonzero().item(), 0)
                inputs[0] = inputs[0].clone().masked_fill(~valid[..., None], float('nan'))
                inputs[1] = inputs[1].clone()
                inputs[1][..., 16:] = float('nan')
                inputs[1].masked_fill_(~valid[..., None], float('nan'))
                inactive = ~valid[..., None] | inputs[3].bool()
                inputs[2] = inputs[2].clone().masked_fill(inactive[..., None], float('nan'))
                inputs[2][..., 16:] = float('nan')
                actual = model(*inputs)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_two_updates_reach_internal_parameters(self):
        for name in CANDIDATE_METHODS:
            with self.subTest(name=name):
                model = ExternalReadoutResidual(12, 32, 20, name)
                opt = torch.optim.Adam(model.parameters(), lr=.001)
                inputs = self.inputs()
                for _ in range(3):
                    opt.zero_grad()
                    output = model(*inputs)
                    loss = (output - .7).square().mean()
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    opt.step()
                internal = [p.grad for n, p in model.named_parameters()
                            if not n.startswith('output.') and p.grad is not None]
                self.assertTrue(any(g.count_nonzero() for g in internal))
                self.assertGreater(model.output.weight.abs().sum().item(), 0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
