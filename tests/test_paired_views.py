import unittest
import torch
import torch.nn.functional as F

from gcnet_missing_m3.paired_views import make_paired_view, symmetric_info_nce


class PairedViewTests(unittest.TestCase):
    def test_subset_padding_and_strict_prior(self):
        a = torch.ones(5, 3, 3)
        a[0, 0] = torch.tensor([1., 0., 0.])
        mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        a[4, 0] = float('nan')
        original = a.clone()
        b, contrast, stats = make_paired_view(a, mask, .5, torch.Generator().manual_seed(7))
        valid = mask.T.bool()
        self.assertTrue(torch.equal(a[valid], original[valid]))
        self.assertTrue((b[valid] <= a[valid]).all())
        self.assertTrue((b[valid].sum(-1) >= 1).all())
        self.assertTrue((b[~valid] == 0).all())
        changed = valid & (a != b).any(-1)
        prior = changed.long().cumsum(0) - changed.long() > 0
        expected = valid & (a == b).all(-1) & prior
        self.assertTrue(torch.equal(contrast, expected))
        self.assertFalse(contrast[0].any())
        self.assertEqual(stats['observed_dropped'], int((a[valid] - b[valid]).sum()))
        self.assertEqual(stats['contrast_anchors'], int(contrast.sum()))

    def test_extreme_probabilities_and_rng_isolation(self):
        a = torch.tensor([[[1., 1., 1.]], [[0., 1., 0.]], [[1., 0., 1.]]])
        mask = torch.ones(1, 3)
        state = torch.get_rng_state().clone()
        b, contrast, _ = make_paired_view(a, mask, 0., torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(b, a))
        self.assertFalse(contrast.any())
        b, contrast, _ = make_paired_view(a, mask, 1., torch.Generator().manual_seed(2))
        self.assertTrue((b.sum(-1) == 1).all())
        self.assertTrue(contrast[1, 0])
        again, _, _ = make_paired_view(a, mask, 1., torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(b, again))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_rejects_all_missing_valid_input(self):
        with self.assertRaises(ValueError):
            make_paired_view(torch.zeros(1, 1, 3), torch.ones(1, 1), .2, torch.Generator())

    def test_same_conversation_excluded_from_negatives(self):
        z1 = torch.tensor([[1., 0.], [1., 1.], [0., 1.]], requires_grad=True)
        z2 = torch.tensor([[1., 1.], [1., 0.], [0., 1.]], requires_grad=True)
        loss, count = symmetric_info_nce(z1, z2, ['repeat', 'repeat', 'other'], .1)
        logits = F.normalize(z1, dim=-1) @ F.normalize(z2, dim=-1).T / .1
        allowed = torch.tensor([[True, False, True], [False, True, True], [True, True, True]])
        logits = logits.masked_fill(~allowed, -float('inf'))
        expected = .5 * (F.cross_entropy(logits, torch.arange(3)) + F.cross_entropy(logits.T, torch.arange(3)))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(count, 3)
        loss.backward()
        self.assertTrue(torch.isfinite(z1.grad).all())
        self.assertTrue(torch.isfinite(z2.grad).all())

    def test_empty_and_single_conversation_have_differentiable_zero(self):
        for n in (0, 1, 3):
            a = torch.ones(n, 4, requires_grad=True)
            b = torch.ones(n, 4, requires_grad=True)
            loss, count = symmetric_info_nce(a, b, ['same'] * n)
            self.assertEqual(count, 0)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(a.grad).all())


if __name__ == '__main__':
    unittest.main()
