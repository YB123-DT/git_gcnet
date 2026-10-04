"""Bounded CPU tests for whole-role masking and the two defining mechanisms."""
import unittest

import torch

from gcnet_missing_m3.meaningful_v3_geometry import METHODS, build


class GeometryTests(unittest.TestCase):
    def test_contract_gradient_and_inference(self):
        torch.set_num_threads(1)
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(17)
                model = build(method, latent_dim=12, num_heads=2, value_dim=4)
                local = torch.randn(3, 12)
                evidence = torch.randn(3, 4, 8)
                active = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
                availability = torch.zeros(3, 3)
                clean = torch.where(active[..., None], evidence, 0.0)
                poisoned = torch.where(active[..., None], evidence, float("nan"))
                a, b = model(local, poisoned, active, availability)
                torch.testing.assert_close(a, local, atol=0, rtol=0)
                torch.testing.assert_close(b, clean, atol=0, rtol=0)
                with torch.no_grad():
                    for head in [model.local_out, *model.evidence_out]:
                        head.weight.normal_(std=0.02)
                local.requires_grad_()
                poisoned.requires_grad_()
                a, b = model(local, poisoned, active, availability)
                ca, cb = model(local, clean, active, availability)
                torch.testing.assert_close(a, ca)
                torch.testing.assert_close(b, cb)
                (a.square().mean() + b.square().mean()).backward()
                self.assertTrue(torch.isfinite(local.grad).all())
                self.assertTrue(torch.isfinite(poisoned.grad).all())
                self.assertEqual(torch.count_nonzero(poisoned.grad[~active]).item(), 0)
                key = model.comparator[0].weight if hasattr(model, "comparator") else model.templates
                self.assertTrue(torch.isfinite(key.grad).all())
                self.assertGreater(key.grad.abs().sum().item(), 0)
                before = key.detach().clone()
                torch.optim.SGD(model.parameters(), lr=0.01).step()
                self.assertFalse(torch.equal(before, key))
                model.eval()
                with torch.no_grad():
                    expected = model(local, poisoned, active, availability)
                with torch.inference_mode():
                    actual = model(local, poisoned, active, availability)
                for output, reference in zip(actual, expected):
                    self.assertTrue(torch.isfinite(output).all())
                    torch.testing.assert_close(output, reference)

    def test_po_is_whole_vector_permutation_invariant(self):
        torch.manual_seed(3)
        model = build("geometry_po_canonical_sequence", 12, 2, 4)
        x = torch.randn(2, 5, 8)
        torch.testing.assert_close(model.summarize(x), model.summarize(x[:, [4, 1, 3, 0, 2]]), atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(model.summarize(torch.zeros(2, 1, 8))).all())

    def test_repset_exact_capacity_score(self):
        model = build("geometry_repset_exact_template_matching", 12, 2, 4)
        with torch.no_grad():
            model.templates.zero_()
            model.templates[:, 0, 0] = 5
            model.templates[:, 1, 0] = 2
        # Both roles prefer support0; two-sided capacity permits it only once.
        x = torch.zeros(1, 2, 8)
        x[0, :, 0] = torch.tensor([2., 1.])
        torch.testing.assert_close(model.matching_scores(x), torch.full((1, 16), 6.))
        torch.testing.assert_close(model.matching_scores(x), model.matching_scores(x.flip(1)))


if __name__ == "__main__":
    unittest.main()
