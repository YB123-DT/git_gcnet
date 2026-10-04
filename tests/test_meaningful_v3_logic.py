"""Bounded CPU checks of proof binding, shared-event WMC and adapter contracts."""
import importlib
import importlib.util
import unittest

import torch

NAME = "gcnet_missing_m3.meaningful_v3_logic"
METHODS = ("survey80_struct_neural_theorem_prover", "survey80_struct_deepproblog")


class LogicTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.assertIsNotNone(importlib.util.find_spec(NAME), "logic implementation missing")
        self.logic = importlib.import_module(NAME)

    def test_shared_variable_and_multihop_proofs(self):
        left = torch.zeros(1, 5, 5)
        right = torch.zeros_like(left)
        left[0, 0, 1], right[0, 2, 3] = 0.9, 0.8
        self.assertEqual(self.logic.join_body(left, right)[0, 0, 3].item(), 0)
        right[0, 1, 3] = 0.7
        torch.testing.assert_close(self.logic.join_body(left, right)[0, 0, 3], torch.tensor(0.7))
        facts = torch.zeros(1, 1, 5, 5)
        for i in range(4):
            facts[0, 0, i, i + 1] = 0.9 - 0.1 * i
        mask = torch.ones(1, 5, dtype=torch.bool)
        depth1 = self.logic.bounded_proofs(facts, torch.ones(1, 1), ((0, 0),), mask, 1)
        depth2 = self.logic.bounded_proofs(facts, torch.ones(1, 1), ((0, 0),), mask, 2)
        self.assertEqual(depth1[0, 0, 0, 4].item(), 0)
        torch.testing.assert_close(depth2[0, 0, 0, 4], torch.tensor(0.6))

    def test_exact_wmc_does_not_independently_count_overlapping_proofs(self):
        worlds = self.logic.enumerate_worlds(3)
        query = (worlds[:, 0] & worlds[:, 1]) | (worlds[:, 0] & worlds[:, 2])
        p = torch.tensor([[0.3, 0.4, 0.7]], requires_grad=True)
        result = self.logic.exact_wmc(p, worlds, query[:, None]).squeeze()
        expected = p[0, 0] * (p[0, 1] + p[0, 2] - p[0, 1] * p[0, 2])
        torch.testing.assert_close(result, expected)
        wrong = 1 - (1 - p[0, 0] * p[0, 1]) * (1 - p[0, 0] * p[0, 2])
        self.assertGreater(abs((result - wrong).item()), 0.01)
        actual_grad = torch.autograd.grad(result, p, retain_graph=True)[0]
        torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, p)[0])

    def test_identity_nan_mask_backward_and_row_independence(self):
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(19)
                model = self.logic.build(method, 12, 2, 4)
                local = torch.randn(3, 12)
                evidence = torch.randn(3, 4, 8)
                active = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
                availability = torch.zeros(3, 3)
                clean = torch.where(active[..., None], evidence, 0.0)
                poisoned = torch.where(active[..., None], evidence, float("nan"))
                a, b = model(local, poisoned, active, availability)
                torch.testing.assert_close(a, local, rtol=0, atol=0)
                torch.testing.assert_close(b, clean, rtol=0, atol=0)
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
                self.assertEqual(torch.count_nonzero(b[~active]).item(), 0)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                                       if not n.startswith(("local_out", "evidence_out")) and p.grad is not None), 0)
                one = model(local[:1], clean[:1], active[:1], availability[:1])
                torch.testing.assert_close(one[0], ca[:1])
                torch.testing.assert_close(one[1], cb[:1])


if __name__ == "__main__":
    unittest.main()
