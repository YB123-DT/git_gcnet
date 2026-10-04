"""CPU-only small-shape contracts for the priority A/H mechanisms."""
import importlib
import importlib.util
import unittest

import torch

NAME = "gcnet_missing_m3.priority40_relations"
METHODS = ("m01_sab", "m02_dat", "m03_gatv2", "m04_edgeconv", "m05_pna",
           "m36_hopfield", "m37_entmax_sab", "m38_soft_moe", "m39_node", "m40_capsule")


class RelationsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.assertIsNotNone(importlib.util.find_spec(NAME), "priority relations missing")
        self.module = importlib.import_module(NAME)

    def test_all_contracts_masks_gradients_and_no_row_mixing(self):
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(29)
                model = self.module.build(method).eval()
                x = torch.randn(3, 5, 64)
                mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
                clean = torch.where(mask[..., None], x, 0.0)
                poisoned = torch.where(mask[..., None], x, float("nan")).requires_grad_()
                output = model.core(poisoned, mask)
                self.assertEqual(output.shape, (3, 64))
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output, model(clean, mask))
                self.assertEqual(torch.count_nonzero(output[-1]).item(), 0)
                torch.testing.assert_close(output[:1], model(clean[:1], mask[:1]))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(poisoned.grad).all())
                self.assertEqual(torch.count_nonzero(poisoned.grad[~mask]).item(), 0)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_node_initializes_once_only_in_training(self):
        node = self.module.build("m39_node").eval()
        x, mask = torch.randn(4, 5, 64), torch.ones(4, 5, dtype=torch.bool)
        node(x, mask)
        self.assertFalse(bool(node.initialized))
        before = node.threshold.detach().clone()
        node.train()
        node(x, torch.zeros_like(mask))
        self.assertFalse(bool(node.initialized))
        node(x, mask)
        self.assertTrue(bool(node.initialized))
        self.assertFalse(torch.equal(before, node.threshold))
        threshold, temperature = node.threshold.detach().clone(), node.log_temperature.detach().clone()
        node(x + 100, mask)
        torch.testing.assert_close(node.threshold, threshold)
        torch.testing.assert_close(node.log_temperature, temperature)

    def test_sab_and_entmax_have_identical_parameter_structure(self):
        soft = self.module.build("m01_sab")
        sparse = self.module.build("m37_entmax_sab")
        sparse.load_state_dict(soft.state_dict(), strict=True)
        scores = torch.tensor([[3.0, 0.0, -3.0], [9.0, 1.0, 5.0]])
        mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.bool)
        result = self.module.masked_entmax15(scores, mask)
        torch.testing.assert_close(result.sum(-1), torch.tensor([1.0, 0.0]))
        self.assertEqual(result[0, -1].item(), 0)

    def test_soft_moe_dual_axis_normalization(self):
        model = self.module.build("m38_soft_moe")
        x = torch.randn(2, 5, 64)
        mask = torch.tensor([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        dispatch, combine = model.routing(x, mask)
        torch.testing.assert_close(dispatch[0].sum(0), torch.ones(4))
        torch.testing.assert_close(combine[0].sum(-1), mask[0].float())
        self.assertEqual(torch.count_nonzero(dispatch[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(combine[1]).item(), 0)


if __name__ == "__main__":
    unittest.main()
