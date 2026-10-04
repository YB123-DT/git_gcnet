"""Small CPU contract checks; no GPU smoke or training run."""
import importlib.util
import unittest

import torch

NAME = "gcnet_missing_m3.meaningful_v3_routing"
METHODS = (
    "routing_predinet_bound_predicates",
    "routing_soft_moe_dispatch_expert_combine",
    "routing_scl_shared_compositional_maps",
    "routing_esbn_ephemeral_symbol_binding",
)


class RoutingTests(unittest.TestCase):
    def test_contract_and_active_core_gradients(self):
        self.assertIsNotNone(importlib.util.find_spec(NAME), "routing implementation missing")
        module = __import__(NAME, fromlist=["build"])
        torch.set_num_threads(1)
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(17)
                model = module.build(method, latent_dim=12, num_heads=2, value_dim=4)
                local = torch.randn(3, 12)
                evidence = torch.randn(3, 4, 8)
                active = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
                availability = torch.zeros(3, 3)
                clean = torch.where(active[..., None], evidence, 0.0)
                poison = torch.where(active[..., None], evidence, float("nan"))
                a, b = model(local, poison, active, availability)
                torch.testing.assert_close(a, local, rtol=0, atol=0)
                torch.testing.assert_close(b, clean, rtol=0, atol=0)
                # Open zero-init output heads to exercise the actual mechanism.
                with torch.no_grad():
                    for head in [model.local_out, *model.evidence_out]:
                        head.weight.normal_(std=0.02)
                local.requires_grad_()
                poison.requires_grad_()
                a, b = model(local, poison, active, availability)
                ca, cb = model(local, clean, active, availability)
                torch.testing.assert_close(a, ca)
                torch.testing.assert_close(b, cb)
                self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
                self.assertEqual(torch.count_nonzero(b[~active]).item(), 0)
                (a.square().mean() + b.square().mean()).backward()
                self.assertTrue(torch.isfinite(local.grad).all())
                self.assertTrue(torch.isfinite(poison.grad).all())
                self.assertEqual(torch.count_nonzero(poison.grad[~active]).item(), 0)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                                       if not n.startswith(("local_out", "evidence_out")) and p.grad is not None), 0)
                # Forward-local workspaces cannot leak between rows/calls.
                alone = model(local[:1], clean[:1], active[:1], availability[:1])
                torch.testing.assert_close(alone[0], ca[:1])
                torch.testing.assert_close(alone[1], cb[:1])


if __name__ == "__main__":
    unittest.main()
