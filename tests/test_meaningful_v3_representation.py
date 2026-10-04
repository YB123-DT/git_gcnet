"""CPU contracts and mechanism checks; no dataset, labels or GPU smoke."""
import unittest

import torch

from gcnet_missing_m3.meaningful_v3_representation import METHODS, MERA, MPS, build


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        self.local = torch.randn(3, 256)
        self.evidence = torch.randn(3, 4, 512)
        self.active = torch.tensor([[1, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
        self.availability = torch.tensor([[1., 0., 1.], [0., 0., 0.], [0., 0., 0.]])

    def test_identity_mask_and_finite_gradients(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = build(method)
                clean = torch.where(self.active[..., None], self.evidence, 0)
                local, evidence = model(self.local, self.evidence, self.active, self.availability)
                torch.testing.assert_close(local, self.local, rtol=0, atol=0)
                torch.testing.assert_close(evidence, clean, rtol=0, atol=0)
                # Open the zero-init residual before checking core gradients.
                for layer in [model.local_decoder, *model.evidence_decoders]:
                    torch.nn.init.normal_(layer.weight, std=.02)
                poisoned = torch.where(self.active[..., None], self.evidence, float('nan'))
                output = model(self.local, poisoned, self.active, self.availability)
                reference = model(self.local, clean, self.active, self.availability)
                for actual, expected in zip(output, reference):
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected)
                self.assertTrue((output[1][~self.active] == 0).all())
                torch.testing.assert_close(output[0][-1], self.local[-1])
                sum(value.square().mean() for value in output).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertTrue(any(p.grad.abs().sum() > 0 for p in model.core.parameters()))

    def test_mps_absent_roles_are_identity_bonds(self):
        core = MPS(16)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        output = core(torch.randn(2, 5, 16), mask, torch.zeros(2, 3))
        torch.testing.assert_close(output, (core.left @ core.right).expand(2, -1))
        a, b = core.cores[0, 0], core.cores[1, 0]
        self.assertGreater((a @ b - b @ a).abs().sum().item(), 0)

    def test_mera_unitarity_density_and_disentangler_gradients(self):
        core = MERA(16)
        gates = core.unitaries()
        torch.testing.assert_close(gates @ gates.mH,
                                   torch.eye(4, dtype=gates.dtype).expand_as(gates),
                                   rtol=1e-5, atol=1e-5)
        rho = core.density(torch.randn(3, 5, 16), torch.ones(3, 5, dtype=torch.bool),
                           self.availability)
        torch.testing.assert_close(rho, rho.mH, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(rho.diagonal(dim1=-2, dim2=-1).sum(-1),
                                   torch.ones(3, dtype=rho.dtype), rtol=1e-5, atol=1e-5)
        self.assertTrue((torch.linalg.eigvalsh(rho) > -1e-6).all())
        rho[:, 0, 0].real.sum().backward()
        # First three gates are genuine cross-pair boundary disentanglers.
        self.assertTrue((core.hermitian_real.grad[:3].abs().flatten(1).sum(-1) > 0).all())

    def test_grande_has_node_specific_not_depth_shared_selectors(self):
        core = build(METHODS[0]).core
        self.assertEqual(core.selectors.shape[1], 7)
        self.assertEqual(core.paths.shape, (8, 3))
        self.assertEqual(core.leaf_weights.shape, (8, 8))


if __name__ == '__main__':
    unittest.main()
