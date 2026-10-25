"""Focused tests for replacing only r_i in [x_i,r_i,mean(r)]."""
import unittest

import torch

try:
    from .intervention import common_root_input, suppress_root_specificity
except ModuleNotFoundError:
    common_root_input = suppress_root_specificity = None


class RootSpecificityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(common_root_input, 'Root-only intervention is not implemented')

    def test_only_middle_chunk_is_replaced(self):
        x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        roots = x.flip(1) * 2
        public = roots.mean(1, keepdim=True).expand_as(roots)
        inputs = torch.cat((x, roots, public), -1)
        saved = inputs.clone()
        output = common_root_input(inputs)
        self.assertTrue(torch.equal(output[..., :4], x))
        self.assertTrue(torch.equal(output[..., 4:8], public))
        self.assertTrue(torch.equal(output[..., 8:], public))
        self.assertTrue(torch.equal(inputs, saved))

    def test_equal_roots_are_exact_identity(self):
        x = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4)
        root = torch.ones_like(x)
        inputs = torch.cat((x, root, root), -1)
        self.assertTrue(torch.equal(common_root_input(inputs), inputs))
        with self.assertRaises(ValueError):
            common_root_input(torch.ones(2, 3, 11))

    def test_real_nested_hook_preserves_state_rng_and_original_path(self):
        self.assertIsNotNone(suppress_root_specificity)
        from gcnet_missing_m3.nested_sweep import NestedSweepGNN
        torch.manual_seed(6677)
        core = NestedSweepGNN(dim=4, depth=2).double().eval().requires_grad_(False)
        core.epsilon.fill_(0.2)
        x = torch.randn(2, 5, 4, dtype=torch.float64)
        columns = torch.tensor([0, 1, 2, 5, 6])  # Local, Base1/2, Gap-T1/2
        state = {k: v.clone() for k, v in core.state_dict().items()}
        captured = []
        h = core.readout.register_forward_pre_hook(lambda _m, args: captured.append(args[0].clone()))
        original = core(x, columns, 2)
        h.remove()
        expected = core.readout(common_root_input(captured[0]))
        rng = torch.get_rng_state().clone()
        with suppress_root_specificity(core) as stats:
            changed = core(x, columns, 2)
        self.assertTrue(torch.equal(changed, expected))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(torch.equal(core(x, columns, 2), original))
        self.assertEqual(stats['calls'], 1)
        self.assertGreater(stats['max_removed_root_difference'], 0)
        for key, value in core.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]))
        self.assertFalse(core.readout._forward_pre_hooks)

    def test_hook_is_removed_after_failure(self):
        core = torch.nn.Module()
        core.readout = torch.nn.Linear(12, 4)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with suppress_root_specificity(core):
                raise RuntimeError('intentional')
        self.assertFalse(core.readout._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()
