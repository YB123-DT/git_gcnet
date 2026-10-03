import unittest
import torch


class RelationOffTests(unittest.TestCase):
    def test_hook_zero_without_mutating_parameters_or_inputs(self):
        from experiments.osram_current_history_relation_20261003.residual_off import ResidualIntervention
        module = torch.nn.Linear(3, 2)
        module.last_diagnostics = {'relation_residual_norm': 1., 'relation_anchor_norm_ratio': .2}
        state = {k: v.clone() for k, v in module.state_dict().items()}
        x = torch.randn(5, 3)
        original = x.clone()
        on = ResidualIntervention(False)
        handle = module.register_forward_hook(on)
        expected = module(x)
        handle.remove()
        off = ResidualIntervention(True)
        handle = module.register_forward_hook(off)
        actual = module(x)
        handle.remove()
        self.assertEqual(actual.count_nonzero(), 0)
        self.assertTrue(torch.equal(x, original))
        self.assertTrue(torch.equal(module(x), expected))
        self.assertEqual(on.digest.hexdigest(), off.digest.hexdigest())
        self.assertEqual(off.calls, 1)
        for k, v in state.items(): self.assertTrue(torch.equal(v, module.state_dict()[k]))
