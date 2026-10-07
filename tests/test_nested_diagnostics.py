"""Classifier-input interventions must not leak deleted evidence through Nested."""
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


class NestedDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'experiments/osram_nested_diagnostics_20261007/evaluate.py'
        if not path.exists():
            cls.runner = None
            return
        spec = importlib.util.spec_from_file_location('nested_diagnostics', path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_runner_exists(self):
        self.assertIsNotNone(self.runner, 'new diagnostic runner not implemented')

    def test_deleted_evidence_is_not_a_nested_node(self):
        self.assertIsNotNone(self.runner)
        class Core(nn.Module):
            def forward(self, local, evidence, active, availability):
                self.active = active.clone()
                # Bias + cross-evidence exchange deliberately tries to leak.
                pooled = torch.where(active[..., None], evidence, 0).sum(-2)
                return local + pooled[..., :2], evidence + pooled[..., None, :] + 3
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.core = Core()
            def forward(self, local, base, gap, availability, umask):
                active = torch.cat((umask.T[..., None].bool(), ~availability.bool()), -1)
                evidence = torch.cat((base[..., None, :], gap), -2)
                l, c = self.core(local, evidence, active, availability)
                return l, c[..., 0, :], c[..., 1:, :]
        class Readout(nn.Module):
            def __init__(self):
                super().__init__()
                self.meaningful_block = Block()
                self.local_skip = nn.Linear(2, 2)
                self.emotion_adapter = nn.Linear(10, 2)
                self.emotion_norm = nn.Identity()
        o = Readout().eval()
        local, base, gap = torch.ones(2, 1, 2), torch.ones(2, 1, 2), torch.ones(2, 1, 3, 2)
        availability, umask = torch.tensor([[[1, 0, 0]], [[1, 0, 0]]]), torch.ones(1, 2)
        first = self.runner.replay_hidden(o, local, base, gap, availability, umask, 'gap_off')
        changed = self.runner.replay_hidden(o, local, base, gap * 1000, availability, umask, 'gap_off')
        torch.testing.assert_close(first, changed, rtol=0, atol=0)
        self.assertFalse(o.meaningful_block.core.active[..., 1:].any())
        original_forward = o.meaningful_block.core.forward.__func__
        self.runner.replay_hidden(o, local, base, gap, availability, umask, 'full')
        self.assertIs(original_forward, o.meaningful_block.core.forward.__func__)
        a = self.runner.replay_hidden(o, local, base, gap, availability, umask, 'no_local')
        b = self.runner.replay_hidden(o, local * 1000, base, gap, availability, umask, 'no_local')
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        p = self.runner.replay_hidden(o, local, base, gap, availability, torch.zeros_like(umask), 'full')
        self.assertEqual(torch.count_nonzero(p), 0)

    def test_undefined_cosines_are_nan(self):
        self.assertIsNotNone(self.runner)
        self.assertTrue(torch.isnan(self.runner.safe_cosine(torch.zeros(2), torch.ones(2))))

    def test_frozen_training_config_device_is_replaced(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig
        config = TrainConfig(device='cuda')
        updated = self.runner.config_for_device(config, 'cpu')
        self.assertEqual(updated.device, 'cpu')
        self.assertEqual(config.device, 'cuda')
        self.assertEqual(updated.seed, config.seed)

    def test_real_nested_excludes_ablated_heads(self):
        from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
        self.assertIsNotNone(self.runner)
        torch.set_num_threads(1)
        class Readout(nn.Module):
            def __init__(self):
                super().__init__()
                self.meaningful_block = MeaningfulInputAdapter(256, 1024, 1600,
                    'nested_gnn_rooted_evidence', 8, 64)
                self.local_skip = nn.Linear(256, 8)
                self.emotion_adapter = nn.Linear(4352, 8)
                self.emotion_norm = nn.LayerNorm(8)
        o = Readout().eval()
        # Exercise trained/nonzero decoder behavior, not vacuous zero-init.
        for p in o.meaningful_block.core.local_decoder.parameters():
            nn.init.constant_(p, .01)
        for decoder in o.meaningful_block.core.memory_decoders:
            for p in decoder.parameters():
                nn.init.constant_(p, .01)
        local, base, gap = torch.randn(3, 1, 256), torch.randn(3, 1, 1024), torch.randn(3, 1, 3, 1024)
        base[0] = 0
        gap[0] = 0
        av, umask = torch.tensor([[[1, 0, 0]]] * 3), torch.tensor([[1, 1, 0]])
        seen = []
        hook = o.meaningful_block.core.tokenizer.register_forward_hook(
            lambda module, inputs, output: seen.append(output[1].sum(-1).tolist()))
        try:
            self.runner.replay_hidden(o, local, base, gap, av, umask, 'gap_off')
            self.assertEqual(seen[-1], [9])
            a = self.runner.replay_hidden(o, local, base, gap, av, umask, 'base_off')
            self.assertEqual(seen[-1], [17])
            b = self.runner.replay_hidden(o, local, base * 100, gap, av, umask, 'base_off')
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            full = self.runner.replay_hidden(o, local, base, gap, av, umask, 'full')
            current = self.runner.replay_hidden(o, local, base, gap, av, umask, 'local_only')
            torch.testing.assert_close(full[0], current[0], rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(full[2]), 0)
        finally:
            hook.remove()


if __name__ == '__main__':
    unittest.main()
