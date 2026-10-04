"""One shared contract; no new GPU experiments."""
import importlib.util
import unittest
from unittest.mock import patch

import torch


class PriorityFortyTests(unittest.TestCase):
    def test_mask_padding_and_updates_shared_contract(self):
        from gcnet_missing_m3.priority40_registry import PRIORITY_RESIDUAL_METHODS, PRIORITY_INPUT_METHODS
        from gcnet_missing_m3.meaningful_blocks import MeaningfulReadoutResidual
        from tests.test_meaningful_input import InputAdapterTests
        torch.set_num_threads(1)
        torch.manual_seed(66)
        local, base = torch.randn(3, 2, 256), torch.randn(3, 2, 1024)
        gap = torch.randn(3, 2, 3, 1024)
        av = torch.tensor([[[1, 0, 1], [1, 1, 1]]]*3).float()
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        anchor = torch.randn(3, 2, 1600)
        valid = mask.T.bool()
        poison = gap.masked_fill(av.bool()[..., None], float('nan'))
        poison[..., 512:] = float('inf')
        for name in PRIORITY_RESIDUAL_METHODS:
            with self.subTest(method=name):
                model = MeaningfulReadoutResidual(256, 1024, 1600, name, 8, 64)
                self.assertEqual(model(local, base, gap, av, mask, anchor).count_nonzero(), 0)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                for _ in range(3):
                    optimizer.zero_grad()
                    output = model(local, base, gap, av, mask, anchor)
                    (output-anchor).square().mean().backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), name)
                    optimizer.step()
                model.eval()
                clean = model(local, base, gap, av, mask, anchor)
                dirty = model(local, base, poison, av, mask, anchor)
                torch.testing.assert_close(clean, dirty, atol=0, rtol=0)
                self.assertEqual(clean[0].count_nonzero(), 0)
                self.assertEqual(clean[~valid].count_nonzero(), 0)
        with patch('gcnet_missing_m3.meaningful_input.INPUT_METHODS', PRIORITY_INPUT_METHODS):
            InputAdapterTests('test_shared_contract').test_shared_contract()

    def test_catalog_and_zero_start(self):
        self.assertIsNotNone(importlib.util.find_spec('gcnet_missing_m3.priority40_registry'))
        from gcnet_missing_m3.priority40_registry import PRIORITY_METHODS
        from tests.test_meaningful_block_integration import IntegrationTests
        self.assertEqual(len(PRIORITY_METHODS), 40)
        self.assertEqual(len(set(PRIORITY_METHODS)), 40)
        identity = tuple(name for name in PRIORITY_METHODS if name != 'm30_dyt')
        # M30 deliberately replaces LN; no claim of baseline function equality.
        with patch('gcnet_missing_m3.meaningful_blocks.MEANINGFUL_METHODS', identity):
            test = IntegrationTests('test_old_default_and_zero_bridge_full_model')
            test.setUp()
            test.test_old_default_and_zero_bridge_full_model()

    def test_dyt_is_only_first_adapter_normalization(self):
        from tests.test_meaningful_block_integration import config, _build_model
        from gcnet_missing_m3.priority40_common import DynamicTanh
        base = _build_model(config(), (3, 4, 5))
        changed = _build_model(config('m30_dyt'), (3, 4, 5))
        self.assertIsInstance(base.osram.emotion_adapter[0], torch.nn.LayerNorm)
        self.assertIsInstance(changed.osram.emotion_adapter[0], DynamicTanh)
        self.assertIsInstance(changed.osram.emotion_norm, torch.nn.LayerNorm)
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, changed.state_dict()[name]), name)
        self.assertEqual(list(changed.osram.meaningful_block.parameters()), [])
        layer = changed.osram.emotion_adapter[0]
        x = torch.randn(2, layer.weight.numel())
        torch.testing.assert_close(layer(x), layer.weight * torch.tanh(layer.alpha * x) + layer.bias)


if __name__ == '__main__':
    unittest.main()
