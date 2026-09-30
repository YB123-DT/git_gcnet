import importlib
import unittest

import torch
from torch import nn


class TinyFlat(nn.Module):
    def __init__(self):
        super().__init__()
        self.osram = nn.Module()
        self.osram.osram_readout_fusion = 'flat'
        self.osram.latent_dim = 2
        self.osram.emotion_adapter = nn.Sequential(nn.Linear(6, 5), nn.ELU(), nn.Linear(5, 3))
        self.osram.local_skip = nn.Linear(2, 3)
        self.osram.emotion_norm = nn.LayerNorm(3)
        self.smax_fc = nn.Linear(3, 2)
        self.readout_type = 'shared'
        self.calls = 0

    def forward(self, x, mask):
        self.calls += 1
        h = self.osram.emotion_norm(self.osram.local_skip(x[..., :2]) + self.osram.emotion_adapter(x))
        h = torch.where(mask.T.bool()[..., None], h, 0.)
        return self.smax_fc(h), h, {}, None


class ReadoutPathsTests(unittest.TestCase):
    def helper(self):
        name = 'experiments.osram_readout_paths_20260930.intervention'
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        self.assertIsNotNone(spec, 'intervention helper must exist')
        return importlib.import_module(name)

    def test_thirteen_paired_replays_identity_skip_bias_padding_and_state(self):
        helper = self.helper()
        torch.manual_seed(3)
        model = TinyFlat().eval()
        x = torch.randn(3, 2, 6)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        state = {k: v.clone() for k, v in model.state_dict().items()}
        cache = helper.capture_flat_readout(model, lambda: model(x, mask), mask)
        predictions = helper.replay_predictions(model, cache)
        self.assertEqual(len(set(helper.SETTINGS)), 13)
        self.assertEqual(predictions.shape, (13, 3, 2, 2))
        self.assertEqual(model.calls, 1)
        for i, (alpha, beta, mu) in enumerate(helper.SETTINGS):
            value = torch.cat((alpha * x[..., :2], mu * x[..., 2:]), -1)
            hidden = model.osram.emotion_norm(beta * model.osram.local_skip(x[..., :2]) + model.osram.emotion_adapter(value))
            hidden = torch.where(mask.T.bool()[..., None], hidden, 0.)
            torch.testing.assert_close(predictions[i], model.smax_fc(hidden), rtol=0, atol=0)
        self.assertFalse(predictions.requires_grad)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_reject_training_and_adaptations_and_remove_hooks_on_failure(self):
        helper = self.helper()
        model = TinyFlat()
        mask = torch.ones(1, 2)
        with self.assertRaises(ValueError):
            helper.capture_flat_readout(model, lambda: None, mask)
        model.eval()
        model.osram.osram_local_evidence_gate = True
        with self.assertRaises(ValueError):
            helper.capture_flat_readout(model, lambda: None, mask)
        model.osram.osram_local_evidence_gate = False
        def fail():
            raise RuntimeError('forward failure')
        with self.assertRaises(RuntimeError):
            helper.capture_flat_readout(model, fail, mask)
        self.assertFalse(model.osram.local_skip._forward_hooks)
        self.assertFalse(model.osram.emotion_adapter._forward_pre_hooks)

    def test_real_osram_cache_preserves_masked_gap_and_identity(self):
        from tests.test_osram_post_grn import PostGRNTests
        helper = self.helper()
        fixture = PostGRNTests()
        model = TinyFlat()
        model.osram = fixture.backbone()
        model.smax_fc = nn.Linear(7, 1)
        model.eval()
        with torch.no_grad():
            model.osram.emotion_adapter[-1].weight.normal_()
        args = fixture.backbone_inputs()
        def forward():
            hidden, _ = model.osram(*args)
            return model.smax_fc(hidden)
        cache = helper.capture_flat_readout(model, forward, args[-1])
        context = cache.adapter_input[..., model.osram.latent_dim:]
        gap = context[..., model.osram.context_dim:].reshape(3, 2, 3, -1)
        self.assertEqual(gap[args[2].bool()].count_nonzero().item(), 0)
        head_strides = []
        hook = model.smax_fc.register_forward_pre_hook(lambda m, args: head_strides.append(args[0].is_contiguous()))
        try:
            predictions = helper.replay_predictions(model, cache)
        finally:
            hook.remove()
        self.assertTrue(all(head_strides), 'Replay must preserve original contiguous task-head input layout')
        torch.testing.assert_close(predictions[helper.IDENTITY_INDEX], cache.original_logits, rtol=0, atol=0)
        model.osram.osram_emotion_ablation = 'local-only'
        with self.assertRaises(ValueError):
            helper.capture_flat_readout(model, forward, args[-1])


if __name__ == '__main__':
    unittest.main()
