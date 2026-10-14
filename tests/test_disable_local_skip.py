import unittest

import torch

from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
from tests.test_osram_post_grn import PostGRNTests


class DisableLocalSkipTests(unittest.TestCase):
    def test_default_is_exact_legacy_path(self):
        helper = PostGRNTests()
        torch.manual_seed(91)
        legacy = helper.backbone().eval()
        torch.manual_seed(91)
        explicit = helper.backbone(osram_disable_local_skip=False).eval()
        args = helper.backbone_inputs()
        with torch.no_grad():
            expected, expected_context = legacy(*args)
            actual, actual_context = explicit(*args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key in expected_context:
            torch.testing.assert_close(actual_context[key], expected_context[key], rtol=0, atol=0)

    def test_only_final_skip_is_removed_and_adapter_local_remains(self):
        helper = PostGRNTests()
        model = helper.backbone(osram_disable_local_skip=True).train()
        captured = {}
        adapter_handle = model.emotion_adapter.register_forward_pre_hook(
            lambda module, args: captured.update(adapter_input=args[0]))
        skip_handle = model.local_skip.register_forward_hook(
            lambda module, args, output: captured.update(skip_called=True))
        args = helper.backbone_inputs()
        output, _ = model(*args)
        adapter_handle.remove()
        skip_handle.remove()
        self.assertNotIn('skip_called', captured)
        self.assertEqual(captured['adapter_input'].shape[-1], model.latent_dim + 4 * model.context_dim)
        self.assertGreater(captured['adapter_input'][..., :model.latent_dim].abs().sum().item(), 0)
        output.square().mean().backward()
        self.assertIsNone(model.local_skip.weight.grad)
        self.assertIsNotNone(model.emotion_adapter[-1].weight.grad)
        self.assertTrue(torch.isfinite(model.emotion_adapter[-1].weight.grad).all())

    def test_cli_and_config_default(self):
        self.assertFalse(TrainConfig().osram_disable_local_skip)
        args = build_parser().parse_args([
            '--audio-feature', 'a', '--text-feature', 't', '--video-feature', 'v',
            '--output-dir', 'unused', '--osram-disable-local-skip',
        ])
        self.assertTrue(args.osram_disable_local_skip)


if __name__ == '__main__':
    unittest.main()
