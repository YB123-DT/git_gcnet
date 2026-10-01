import unittest
import torch
from tests.test_osram_post_grn import PostGRNTests


class MemoryOnlyAdapterTests(unittest.TestCase):
    def test_formula_and_local_dependency(self):
        helper = PostGRNTests()
        model = helper.backbone(osram_memory_only_adapter=True).eval()
        self.assertEqual(model.emotion_adapter[0].normalized_shape, (4 * model.context_dim,))
        self.assertEqual(model.emotion_adapter[1].in_features, 4 * model.context_dim)
        with torch.no_grad():
            model.local_skip.bias.fill_(2.)
            model.emotion_adapter[-1].weight.normal_(std=.01)
        captured = {}
        handles = [model.local_skip.register_forward_hook(lambda m,a,o: captured.update(skip=o)),
                   model.emotion_adapter.register_forward_pre_hook(lambda m,a: captured.update(adapter_input=a[0])),
                   model.emotion_adapter.register_forward_hook(lambda m,a,o: captured.update(adapter=o))]
        args = helper.backbone_inputs()
        output, _ = model(*args)
        expected = model.emotion_norm(captured['skip'] + captured['adapter'])
        expected = torch.where(args[4].T.bool()[..., None], expected, 0.)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        original_input = captured['adapter_input'].detach().clone()
        handle = model.local_path.register_forward_hook(lambda m,a,o: o + 3.)
        model(*args)
        torch.testing.assert_close(original_input, captured['adapter_input'], rtol=0, atol=0)
        handle.remove()
        for handle in handles: handle.remove()

    def test_default_and_gradients(self):
        helper = PostGRNTests()
        torch.manual_seed(19); old = helper.backbone()
        torch.manual_seed(19); off = helper.backbone(osram_memory_only_adapter=False)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, off.state_dict()[name], rtol=0, atol=0)
        args = helper.backbone_inputs()
        torch.manual_seed(7); expected = old(*args)[0]
        torch.manual_seed(7); actual = off(*args)[0]
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.manual_seed(19); model = helper.backbone(osram_memory_only_adapter=True)
        for name, value in old.state_dict().items():
            if not name.startswith('emotion_adapter.'):
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        before = model.local_skip.weight.detach().clone()
        for _ in range(3):
            opt.zero_grad(); model(*args)[0].square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            opt.step()
        self.assertFalse(torch.equal(before, model.local_skip.weight))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_safe_masks_and_readout_ablation(self):
        from unittest.mock import patch
        helper = PostGRNTests()
        args = helper.backbone_inputs()
        valid = args[4].T.bool()
        inactive = args[2].bool() | ~valid[..., None]
        for ablation in ('full', 'local-only', 'local-base', 'local-gap'):
            model = helper.backbone(osram_memory_only_adapter=True,
                osram_emotion_ablation=ablation).eval()
            with torch.no_grad(): model.emotion_adapter[-1].weight.normal_(std=.01)
            inputs = []
            handle = model.emotion_adapter.register_forward_pre_hook(lambda m,a: inputs.append(a[0].detach().clone()))
            expected, _ = model(*args)
            original_scan = model._scan
            def poisoned(*scan_args, **scan_kwargs):
                base, gap, diag = original_scan(*scan_args, **scan_kwargs)
                base = base.clone(); gap = gap.clone()
                base[~valid] = float('nan'); gap[inactive] = float('nan')
                return base, gap, diag
            with patch.object(model, '_scan', side_effect=poisoned): actual, _ = model(*args)
            handle.remove()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual[~valid].count_nonzero().item(), 0)
            adapter_input = inputs[-1]
            self.assertTrue(torch.isfinite(adapter_input).all())
            self.assertEqual(adapter_input[~valid].count_nonzero().item(), 0)
            gap_input = adapter_input[..., model.context_dim:].reshape(3,2,3,model.context_dim)
            self.assertEqual(gap_input[inactive].count_nonzero().item(), 0)
            if ablation in ('local-only', 'local-gap'):
                self.assertEqual(adapter_input[..., :model.context_dim].count_nonzero().item(), 0)
            if ablation in ('local-only', 'local-base'):
                self.assertEqual(gap_input.count_nonzero().item(), 0)

    def test_config_and_builder(self):
        from dataclasses import replace
        from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        config = TrainConfig(backbone_type='osram', training_objective='emotion-only',
            osram_bidirectional=False, latent_dim=8, osram_output_dim=19,
            osram_num_heads=2, osram_key_dim=3, osram_value_dim=4)
        self.assertFalse(config.osram_memory_only_adapter)
        args = build_parser().parse_args(['--audio-feature','a','--text-feature','t',
            '--video-feature','v','--output-dir','unused','--osram-memory-only-adapter'])
        self.assertTrue(args.osram_memory_only_adapter)
        enabled = replace(config, osram_memory_only_adapter=True)
        model = _build_model(enabled, (3,4,5))
        self.assertTrue(model.osram.osram_memory_only_adapter)
        for key, value in [('osram_local_skip_gate',True), ('osram_post_grn',True),
            ('osram_history_query_adapter',True), ('osram_history_input_gate',True),
            ('osram_local_evidence_gate',True), ('osram_hierarchical_evidence_gate',True),
            ('osram_bidirectional',True), ('osram_readout_fusion','history-innovation'),
            ('training_objective','joint'), ('train_rate_mode','conversation-mixed')]:
            with self.subTest(key=key), self.assertRaises(ValueError): replace(enabled, **{key:value})


if __name__ == '__main__': unittest.main()
