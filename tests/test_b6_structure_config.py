"""B6 structural knobs preserve legacy defaults and the Flat/CHM interface."""
import inspect
import unittest
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
from gcnet_missing_m3.meaningful_input_structured import _LearnedRobustPCA, build_structured


METHOD = 'learned_robust_pca_evidence_decomposition'


class LegacyPCA(nn.Module):
    """Frozen pre-change rank-one equations, independent of the new rank path."""
    def __init__(self):
        super().__init__()
        thresholds = torch.tensor([.1 * .8 ** t for t in range(1, 15)])
        self.threshold_logits = nn.Parameter(torch.log(torch.expm1(thresholds)))
        self.step_logits = nn.Parameter(torch.zeros(14))
        self.gamma = nn.Parameter(torch.zeros(()))

    @staticmethod
    def threshold(x, value):
        return x.sign() * F.relu(x.abs() - value)

    def decompose(self, y):
        scale = y.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        with torch.no_grad():
            initial = y - self.threshold(y, .1 * scale)
            left, singular, right_t = torch.linalg.svd(initial, full_matrices=False)
            root = singular[:, :1, None].sqrt()
            u = left[..., :1] * root
            v = right_t[:, :1].transpose(-2, -1) * root
        thresholds = F.softplus(self.threshold_logits)
        for coefficient, step in zip(thresholds, self.step_logits.sigmoid()):
            residual = y - u @ v.transpose(-2, -1)
            sparse = self.threshold(residual, coefficient * scale)
            error = residual - sparse
            next_u = u + step * (error @ v) / v.square().sum((-2, -1), keepdim=True).clamp_min(1e-6)
            next_v = v + step * (error.transpose(-2, -1) @ u) / u.square().sum((-2, -1), keepdim=True).clamp_min(1e-6)
            u, v = next_u, next_v
        return self.threshold(y - u @ v.transpose(-2, -1), thresholds[-1] * scale)


class B6StructureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(66)

    def test_new_knobs_registered_at_all_layers(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
        from gcnet_missing_m3.model import MissingM3GraphModel
        from gcnet_missing_m3.osram import OSRAMBackbone
        for cls in (TrainConfig, MissingM3GraphModel, OSRAMBackbone, MeaningfulInputAdapter):
            for name, default in (('osram_rpca_rank', 1), ('osram_rpca_steps', 14)):
                self.assertIn(name, inspect.signature(cls).parameters)
                self.assertEqual(inspect.signature(cls).parameters[name].default, default)
        args = build_parser().parse_args(['--audio-feature', 'a', '--text-feature', 't',
            '--video-feature', 'v', '--output-dir', 'unused', '--osram-rpca-rank', '3',
            '--osram-rpca-steps', '8'])
        self.assertEqual((args.osram_rpca_rank, args.osram_rpca_steps), (3, 8))

    def test_default_strict_checkpoint_forward_and_gradient_exact_legacy(self):
        old = LegacyPCA()
        new = _LearnedRobustPCA(rank=1, steps=14)
        with torch.no_grad():
            old.gamma.fill_(.37)
        new.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(set(new.state_dict()), set(old.state_dict()))
        self.assertEqual(sum(p.numel() for p in new.parameters()), 29)
        y1 = torch.randn(3, 12, 4, requires_grad=True)
        y2 = y1.detach().clone().requires_grad_()
        f1 = y1 - old.gamma.tanh() * old.decompose(y1)
        f2 = y2 - new.gamma.tanh() * new._decompose(y2)
        torch.testing.assert_close(f1, f2, atol=0, rtol=0)
        upstream = torch.randn_like(f1)
        (f1 * upstream).sum().backward()
        (f2 * upstream).sum().backward()
        torch.testing.assert_close(y1.grad, y2.grad, atol=0, rtol=0)
        for name, param in old.named_parameters():
            torch.testing.assert_close(param.grad, dict(new.named_parameters())[name].grad, atol=0, rtol=0)

    def test_factory_nondefault_rank_clamps_to_actual_matrix_and_preserves_rng(self):
        rng = torch.get_rng_state().clone()
        core = build_structured(METHOD, osram_rpca_rank=4, osram_rpca_steps=6)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual((core.rank, core.steps), (4, 6))
        self.assertEqual(sum(p.numel() for p in core.parameters()), 13)
        with torch.no_grad():
            core.gamma.fill_(.3)
        local = torch.randn(3, 8)
        evidence = torch.randn(3, 4, 2, requires_grad=True)
        active = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]).bool()
        _, output = core(local, evidence, active, torch.zeros(3, 3))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output[~active].count_nonzero(), 0)
        torch.testing.assert_close(output[2, 0], evidence[2, 0], atol=0, rtol=0)
        output.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in core.parameters()))

    def test_adapter_masks_padding_inactive_and_empty_history(self):
        block = MeaningfulInputAdapter(8, 1024, 12, METHOD, 8, 64,
            osram_rpca_rank=2, osram_rpca_steps=4)
        with torch.no_grad():
            block.core.gamma.fill_(.4)
        availability = torch.tensor([[[1., 0, 1], [0., 1, 0]]] * 3)
        umask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
        valid = umask.T.bool()
        local = torch.randn(3, 2, 8)
        base = torch.cat((torch.randn(3, 2, 512), torch.zeros(3, 2, 512)), -1)
        gap = torch.cat((torch.randn(3, 2, 3, 512), torch.zeros(3, 2, 3, 512)), -1)
        base[0] = 0.; gap[0] = 0.
        clean_gap = torch.where((valid[..., None] & ~availability.bool())[..., None], gap, 0.)
        clean = block(local, base, clean_gap, availability, umask)
        dirty_gap = clean_gap.masked_fill(availability.bool()[..., None], float('nan'))
        dirty_gap[~valid] = float('nan')
        dirty = block(local, base, dirty_gap, availability, umask)
        for a, b in zip(clean, dirty):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertEqual(a[~valid].count_nonzero(), 0)
        torch.testing.assert_close(clean[0][valid], local[valid], atol=0, rtol=0)
        self.assertEqual(clean[2][availability.bool()].count_nonzero(), 0)
        self.assertEqual(clean[1][0].count_nonzero(), 0)
        self.assertEqual(clean[2][0].count_nonzero(), 0)
        self.assertEqual(clean[1][..., 512:].count_nonzero(), 0)
        self.assertEqual(clean[2][..., 512:].count_nonzero(), 0)

    def test_config_and_module_reject_bool_fractional_and_out_of_range(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig
        cfg = TrainConfig()
        for name, values in (('osram_rpca_rank', (True, 0, 5, 1.5)),
                             ('osram_rpca_steps', (True, 0, 65, 3.5))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(cfg, **{name: value})
                with self.subTest(factory=name, value=value), self.assertRaises(ValueError):
                    build_structured(METHOD, **{name: value})

    def test_backbone_routes_nondefault_without_altering_flat_parameters(self):
        from gcnet_missing_m3.osram import OSRAMBackbone
        kwargs = dict(latent_dim=8, output_dim=12, num_heads=8, key_dim=64,
                      value_dim=64, n_speakers=2, bidirectional=False, dropout=0.)
        torch.manual_seed(66)
        baseline = OSRAMBackbone(**kwargs)
        torch.manual_seed(66)
        changed = OSRAMBackbone(**kwargs, osram_meaningful_block=METHOD,
                               osram_rpca_rank=3, osram_rpca_steps=8)
        self.assertEqual((changed.meaningful_block.core.rank, changed.meaningful_block.core.steps), (3, 8))
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, changed.state_dict()[name]), name)

    def test_shared_training_replay_constructor_routes_real_config(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig
        from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
        config = TrainConfig(dataset='MELD', device='cpu', backbone_type='osram',
            training_objective='emotion-only', disable_unused_aux_modules=True,
            osram_bidirectional=False, osram_num_heads=8, osram_key_dim=64,
            osram_value_dim=64, osram_output_dim=12, latent_dim=8,
            osram_meaningful_block=METHOD, osram_rpca_rank=3, osram_rpca_steps=6)
        model = _build_model(config, (3, 4, 5), torch.device('cpu'))
        self.assertEqual((model.osram.meaningful_block.core.rank,
                          model.osram.meaningful_block.core.steps), (3, 6))


if __name__ == '__main__':
    unittest.main()
