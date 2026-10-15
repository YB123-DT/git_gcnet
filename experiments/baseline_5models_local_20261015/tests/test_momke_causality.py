"""CPU-only causality and architecture compatibility checks; run with unittest."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

# Upstream tracks .pyc files; validation must not dirty that checkout.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.momke import build_model, stage_loss


class MoMKECausalityTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(66)
        self.config = dict(hidden=12, depth=2, num_heads=2, drop_rate=0., attn_drop_rate=0., causal=True)
        self.model = build_model(self.config, (3, 4, 5), 7, 9).eval()
        self.features = [torch.randn(2, 7, d) for d in (3, 4, 5)]
        self.mask = torch.tensor([[[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0]]] * 2).bool()
        self.valid = torch.ones(2, 7, dtype=torch.bool)

    def output(self, features=None, mask=None, valid=None, first_stage=False, model=None):
        return (model or self.model)(*(features or self.features),
                                    availability=self.mask if mask is None else mask,
                                    padding=self.valid if valid is None else valid,
                                    first_stage=first_stage)

    def check_close(self, left, right, first_stage, prefix=3):
        keys = ['logits']
        if first_stage:
            keys.append('auxiliary_logits')
        for key in keys:
            lhs, rhs = left[key], right[key]
            if key == 'auxiliary_logits':
                for a, b in zip(lhs, rhs):
                    torch.testing.assert_close(a[:, :prefix], b[:, :prefix], atol=2e-6, rtol=2e-5)
            else:
                torch.testing.assert_close(lhs[:, :prefix], rhs[:, :prefix], atol=2e-6, rtol=2e-5)

    def test_future_features_both_stages(self):
        for stage in (True, False):
            with self.subTest(first_stage=stage):
                altered = [x.clone() for x in self.features]
                for x in altered:
                    x[:, 3:] = torch.randn_like(x[:, 3:]) * 100
                self.check_close(self.output(first_stage=stage), self.output(altered, first_stage=stage), stage)

    def test_future_missing_masks_both_stages(self):
        for stage in (True, False):
            changed = self.mask.clone()
            changed[:, 3:] = ~changed[:, 3:]
            changed[:, 3:, 0] = True
            self.check_close(self.output(first_stage=stage), self.output(mask=changed, first_stage=stage), stage)

    def test_prefix_equivalence_both_stages(self):
        for stage in (True, False):
            prefix = self.output([x[:, :3] for x in self.features], self.mask[:, :3], self.valid[:, :3], stage)
            self.check_close(self.output(first_stage=stage), prefix, stage)

    def test_padding_invariance_both_stages(self):
        for stage in (True, False):
            features = [torch.cat((x, torch.randn(2, 4, x.size(-1)) * 100), dim=1) for x in self.features]
            mask = torch.cat((self.mask, torch.ones(2, 4, 3, dtype=torch.bool)), dim=1)
            valid = torch.cat((self.valid, torch.zeros(2, 4, dtype=torch.bool)), dim=1)
            self.check_close(self.output(first_stage=stage), self.output(features, mask, valid, stage), stage, prefix=7)

    def test_batch_independence_both_stages(self):
        for stage in (True, False):
            full = self.output(first_stage=stage)
            single = self.output([x[:1] for x in self.features], self.mask[:1], self.valid[:1], stage)
            torch.testing.assert_close(full['logits'][:1], single['logits'], atol=2e-6, rtol=2e-5)
            perturbed = [x.clone() for x in self.features]
            for x in perturbed:
                x[1] *= 100
            second = self.output(perturbed, first_stage=stage)
            torch.testing.assert_close(full['logits'][:1], second['logits'][:1], atol=2e-6, rtol=2e-5)

    def test_missing_feature_invariance_both_stages(self):
        for stage in (True, False):
            altered = [x.clone() for x in self.features]
            for i, x in enumerate(altered):
                x[~self.mask[:, :, i]] = 10000
            self.check_close(self.output(first_stage=stage), self.output(altered, first_stage=stage), stage, prefix=7)

    def test_finite_forward_backward_update_in_both_stages(self):
        # A whole modality absent and padded examples exercise all-masked softmax.
        mask = self.mask.clone()
        mask[:, :, 2] = False
        mask[:, :, 0] = True
        valid = self.valid.clone()
        valid[1, 4:] = False
        labels = torch.randint(0, 7, (2, 7))
        labels[~valid] = -100
        for stage in (True, False):
            self.model.train()
            before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            out = self.output(mask=mask, valid=valid, first_stage=stage)
            loss = stage_loss(out, labels, valid, stage)
            self.assertTrue(torch.isfinite(out['logits']).all())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [(n, p) for n, p in self.model.named_parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(p.grad).all() for _, p in gradients))
            optimizer.step()
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in gradients))
            if stage:
                self.assertIsNone(self.model.nlp_head.weight.grad)
                self.assertIsNone(self.model.router_a.fc1.weight.grad)
            else:
                self.assertIsNotNone(self.model.nlp_head.weight.grad)
                self.assertIsNotNone(self.model.router_a.fc1.weight.grad)
                self.assertIsNotNone(self.model.block.blocks[0].Transformer_a.q.weight.grad)
            self.model.eval()

    def test_original_is_noncausal_negative_control(self):
        original = build_model(dict(self.config, causal=False), (3, 4, 5), 7, 9).eval()
        original.load_state_dict(self.model.state_dict(), strict=True)
        changed = [x.clone() for x in self.features]
        for x in changed:
            x[:, 3:] *= 50
        for stage in (True, False):
            before = self.output(first_stage=stage, model=original)['logits'][:, :3]
            after = self.output(changed, first_stage=stage, model=original)['logits'][:, :3]
            self.assertGreater((before - after).abs().max().item(), 1e-5)

    def test_original_matches_official_architecture_and_logits(self):
        source = ROOT / 'tests/reference_momke'
        spec = importlib.util.spec_from_file_location('modules.Attention_softmoe', source / 'modules/Attention_softmoe.py')
        attention = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(attention)
        # Upstream imports this absolute module. Restore the namespace afterwards.
        old = sys.modules.get('modules.Attention_softmoe')
        sys.modules['modules.Attention_softmoe'] = attention
        try:
            spec = importlib.util.spec_from_file_location('_official_momke', source / 'model_expert_softmoe.py')
            upstream = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(upstream)
        finally:
            if old is None:
                del sys.modules['modules.Attention_softmoe']
            else:
                sys.modules['modules.Attention_softmoe'] = old
        args = SimpleNamespace(device='cpu', drop_rate=0.)
        official = upstream.MoMKE(args, 3, 4, 5, 12, 7, depth=2, num_heads=2, mlp_ratio=1).eval()
        original = build_model(dict(self.config, causal=False), (3, 4, 5), 7, 9).eval()
        original.load_state_dict(official.state_dict(), strict=True)
        clean = [x * self.mask[:, :, i:i+1] for i, x in enumerate(self.features)]
        concatenated = torch.cat(clean, dim=-1).transpose(0, 1)
        for stage in (True, False):
            out = official(concatenated, self.mask.transpose(0, 1), self.valid, stage)
            adapted = self.output(first_stage=stage, model=original)
            torch.testing.assert_close(out[1], adapted['logits'], atol=1e-6, rtol=1e-5)
            if stage:
                for official_logits, adapted_logits in zip(out[2:5], adapted['auxiliary_logits']):
                    torch.testing.assert_close(official_logits, adapted_logits, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
