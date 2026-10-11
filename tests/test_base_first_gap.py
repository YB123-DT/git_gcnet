import importlib
import base64
import os
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from gcnet_missing_m3.osram import OSRAMBackbone


class BaseFirstGapTests(unittest.TestCase):
    def fusion(self):
        self.assertIsNotNone(importlib.util.find_spec('gcnet_missing_m3.base_first_gap'),
                             'Base-first fusion is not implemented')
        cls = importlib.import_module('gcnet_missing_m3.base_first_gap').BaseFirstGapFusion
        return cls(8, 12, 10, dropout=0)

    def inputs(self):
        return (torch.randn(4, 2, 8), torch.randn(4, 2, 12),
                torch.randn(4, 2, 3, 12),
                torch.tensor([[[0,1,1],[1,1,1]], [[1,1,1],[0,1,0]],
                              [[0,1,0],[1,0,1]], [[0,0,0],[0,0,0]]]),
                torch.tensor([[1,1,1,0],[0,1,1,0]]))

    def test_zero_init_and_hard_masks_after_bias(self):
        f = self.fusion().eval()
        args = self.inputs()
        skip, norm = nn.Linear(8, 10), nn.LayerNorm(10)
        valid = args[-1].T.bool()
        expected = torch.where(valid[...,None], norm(skip(args[0])), 0.)
        torch.testing.assert_close(f(*args, local_skip=skip, emotion_norm=norm), expected)
        nn.init.normal_(f.base_mlp[-1].weight)
        nn.init.normal_(f.gap_mlp[-1].weight)
        nn.init.constant_(f.gap_mlp[-1].bias, 5)
        clean = f(*args, local_skip=skip, emotion_norm=norm)
        history = valid & ((valid.long().cumsum(0) - valid.long()) > 0)
        active = history[...,None] & ~args[3].bool()
        dirty = [x.clone() for x in args]
        dirty[0][~valid] = float('nan')
        dirty[1][~history] = float('nan')
        dirty[2][~active] = float('nan')
        output = f(*dirty, local_skip=skip, emotion_norm=norm)
        torch.testing.assert_close(output, clean)
        self.assertTrue(torch.isfinite(output).all())
        output.square().sum().backward()
        for p in f.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        # An arbitrary final bias still cannot correct a no-gap/no-history row.
        with torch.no_grad():
            f.gap_mlp[-1].bias.add_(100)
        changed = f(*args, local_skip=skip, emotion_norm=norm)
        torch.testing.assert_close(changed[~active.any(-1)], clean[~active.any(-1)])

    def test_budget_and_full_context_width(self):
        cls = type(self.fusion())
        f = cls(256, 1024, 1600, dropout=.5)
        old_input = 256 + 4 * 1024
        budget = 2*old_input + old_input*1600 + 1600 + 1600*1600 + 1600
        def count(h):
            a, b, d = 256+1024, 1600+3*1024, 1600
            return 2*(a+b) + 2*d + h*(a+b+2*d+2) + 2*d
        self.assertEqual(sum(p.numel() for p in f.parameters()), count(f.hidden_dim))
        for h in (f.hidden_dim-1, f.hidden_dim+1):
            self.assertLessEqual(abs(count(f.hidden_dim)-budget), abs(count(h)-budget))
        self.assertEqual(f.base_mlp[0].normalized_shape, (1280,))
        self.assertEqual(f.gap_mlp[0].normalized_shape, (4672,))

    def test_default_matches_legacy_source_state_outputs_and_rng(self):
        root = Path(__file__).resolve().parents[1]
        fixture = os.environ.get('OSRAM_BASEFIRST_LEGACY_SOURCE_B64')
        source = (base64.b64decode(fixture).decode() if fixture else
                  subprocess.check_output(['git', 'show', '7d4bb0f:gcnet_missing_m3/osram.py'],
                                          cwd=root, text=True))
        legacy = types.ModuleType('gcnet_missing_m3.legacy_osram')
        exec(compile(source, '<legacy-osram>', 'exec'), legacy.__dict__)
        kw = dict(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
                  value_dim=3, dropout=0, n_speakers=1, bidirectional=False)
        torch.manual_seed(37)
        before = legacy.OSRAMBackbone(**kw).eval()
        rng = torch.random.get_rng_state()
        torch.manual_seed(37)
        after = OSRAMBackbone(**kw).eval()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(set(before.state_dict()), set(after.state_dict()))
        for k, v in before.state_dict().items():
            self.assertTrue(torch.equal(v, after.state_dict()[k]), k)
        node, _, _, availability, umask = self.inputs()
        umask[1] = umask[0]
        availability[~umask.T.bool()] = 0
        latents = {k: torch.randn_like(node) for k in ('audio','text','visual')}
        args = (node, latents, availability, torch.zeros(2,4,dtype=torch.long), umask, [3,3])
        torch.testing.assert_close(before(*args)[0], after(*args)[0], rtol=0, atol=0)

    def test_backbone_preserves_memory_shared_weights_and_rng(self):
        self.fusion()
        kw = dict(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
                  value_dim=3, dropout=0, n_speakers=1, bidirectional=False)
        torch.manual_seed(31)
        old = OSRAMBackbone(**kw).eval()
        rng = torch.random.get_rng_state()
        torch.manual_seed(31)
        new = OSRAMBackbone(**kw, osram_base_first_gap=True).eval()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertIsInstance(new.emotion_adapter, nn.Identity)
        for k, v in old.state_dict().items():
            if not k.startswith('emotion_adapter.'):
                self.assertTrue(torch.equal(v, new.state_dict()[k]), k)
        node, _, _, availability, umask = self.inputs()
        umask[1] = umask[0]
        availability[~umask.T.bool()] = 0
        latents = {k: torch.randn_like(node) for k in ('audio','text','visual')}
        args = (node, latents, availability, torch.zeros(2,4,dtype=torch.long), umask, [3,3])
        states = [[], []]
        outputs = []
        for i, model in enumerate((old,new)):
            original = model.block_write
            def capture(*a, **kw):
                state = original(*a, **kw)
                states[i].append(state.detach().clone())
                return state
            with patch.object(model, 'block_write', side_effect=capture):
                outputs.append(model(*args))
        for a,b in zip(states[0],states[1]):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(outputs[0][0], outputs[1][0],rtol=0,atol=0)

    def test_config_accepts_independent_path_and_rejects_conflicts(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig
        kw = dict(backbone_type='osram', osram_bidirectional=False,
                  training_objective='emotion-only', osram_base_first_gap=True)
        self.assertTrue(TrainConfig(**kw).osram_base_first_gap)
        for field, value in dict(osram_bidirectional=True, training_objective='joint',
                                osram_readout_fusion='base-gap-delta',
                                classification_completion=True, osram_post_grn=True,
                                osram_adapter_hidden_dim=100, core20_method='C01').items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                TrainConfig(**dict(kw, **{field: value}))


if __name__ == '__main__':
    unittest.main()
