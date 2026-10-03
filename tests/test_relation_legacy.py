"""Compare the disabled path to actual pre-change source, not itself."""
import subprocess
import types
import unittest

import torch
from gcnet_missing_m3.osram import OSRAMBackbone


class LegacyRelationTests(unittest.TestCase):
    def test_disabled_equals_prechange_commit(self):
        source = subprocess.check_output([
            'git', 'show', '7d56881:gcnet_missing_m3/osram.py'], text=True)
        old_module = types.ModuleType('legacy_osram_relation_verification')
        exec(compile(source, '7d56881:osram.py', 'exec'), old_module.__dict__)
        kwargs = dict(latent_dim=8, output_dim=10, num_heads=2,
                      key_dim=3, value_dim=4, dropout=.5, bidirectional=False)
        torch.set_num_threads(1)
        torch.manual_seed(19)
        old = old_module.OSRAMBackbone(**kwargs)
        rng = torch.get_rng_state()
        torch.manual_seed(19)
        new = OSRAMBackbone(**kwargs, osram_relation_block=False)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        new.load_state_dict(old.state_dict(), strict=True)
        node = torch.randn(4, 2, 8)
        availability = torch.tensor([[[1., 0, 1], [0, 1, 0]]] * 4)
        umask = torch.tensor([[1., 1, 1, 1], [1, 1, 0, 0]])
        availability[~umask.T.bool()] = 0
        args = (node, {m: torch.randn_like(node) for m in ('audio', 'text', 'visual')},
                availability, torch.zeros(2, 4, dtype=torch.long), umask)
        for train in (False, True):
            old.train(train)
            new.train(train)
            before = torch.get_rng_state()
            expected = old(*args)[0]
            after = torch.get_rng_state()
            torch.set_rng_state(before)
            actual = new(*args)[0]
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
