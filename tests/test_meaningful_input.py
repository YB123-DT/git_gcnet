"""One shared contract for all second-round input adapters (CPU, no training run)."""
import importlib.util
import unittest

import torch


class InputAdapterTests(unittest.TestCase):
    def test_shared_contract(self):
        self.assertIsNotNone(importlib.util.find_spec('gcnet_missing_m3.meaningful_input'))
        from gcnet_missing_m3.meaningful_input import INPUT_METHODS, MeaningfulInputAdapter
        torch.set_num_threads(1)
        torch.manual_seed(66)
        local = torch.randn(3, 7, 256)
        base = torch.cat((torch.randn(3, 7, 512), torch.zeros(3, 7, 512)), -1)
        gap = torch.cat((torch.randn(3, 7, 3, 512), torch.zeros(3, 7, 3, 512)), -1)
        av = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]).expand(3,-1,-1)
        umask = torch.ones(7, 3)
        umask[:, -1] = 0
        valid = umask.T.bool()
        gap = torch.where((valid[...,None] & ~av.bool())[...,None], gap, 0.)
        base = torch.where(valid[...,None], base, 0.)
        local = torch.where(valid[...,None], local, 0.)
        for name in INPUT_METHODS:
            with self.subTest(method=name):
                adapter = MeaningfulInputAdapter(256,1024,1600,name,8,64)
                actual = adapter(local,base,gap,av,umask)
                for expected, value in zip((local,base,gap),actual):
                    torch.testing.assert_close(value,expected,rtol=1e-6,atol=1e-6)
                dirty = gap.masked_fill(av.bool()[...,None],float('nan'))
                dirty[~valid] = float('inf')
                clean = adapter(local,base,dirty,av,umask)
                for a,b in zip(actual,clean):
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
                optimizer = torch.optim.Adam(adapter.parameters(),lr=1e-4)
                before = {k:p.detach().clone() for k,p in adapter.named_parameters()}
                for _ in range(3):
                    optimizer.zero_grad()
                    values = adapter(local,base,gap,av,umask)
                    loss = sum(v.square().mean() for v in values)
                    loss.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None))
                    optimizer.step()
                self.assertTrue(any(not torch.equal(before[k],p) for k,p in adapter.named_parameters()))
                values = adapter(local,base,gap,av,umask)
                for old,new in zip((local,base,gap),values):
                    self.assertTrue(torch.equal(old[0],new[0]))
                    self.assertEqual(new[~valid].count_nonzero(),0)
                self.assertEqual(values[2][av.bool()].count_nonzero(),0)


if __name__ == '__main__':
    unittest.main()
