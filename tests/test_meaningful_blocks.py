import importlib.util
import unittest
import torch


class WrapperTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_factory_and_safe_zero_bridge(self):
        self.assertIsNotNone(importlib.util.find_spec('gcnet_missing_m3.meaningful_blocks'))
        from gcnet_missing_m3.meaningful_blocks import MEANINGFUL_METHODS, MeaningfulReadoutResidual
        self.assertEqual(len(set(MEANINGFUL_METHODS)),20)
        local = torch.randn(4,2,256)
        base = torch.randn(4,2,1024)
        gap = torch.randn(4,2,3,1024)
        av = torch.tensor([[[1,0,1],[1,1,1]]]*4).float()
        umask = torch.tensor([[1,1,1,0],[0,1,1,1]])
        anchor = torch.randn(4,2,1600)
        model = MeaningfulReadoutResidual(256,1024,1600,'tabnet',8,64).eval()
        self.assertEqual(model(local,base,gap,av,umask,anchor).count_nonzero(),0)
        model.output.weight.data.normal_(std=.01)
        model.output.bias.data.fill_(.3)
        expected = model(local,base,gap,av,umask,anchor)
        valid = umask.T.bool()
        first = valid & (valid.long().cumsum(0)==1)
        self.assertEqual(expected[~valid|first].count_nonzero(),0)
        gap = gap.masked_fill(av.bool()[...,None],float('nan'))
        gap[...,512:] = float('inf')
        base[...,512:] = float('nan')
        local = local.masked_fill(~valid[...,None],float('nan'))
        torch.testing.assert_close(model(local,base,gap,av,umask,anchor),expected,atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
