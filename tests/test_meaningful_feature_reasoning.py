import importlib.util
import unittest
import torch


class FeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_feature_factory_and_masked_simplexes(self):
        self.assertIsNotNone(importlib.util.find_spec('gcnet_missing_m3.meaningful_blocks_feature_reasoning'))
        from gcnet_missing_m3.meaningful_blocks_feature_reasoning import sparse_simplex
        for alpha in (1., 1.5):
            x = torch.tensor([[.3, -.1, 99., .2]], dtype=torch.float64, requires_grad=True)
            mask = torch.tensor([[1, 1, 0, 1]]).bool()
            p = sparse_simplex(x, mask, alpha=alpha)
            self.assertEqual(p[0, 2], 0)
            torch.testing.assert_close(p.sum(-1), torch.ones(1, dtype=x.dtype))
            self.assertTrue(torch.autograd.gradcheck(lambda a: sparse_simplex(a, mask, alpha=alpha), (x,)))

    def test_core_masks_gradients_and_train_only_tree_initialization(self):
        from gcnet_missing_m3.meaningful_blocks_feature_reasoning import build_feature_reasoning
        torch.manual_seed(12)
        local = torch.randn(7, 256)
        active = torch.tensor([[1,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1],
                               [1,1,1,0],[1,1,0,1],[1,0,1,1]]).bool()
        evidence = torch.randn(7, 4, 512)
        availability = (~active[:, 1:]).float()
        for name, width in (('tabnet', 64), ('node', 192)):
            core = build_feature_reasoning(name, 256, 8, 64).eval()
            state = {k: v.clone() for k, v in core.state_dict().items()}
            expected = core(local, evidence, active, availability)
            self.assertEqual(expected.shape, (7, width))
            actual = core(local, evidence.masked_fill(~active[...,None], float('nan')), active, availability)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for k,v in state.items():
                self.assertTrue(torch.equal(v, core.state_dict()[k]), k)
            core.train()
            rng = torch.get_rng_state().clone()
            result = core(local, evidence, active, availability)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            result.square().mean().backward()
            grads = [p.grad for p in core.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(g.count_nonzero() for g in grads))
            if name == 'node':
                self.assertTrue(all(layer.initialized for layer in core.layers))
                for layer in core.layers:
                    self.assertEqual(layer.leaf_bits.shape, (4,16))
                self.assertEqual([layer.input_dim for layer in core.layers], [579,643,707])

    def test_node_leaf_numbering_matches_source_bin_codes(self):
        from gcnet_missing_m3.meaningful_blocks_feature_reasoning import ObliviousTrees
        layer = ObliviousTrees(3).double().eval()
        with torch.no_grad():
            layer.threshold.fill_(-.7)
            layer.response.copy_(torch.arange(16,dtype=torch.float64)[None,:,None].expand(32,-1,2))
        result = layer(torch.zeros(1,3,dtype=torch.float64),torch.ones(1,3,dtype=torch.bool))
        torch.testing.assert_close(result,torch.full_like(result,3.9031752284),atol=1e-9,rtol=0)


if __name__ == '__main__':
    unittest.main()
