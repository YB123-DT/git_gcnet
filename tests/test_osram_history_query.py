import unittest

import torch
from torch.nn import functional as F

from gcnet_missing_m3.osram import MODALITIES, OSRAMBackbone


def backbone(enabled=False):
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
                         key_dim=3, value_dim=4, dropout=0,
                         bidirectional=False, history_query_adapter=enabled)


class HistoryQueryTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for integer support encoding regression')
    def test_history_support_encoding_on_cuda(self):
        model = backbone(True).cuda()
        availability = torch.tensor([[[1, 0, 1]], [[0, 1, 0]]], device='cuda')
        valid = torch.ones(2, 1, dtype=torch.bool, device='cuda')
        queries = F.normalize(torch.randn(2, 1, 4, 2, 3, device='cuda'), dim=-1)
        out = model._adapt_history_queries(queries, availability, valid)
        self.assertEqual(model.last_history_codes[:, 0].tolist(), [0, 5])
        self.assertTrue(torch.isfinite(out).all())
        out.square().sum().backward()

    def test_training_config_cli_and_model_propagation(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
        from gcnet_missing_m3.model import MissingM3GraphModel
        from tests.test_completion_memory_write import model_kwargs
        self.assertFalse(TrainConfig().osram_history_query_adapter)
        args = ['--audio-feature','a','--text-feature','t','--video-feature','v',
                '--output-dir','unused','--osram-history-query-adapter']
        self.assertTrue(build_parser().parse_args(args).osram_history_query_adapter)
        kwargs = model_kwargs(); kwargs['completion_path'] = 'none'
        torch.manual_seed(88)
        baseline = MissingM3GraphModel(**kwargs)
        baseline_rng = torch.get_rng_state()
        torch.manual_seed(88)
        model = MissingM3GraphModel(**kwargs, osram_history_query_adapter=True)
        self.assertTrue(model.osram.history_query_adapter)
        self.assertTrue(torch.equal(baseline_rng,torch.get_rng_state()))
        for key,value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value,model.state_dict()[key]),key)
        kwargs['completion_path'] = 'pre_osram_joint_dual_projector'
        with self.assertRaisesRegex(ValueError, 'NoJEPA'):
            MissingM3GraphModel(**kwargs, osram_history_query_adapter=True)

    def test_flag_exists_and_defaults_off(self):
        self.assertFalse(OSRAMBackbone().history_query_adapter)

    def test_initialization_preserves_existing_parameters_and_rng(self):
        torch.manual_seed(19)
        old = backbone()
        old_rng = torch.get_rng_state()
        torch.manual_seed(19)
        new = backbone(True)
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        self.assertFalse(any('history_' in key for key in old.state_dict()))

    def test_exclusive_prefix_padding_batch_independence_and_reset(self):
        model = backbone(True)
        avail = torch.tensor([[[1,0,0],[0,1,0]], [[0,0,1],[1,0,0]],
                              [[0,1,0],[0,0,1]], [[1,1,1],[1,1,1]]])
        valid = torch.tensor([[1,1],[0,1],[1,1],[1,0]], dtype=torch.bool)
        q = F.normalize(torch.randn(4,2,4,2,3), dim=-1)
        model._adapt_history_queries(q, avail, valid)
        expected = torch.tensor([[[0,0,0],[0,0,0]], [[1,0,0],[0,1,0]],
                                 [[1,0,0],[1,1,0]], [[1,1,0],[1,1,1]]], dtype=torch.bool)
        self.assertTrue(torch.equal(model.last_history_support, expected))
        self.assertTrue(torch.equal(model.last_history_active, ~expected & valid[...,None]))
        model._adapt_history_queries(q[:1], avail[:1], valid[:1])
        self.assertFalse(model.last_history_support.any())

    def test_base_and_seen_queries_exact_zero_init_close(self):
        model = backbone(True)
        avail = torch.tensor([[[1,0,1]], [[0,1,0]], [[1,1,1]]])
        valid = torch.ones(3,1,dtype=torch.bool)
        q = F.normalize(torch.randn(3,1,4,2,3), dim=-1)
        out = model._adapt_history_queries(q, avail, valid)
        torch.testing.assert_close(out, q, atol=2e-7, rtol=0)
        with torch.no_grad():
            for adapter in model.history_query_adapters.values():
                adapter[-1].bias.copy_(torch.tensor([1.,2.,3.]))
        out = model._adapt_history_queries(q, avail, valid)
        self.assertTrue(torch.equal(out[:,:,0], q[:,:,0]))
        inactive = ~model.last_history_active
        self.assertTrue(torch.equal(out[:,:,1:][inactive], q[:,:,1:][inactive]))
        self.assertFalse(torch.equal(out[0,:,1:], q[0,:,1:]))

    def test_all_eight_support_codes_and_other_modality_bits_matter(self):
        model = backbone(True)
        avail = torch.tensor([[[int(i&1>0),int(i&2>0),int(i&4>0)] for i in range(8)],
                              [[0,0,0]]*8])
        q = torch.zeros(2,8,4,2,3); q[...,0] = 1
        valid = torch.ones(2,8,dtype=torch.bool)
        with torch.no_grad():
            model.history_support_embedding.weight.zero_()
            model.history_support_embedding.weight[:,0] = torch.arange(8.)
            for a in model.history_query_adapters.values():
                for p in a.parameters(): p.zero_()
                a[0].weight[0,3] = 1
                a[-1].weight[1,0] = 1
        out = model._adapt_history_queries(q,avail,valid)
        self.assertEqual(model.last_history_codes[1].tolist(), list(range(8)))
        # Audio unseen in both support 000 and 010; other bits change its query.
        self.assertFalse(torch.equal(out[1,0,1],out[1,2,1]))

    def test_nonzero_adapter_changes_gap_not_base_or_write_and_has_gradient(self):
        torch.manual_seed(12)
        model = backbone(True)
        node = torch.randn(4,2,8)
        latents = {name:torch.randn(4,2,8) for name in MODALITIES}
        avail = torch.tensor([[[1.,0,1]]*2]*4)
        qmask = torch.zeros(2,4,dtype=torch.long)
        umask = torch.ones(2,4)
        snapshots = []
        def observer(*args, **kwargs):
            snapshots.append((args,kwargs))
        model.history_query_adapter = False
        _, old = model(node,latents,avail,qmask,umask,post_write_observer=observer)
        old_snapshots = snapshots.copy(); snapshots.clear()
        model.history_query_adapter = True
        with torch.no_grad():
            model.history_query_adapters['text'][-1].bias.copy_(torch.tensor([2.,-1.,3.]))
        _, new = model(node,latents,avail,qmask,umask,post_write_observer=observer)
        self.assertTrue(torch.equal(old['base'],new['base']))
        self.assertFalse(torch.equal(old['gap'],new['gap']))
        self.assertEqual(len(snapshots),4)
        self.assertEqual(len(old_snapshots),4)
        for (before_args,before),(after_args,after) in zip(old_snapshots,snapshots):
            self.assertEqual(before_args[0],after_args[0])
            for v,w in zip(before_args[1:],after_args[1:]):
                self.assertTrue(torch.equal(v,w))
            for k,v in before.items():
                if torch.is_tensor(v): self.assertTrue(torch.equal(v,after[k]),k)
        new['gap'].square().sum().backward()
        self.assertGreater(model.history_query_adapters['text'][-1].weight.grad.abs().sum().item(),0)
        changed = node.clone(); changed[2:] += 10
        changed_avail = avail.clone(); changed_avail[2:] = torch.tensor([0.,1.,0.])
        _, future = model(changed,latents,changed_avail,qmask,umask)
        self.assertTrue(torch.equal(new['gap'][:2],future['gap'][:2]))

    def test_reject_reverse(self):
        with self.assertRaisesRegex(ValueError,'causal'):
            OSRAMBackbone(history_query_adapter=True)


if __name__ == '__main__':
    unittest.main()
