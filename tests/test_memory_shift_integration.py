import unittest
from types import SimpleNamespace

import torch
from gcnet_missing_m3 import train_gcnet as training
from gcnet_missing_m3.model import MissingM3GraphModel
from tests.test_completion_memory_write import model_kwargs, inputs


class MemoryShiftIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA cfg84 equivalence check')
    def test_cuda_cfg84_initial_logits_exact(self):
        from dataclasses import replace
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        config = training.TrainConfig(
            dataset='CMUMOSI', backbone_type='osram', training_objective='emotion-only',
            disable_unused_aux_modules=True, osram_bidirectional=False,
            osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64,
            osram_value_dim=64, osram_write_step=.6)
        torch.manual_seed(91)
        flat = _build_model(config, (512,1024,1024)).cuda()
        torch.manual_seed(91)
        new = _build_model(replace(config, osram_readout_fusion='memory-shift-residual'),
                           (512,1024,1024)).cuda()
        x = torch.randn(4,2,2560,device='cuda')
        a = torch.tensor([[[1.,0,1],[0,1,0]]]*4,device='cuda')
        u = torch.tensor([[1.,1,1,1],[1,1,0,0]],device='cuda')
        a[~u.T.bool()] = 0
        q = torch.zeros(2,4,dtype=torch.long,device='cuda')
        for train in (False,True):
            flat.train(train); new.train(train)
            before = torch.cuda.get_rng_state()
            old = flat([x],a,q,u,[4,2],predict_missing=False)[0]
            after = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(before)
            actual = new([x],a,q,u,[4,2],predict_missing=False)[0]
            self.assertTrue(torch.equal(old,actual))
            self.assertTrue(torch.equal(after,torch.cuda.get_rng_state()))
            self.assertEqual(new.osram.last_diagnostics['memory_shift_filter']['shift_residual_norm'],0.)

    def test_config_and_cli_and_retired_interpolation(self):
        config = training.TrainConfig(backbone_type='osram', osram_readout_fusion='memory-shift-residual')
        self.assertEqual(config.osram_readout_fusion, 'memory-shift-residual')
        args = training.build_parser().parse_args([
            '--audio-feature', 'a', '--text-feature', 't', '--video-feature', 'v',
            '--output-dir', 'unused', '--osram-readout-fusion', 'memory-shift-residual'])
        self.assertEqual(args.osram_readout_fusion, 'memory-shift-residual')
        self.assertFalse(config.osram_history_query_adapter)
        self.assertEqual(training.TrainConfig().osram_readout_fusion, 'flat')
        with self.assertRaises(ValueError):
            training.TrainConfig(backbone_type='osram', osram_readout_fusion='history-innovation')

    def test_full_model_initialization_and_logits_exact(self):
        kwargs = model_kwargs(); kwargs['completion_path'] = 'none'
        kwargs.update(dropout=.2, projector_dropout=.1)
        torch.manual_seed(44)
        flat = MissingM3GraphModel(**kwargs)
        rng = torch.get_rng_state()
        torch.manual_seed(44)
        new = MissingM3GraphModel(**kwargs, osram_readout_fusion='memory-shift-residual')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in flat.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        self.assertFalse(new.osram.history_query_adapter)
        x,a,q,u,lengths = inputs()
        for train in (False, True):
            flat.train(train); new.train(train)
            before = torch.get_rng_state()
            old_logits = flat([x],a,q,u,lengths,predict_missing=False)[0]
            after = torch.get_rng_state()
            torch.set_rng_state(before)
            new_logits = new([x],a,q,u,lengths,predict_missing=False)[0]
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            self.assertTrue(torch.equal(old_logits,new_logits))
        self.assertTrue(all(p.requires_grad for p in new.osram.emotion_adapter.parameters()))
        self.assertTrue(all(p.requires_grad for p in new.osram.local_skip.parameters()))

    def test_diagnostics_active_weighted_and_valid_weighted(self):
        self.assertTrue(hasattr(training, '_accumulate_memory_shift'))
        names = ('base','gap_audio','gap_text','gap_visual')
        totals = {}
        model = SimpleNamespace(osram=SimpleNamespace(last_diagnostics={}))
        for count, active, gate, norm in [(2,(2,2,0,0),.2,3.),(6,(6,0,3,0),.8,7.)]:
            model.osram.last_diagnostics = {'memory_shift_filter': {
                'valid_count':count, 'active_counts':dict(zip(names,active)),
                'filter_mean': {name: gate if n else None for name,n in zip(names,active)},
                'filtered_shift_norm':norm, 'shift_residual_norm':norm/2,
                'residual_anchor_norm_ratio':norm/10}}
            training._accumulate_memory_shift(model,totals)
        result=training._memory_shift_metrics(totals)['memory_shift_filter']
        self.assertEqual(result['valid_count'],8)
        self.assertAlmostEqual(result['filter_mean']['base'],.65)
        self.assertAlmostEqual(result['filter_mean']['gap_audio'],.2)
        self.assertAlmostEqual(result['filter_mean']['gap_text'],.8)
        self.assertIsNone(result['filter_mean']['gap_visual'])
        self.assertEqual(result['active_counts']['gap_text'],3)
        self.assertAlmostEqual(result['filtered_shift_norm'],6.)
        self.assertAlmostEqual(result['shift_residual_norm'],3.)
        self.assertAlmostEqual(result['residual_anchor_norm_ratio'],.6)
        totals={}
        training._accumulate_memory_shift(SimpleNamespace(),totals)
        self.assertEqual(training._memory_shift_metrics(totals),{})


if __name__ == '__main__':
    unittest.main()
