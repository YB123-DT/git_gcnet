from dataclasses import replace
import inspect
import subprocess
import types
import unittest
import torch
from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_missing_m3.model import MissingM3GraphModel
from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser, run_experiment
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model


def config(method='none'):
    return TrainConfig(dataset='CMUMOSI',backbone_type='osram',training_objective='emotion-only',
        disable_unused_aux_modules=True,osram_bidirectional=False,latent_dim=256,
        osram_output_dim=1600,osram_num_heads=8,osram_key_dim=64,osram_value_dim=64,
        dropout=.5,projector_dropout=.1,osram_meaningful_block=method)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_new_independent_flag(self):
        self.assertIn('training_state', inspect.signature(run_experiment).parameters)
        for cls in (TrainConfig,OSRAMBackbone,MissingM3GraphModel):
            self.assertIn('osram_meaningful_block',inspect.signature(cls).parameters)
        cfg=config('tabnet')
        for changes in ({'osram_readout_candidate':'film'},{'osram_relation_block':True},
                        {'osram_bidirectional':True},{'paired_history_views':True},
                        {'osram_num_heads':4},{'training_objective':'joint'},
                        {'train_rate_mode':'conversation-mixed'}):
            with self.assertRaises(ValueError):
                replace(cfg,**changes)
        args=build_parser().parse_args(['--audio-feature','a','--text-feature','t',
            '--video-feature','v','--output-dir','unused','--osram-meaningful-block','tabnet'])
        self.assertEqual(args.osram_meaningful_block,'tabnet')

    def test_old_default_and_zero_bridge_full_model(self):
        from gcnet_missing_m3.meaningful_blocks import MEANINGFUL_METHODS
        model=_build_model(config(),(3,4,5))
        state={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.randn(4,2,12)
        a=torch.tensor([[[1.,0,1],[0,1,0]]]*4)
        u=torch.tensor([[1.,1,1,1],[1,1,0,0]])
        a[~u.T.bool()]=0
        args=([x],a,torch.zeros(2,4,dtype=torch.long),u,[4,2])
        with torch.no_grad():
            model.osram.emotion_adapter[-1].weight.normal_(std=.005)
        for name in MEANINGFUL_METHODS:
            enabled=_build_model(config(name),(3,4,5))
            for key,value in state.items():
                self.assertTrue(torch.equal(value,enabled.state_dict()[key]),(name,key))
            incompatible=enabled.load_state_dict(model.state_dict(),strict=False)
            self.assertFalse(incompatible.unexpected_keys)
            self.assertTrue(all(k.startswith('osram.meaningful_block.') for k in incompatible.missing_keys))
            for training in (False,True):
                model.train(training); enabled.train(training)
                torch.manual_seed(17); expected=model(*args)[0]; rng=torch.get_rng_state()
                torch.manual_seed(17); actual=enabled(*args)[0]
                self.assertTrue(torch.equal(expected,actual),name)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()),name)


if __name__=='__main__':
    unittest.main()
