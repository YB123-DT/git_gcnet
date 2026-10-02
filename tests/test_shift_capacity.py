import unittest
import torch
from gcnet_missing_m3.osram import MemoryShiftFilter, OSRAMBackbone
from tests import test_memory_shift_residual as legacy_tests


class ShiftCapacityTests(unittest.TestCase):
    def test_width_depth_scalar_and_gradients(self):
        for width in (128, 256):
            for depth in (1, 2, 3):
                with self.subTest(width=width, depth=depth):
                    m = MemoryShiftFilter(4, 3, 5, relation_dim=7,
                                          filter_width=width, filter_depth=depth)
                    layers = [x for x in m.filter if isinstance(x, torch.nn.Linear)]
                    self.assertEqual(len(layers), depth + 1)
                    self.assertEqual([x.out_features for x in layers], [width]*depth+[1])
                    args = legacy_tests.MemoryShiftResidualTests().inputs()
                    self.assertEqual(m(*args).count_nonzero().item(), 0)
                    with torch.no_grad(): m.residual_projector.weight.fill_(.1)
                    m(*args).square().mean().backward()
                    for p in m.parameters():
                        self.assertIsNotNone(p.grad)
                        self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in m.filter.parameters()), 0)

    def test_default_state_and_rng_unchanged(self):
        # Reproduce the original initialization sequence, not another call to
        # the configurable constructor, to protect legacy weights and RNG.
        class Legacy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.local_relation = torch.nn.Linear(4,128)
                self.memory_relation = torch.nn.Linear(3,128)
                self.evidence_type = torch.nn.Embedding(4,128)
                self.filter = torch.nn.Sequential(torch.nn.Linear(768,128),
                    torch.nn.GELU(),torch.nn.Linear(128,1))
                self.residual_projector = torch.nn.Linear(128,5)
                torch.nn.init.zeros_(self.residual_projector.weight)
                torch.nn.init.zeros_(self.residual_projector.bias)
        torch.manual_seed(18)
        old = Legacy()
        rng = torch.get_rng_state()
        torch.manual_seed(18)
        new = MemoryShiftFilter(4, 3, 5, filter_width=128, filter_depth=1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        self.assertEqual(old.filter[0].weight.shape, (128, 768))
        self.assertEqual(old.filter[2].weight.shape, (1, 128))
        for k,v in old.state_dict().items(): self.assertTrue(torch.equal(v,new.state_dict()[k]))

    def test_deeper_initially_exact_flat(self):
        kwargs = dict(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
                      value_dim=4, dropout=0., bidirectional=False)
        torch.manual_seed(81); flat = OSRAMBackbone(**kwargs).eval()
        rng = torch.get_rng_state()
        torch.manual_seed(81)
        new = OSRAMBackbone(**kwargs, osram_readout_fusion='memory-shift-residual',
                            osram_shift_filter_width=256, osram_shift_filter_depth=3).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in flat.state_dict().items(): self.assertTrue(torch.equal(v,new.state_dict()[k]))
        node=torch.randn(3,2,8)
        args=(node,{m:torch.randn_like(node) for m in ('audio','text','visual')},
              torch.tensor([[[1.,0,1],[0,1,0]]]*3), torch.zeros(2,3,dtype=torch.long),torch.ones(2,3))
        self.assertTrue(torch.equal(flat(*args)[0],new(*args)[0]))

    def test_invalid_dimensions(self):
        for name in ('filter_width','filter_depth'):
            for value in (0,-1,1.5,True):
                with self.assertRaises(ValueError): MemoryShiftFilter(4,3,5,**{name:value})

    def test_config_cli_and_eval_builder(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        args = build_parser().parse_args(['--audio-feature','a','--text-feature','t',
            '--video-feature','v','--output-dir','unused',
            '--osram-shift-filter-width','256','--osram-shift-filter-depth','3'])
        self.assertEqual((args.osram_shift_filter_width,args.osram_shift_filter_depth),(256,3))
        config = TrainConfig(backbone_type='osram', osram_readout_fusion='memory-shift-residual',
            osram_shift_filter_width=256,osram_shift_filter_depth=3)
        model = _build_model(config,(4,5,6))
        self.assertEqual(model.osram.memory_shift_filter.filter[-1].weight.shape,(1,256))
        self.assertEqual(len(model.osram.memory_shift_filter.filter),7)
        for name in ('osram_shift_filter_width','osram_shift_filter_depth'):
            for value in (0,-1,1.5,True):
                with self.assertRaises(ValueError): TrainConfig(**{name:value})


if __name__ == '__main__': unittest.main()
