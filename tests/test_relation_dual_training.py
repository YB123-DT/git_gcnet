"""Exercise the actual epoch loop, not merely a hand-written loss expression."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model


class DualTrainingTests(unittest.TestCase):
    def test_epoch_single_trajectory_and_fixed_loss(self):
        torch.set_num_threads(1)
        cfg = tr.TrainConfig(dataset='CMUMOSI', backbone_type='osram',
            training_objective='emotion-only', disable_unused_aux_modules=True,
            latent_dim=8, osram_output_dim=19, osram_num_heads=2,
            osram_key_dim=3, osram_value_dim=4, osram_bidirectional=False,
            osram_relation_block=True, osram_relation_dual_readout=True)
        torch.manual_seed(3)
        model = _build_model(cfg, (1, 1, 1))
        single = copy.deepcopy(model)
        single.osram_relation_dual_readout = False
        single.osram.osram_relation_dual_readout = False
        batch = [torch.randn(6, 2, 1) for _ in range(6)]
        batch += [torch.zeros(2, 6), torch.ones(2, 6),
                  torch.tensor([[1., -1.]*3, [-1., 1.]*3]), ['a', 'b']]
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        captures = []
        for net, config in ((single, replace(cfg, osram_relation_dual_readout=False)), (model, cfg)):
            masks = []
            handle = net.register_forward_pre_hook(lambda _, args: masks.append(args[1].detach().clone()))
            with patch.object(net.osram, '_scan', wraps=net.osram._scan) as scan:
                torch.manual_seed(23)
                result = tr.train_epoch(net, [batch, batch], torch.optim.Adam(net.parameters(), lr=.001),
                    config, tr._schedules(config, 'train'), 0, (1, 1, 1), torch.device('cpu'))
                self.assertEqual(scan.call_count, 2)
            handle.remove()
            self.assertEqual(result['model_forward_count'], 2)
            self.assertEqual(result['optimizer_steps'], 2)
            self.assertEqual(result['jepa_loss'], 0)
            self.assertNotIn('paired_history', result)
            captures.append((masks, result))
        for a, b in zip(captures[0][0], captures[1][0]):
            self.assertTrue(torch.equal(a, b))
        self.assertNotIn('relation_dual_readout', captures[0][1])
        result = captures[1][1]
        values = result['relation_dual_readout']
        self.assertEqual(values['batches'], 2)
        self.assertAlmostEqual(values['combined_task_loss'],
                               .5*(values['base_task_loss']+values['full_task_loss']), places=5)
        self.assertAlmostEqual(result['loss'], values['combined_task_loss'], places=5)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        for prefix in ('osram.relation_block.', 'osram.emotion_adapter.', 'smax_fc.'):
            self.assertTrue(any(not torch.equal(before[k], p) for k, p in model.named_parameters()
                                if k.startswith(prefix)), prefix)
