import importlib.util
from pathlib import Path
import unittest
import torch
from torch import nn

PATH = Path(__file__).resolve().parents[1] / 'experiments/osram_frozen_evidence_gate_20260930/run.py'


class FrozenGateTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.exists(), 'frozen Gate runner is not implemented')
        spec = importlib.util.spec_from_file_location('frozen_gate_runner', PATH)
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)

    def model(self):
        model = nn.Sequential()
        model.add_module('osram', nn.Module())
        model.osram.add_module('local_evidence_gate', nn.Linear(3, 3))
        model.osram.add_module('anchor', nn.BatchNorm1d(3))
        model.add_module('classifier', nn.Linear(3, 1))
        return model

    def test_only_gate_trains_and_frozen_buffers_stay_eval(self):
        m = self.model()
        params = self.runner.freeze_except_gate(m)
        m.train()
        self.assertTrue(m.osram.local_evidence_gate.training)
        self.assertFalse(m.training)
        self.assertFalse(m.osram.training)
        self.assertFalse(m.osram.anchor.training)
        self.assertEqual({id(p) for p in params}, {id(p) for p in m.osram.local_evidence_gate.parameters()})
        before = self.runner.frozen_state_hash(m)
        optimizer = torch.optim.Adam(params, lr=.01)
        gate_before = m.osram.local_evidence_gate.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            x = m.osram.local_evidence_gate(torch.ones(4, 3))
            m.classifier(m.osram.anchor(x)).square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in params))
            optimizer.step()
        self.assertFalse(torch.equal(gate_before, m.osram.local_evidence_gate.weight))
        self.assertEqual(before, self.runner.frozen_state_hash(m))
        m.eval()
        self.assertFalse(m.osram.local_evidence_gate.training)

    def test_load_rejects_missing_anchor_and_existing_gate(self):
        m = self.model()
        state = {k: v for k,v in m.state_dict().items() if not k.startswith(self.runner.GATE_PREFIX)}
        self.runner.load_flat_state(m, state)
        broken = dict(state)
        broken.pop('classifier.bias')
        with self.assertRaises(ValueError):
            self.runner.load_flat_state(m, broken)
        with self.assertRaises(ValueError):
            self.runner.load_flat_state(m, m.state_dict())

    def test_buffer_mutation_changes_hash(self):
        m = self.model()
        before = self.runner.frozen_state_hash(m)
        m.osram.anchor.running_mean.add_(1)
        self.assertNotEqual(before, self.runner.frozen_state_hash(m))

    def test_actual_train_epoch_updates_only_gate(self):
        from gcnet_missing_m3 import train_gcnet as tr
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        c = tr.TrainConfig(dataset='CMUMOSI', backbone_type='osram', training_objective='emotion-only',
            latent_dim=8, osram_output_dim=19, osram_num_heads=2, osram_key_dim=3, osram_value_dim=4,
            osram_bidirectional=False, osram_local_evidence_gate=True)
        model = _build_model(c, (4, 4, 4))
        # A trained Flat anchor has learned this layer; the untrained factory
        # zero-initializes it and deliberately blocks all history sensitivity.
        with torch.no_grad():
            model.osram.emotion_adapter[-1].weight.normal_(std=.02)
        parameters = self.runner.freeze_except_gate(model)
        before = self.runner.frozen_state_hash(model)
        gate_before = model.osram.local_evidence_gate.output.weight.detach().clone()
        batch = [torch.randn(3, 2, 4) for _ in range(6)]
        batch += [torch.zeros(2, 3), torch.ones(2, 3),
                  torch.tensor([[1., -1., 1.], [-1., 1., -1.]]), ['a', 'b']]
        optimizer = torch.optim.Adam(parameters, lr=.001)
        result = tr.train_epoch(model, [batch, batch, batch], optimizer, c, tr._schedules(c, 'train'),
                                0, (4, 4, 4), torch.device('cpu'))
        self.assertEqual(result['optimizer_steps'], 3)
        self.assertEqual(before, self.runner.frozen_state_hash(model))
        self.assertFalse(torch.equal(gate_before, model.osram.local_evidence_gate.output.weight))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
        for name, module in model.named_modules():
            if not name.startswith('osram.local_evidence_gate'):
                self.assertFalse(module.training, name)


if __name__ == '__main__':
    unittest.main()
