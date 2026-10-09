"""Frozen probe derivatives and causal Flat replay, without training."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DirectionTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'experiments/osram_probe_direction_20261009/direction.py'
        self.assertTrue(path.exists(), 'direction implementation missing')
        self.d = load('direction', path)
        self.old = load('old_probe', ROOT / 'experiments/osram_frozen_memory_audit_20261009/probe.py')
        torch.set_num_threads(1)
        torch.manual_seed(19)
        rng = np.random.default_rng(31)
        self.data = dict(local=rng.normal(size=(6, 256)).astype('float32'),
                         memory=rng.normal(size=(6, 4, 512)).astype('float32'),
                         availability=np.array([[1, 0, 1]] * 6, dtype='float32'),
                         utterance_indices=np.array([0, 1, 2, 0, 1, 2]))
        self.normalizer = self.old.fit_normalizer(self.data)
        self.projection = (rng.normal(size=(512, 64)) / np.sqrt(512)).astype('float32')
        self.network = nn.Sequential(nn.Linear(515, 64), nn.GELU(), nn.Linear(64, 1))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.weights = Path(self.temp.name) / 'best.pt'
        self.preprocessing = Path(self.temp.name) / 'preprocessing.npz'
        torch.save(dict(state_dict=self.network.state_dict(), input_dim=515, hidden_dim=64), self.weights)
        np.savez(self.preprocessing, projection=self.projection, **self.normalizer)
        self.probe = self.d.FrozenHistoryProbe(self.weights, self.preprocessing)
        self.args = [torch.from_numpy(self.data[k]) for k in
                     ('local', 'memory', 'availability', 'utterance_indices')]

    def test_exact_old_preprocessing_and_frozen_parameters(self):
        local, memory = self.old.transform(self.data, self.normalizer)
        features = np.concatenate((local, (memory @ self.projection).reshape(6, 256),
                                   self.data['availability']), axis=-1)
        expected = self.network(torch.from_numpy(features)).squeeze(-1)
        torch.testing.assert_close(self.probe(*self.args), expected, atol=2e-6, rtol=2e-6)
        self.probe.train()
        self.assertFalse(any(m.training for m in self.probe.modules()))
        self.assertFalse(any(p.requires_grad for p in self.probe.parameters()))
        self.assertFalse(any(b.requires_grad for b in self.probe.buffers()))

    def test_raw_memory_gradient_chain_rule_and_finite_difference(self):
        self.probe.double()
        args = [a.double() if a.is_floating_point() else a for a in self.args]
        result = self.d.probe_direction(self.probe, *args)
        active = torch.from_numpy(self.old.active_slots(self.data))
        self.assertEqual(result['gradient'][~active].count_nonzero().item(), 0)
        self.assertEqual(result['valid_gradient'].tolist(), [False, True, True, False, True, True])
        torch.testing.assert_close(result['u'][result['valid_gradient']].flatten(1).norm(dim=1), torch.ones(4, dtype=torch.double))
        perturbation = torch.randn_like(args[1]) * active[..., None]
        eps = 1e-5
        plus = self.probe(args[0], args[1] + eps * perturbation, *args[2:])
        minus = self.probe(args[0], args[1] - eps * perturbation, *args[2:])
        torch.testing.assert_close((plus - minus) / (2 * eps), (result['gradient'] * perturbation).sum((1, 2)), atol=1e-8, rtol=1e-6)
        z = ((args[1] - self.probe.memory_mean) / self.probe.memory_scale).detach().requires_grad_()
        features = torch.cat(((args[0] - self.probe.local_mean) / self.probe.local_scale,
                              (torch.where(active[..., None], z, 0.) @ self.probe.projection).flatten(1), args[2]), -1)
        self.network.double()
        dz = torch.autograd.grad(self.network(features).sum(), z)[0]
        torch.testing.assert_close(result['gradient'], dz / self.probe.memory_scale)
        self.assertFalse(any(p.grad is not None for p in self.probe.parameters()))

    def test_zero_gradient_and_inactive_nan_are_safe(self):
        with torch.no_grad():
            for p in self.probe.parameters():
                p.zero_()
        active = torch.from_numpy(self.old.active_slots(self.data))
        self.args[1][~active] = float('nan')
        result = self.d.probe_direction(self.probe, *self.args)
        self.assertFalse(result['valid_gradient'].any())
        self.assertTrue(torch.isfinite(result['prediction']).all())
        self.assertEqual(result['u'].count_nonzero().item(), 0)
        split = self.d.split_delta(torch.ones_like(self.args[1]), result['u'], result['valid_gradient'])
        self.assertEqual(split['history'].count_nonzero().item(), 0)
        torch.testing.assert_close(split['other'], torch.ones_like(self.args[1]).double())

    def test_decomposition_reconstructs_and_is_orthogonal(self):
        result = self.d.probe_direction(self.probe, *self.args)
        delta = torch.randn_like(self.args[1])
        split = self.d.split_delta(delta, result['u'], result['valid_gradient'])
        torch.testing.assert_close(split['history'] + split['other'], delta.double())
        self.assertLess(float(split['reconstruction_error'].max()), 1e-5)
        self.assertLess(float(split['orthogonality_error'].max()), 1e-5)
        torch.testing.assert_close(split['full_norm'].square(), split['history_norm'].square() + split['other_norm'].square(), atol=1e-3, rtol=1e-5)

    def test_strict_preprocessing_and_checkpoint(self):
        torch.save(dict(state_dict=self.network.state_dict(), input_dim=259, hidden_dim=64), self.weights)
        with self.assertRaises(ValueError):
            self.d.FrozenHistoryProbe(self.weights, self.preprocessing)
        torch.save(dict(state_dict=self.network.state_dict(), input_dim=515, hidden_dim=64), self.weights)
        self.normalizer['memory_scale'][0, 0] = 0
        np.savez(self.preprocessing, projection=self.projection, **self.normalizer)
        with self.assertRaises(ValueError):
            self.d.FrozenHistoryProbe(self.weights, self.preprocessing)

    def test_only_raw_memory_is_differentiable_and_nonfinite_is_explicit(self):
        local, memory, availability, indices = [a.clone() for a in self.args]
        for value in (local, memory, availability):
            value.requires_grad_()
        prediction = self.probe(local, memory, availability, indices)
        gradients = torch.autograd.grad(prediction.sum(), (local, memory, availability), allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[2])
        self.assertIsNotNone(gradients[1])
        with torch.no_grad():
            memory[1, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.d.probe_direction(self.probe, local, memory, availability, indices)

    def test_flat_replay_matches_full_backbone_valid_rows(self):
        from gcnet_missing_m3.osram import OSRAMBackbone
        model = nn.Module()
        model.osram = OSRAMBackbone(latent_dim=256, output_dim=16, num_heads=4,
                                    key_dim=4, value_dim=128, n_speakers=1,
                                    dropout=.2, bidirectional=False)
        model.smax_fc = nn.Sequential(nn.Linear(16, 7), nn.GELU(), nn.Dropout(.2), nn.Linear(7, 1))
        model.eval().requires_grad_(False)
        with torch.no_grad():
            model.osram.emotion_adapter[-1].weight.normal_(std=.02)
        node = torch.randn(4, 2, 256)
        latents = {name: torch.randn_like(node) for name in ('audio', 'text', 'visual')}
        availability = torch.tensor([[[1, 0, 1], [0, 1, 1]]] * 4).float()
        umask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]).float()
        valid = umask.T.bool()
        availability[~valid] = 0
        hidden, contexts = model.osram(node, latents, availability, torch.zeros(2, 4).long(), umask, [4, 2])
        memory = torch.cat((contexts['base'].unsqueeze(-2), contexts['gap']), -2)[..., :512][valid]
        indices = torch.arange(4)[:, None].expand(4, 2)[valid]
        replay_args = (contexts['local'][valid], memory, availability[valid], indices)
        actual = self.d.flat_predict(model, *replay_args)
        expected = model.smax_fc(hidden)[valid].squeeze(-1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        singles = torch.cat([self.d.flat_predict(model, *(v[i:i+1] for v in replay_args)) for i in range(len(indices))])
        torch.testing.assert_close(singles, expected, atol=2e-6, rtol=2e-5)
        self.assertFalse(actual.requires_grad)
        model.osram.osram_meaningful_block = 'nested_gnn_rooted_evidence'
        with self.assertRaises(ValueError):
            self.d.flat_predict(model, *replay_args)


if __name__ == '__main__':
    unittest.main()
