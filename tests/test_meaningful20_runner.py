"""Real tiny CPU trainer recovery; only the external data loader is replaced."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from gcnet_missing_m3 import train_gcnet as trainer
from gcnet_missing_m3.training_resume import TrainingState
from gcnet_modality_jepa.protocol import EpochSeededSubsetSampler
from experiments.osram_meaningful20_20261003 import run


def collate(indices):
    features = [torch.stack([(torch.arange(4).float() + i + j / 7) / 10 for i in indices], 1)[..., None]
                for j in range(6)]
    return features + [torch.zeros(len(indices), 4), torch.ones(len(indices), 4),
                       torch.tensor([[1., -1., 1., -1.] for _ in indices]),
                       [f'sample_{i}' for i in indices]]


def loaders(**kwargs):
    def one():
        return DataLoader(list(range(2)), batch_size=2,
                          sampler=EpochSeededSubsetSampler([0, 1], 66), collate_fn=collate)
    return [one()], [one()], [one()], 1, 1, 1


class InterruptAfterFirstCommit(TrainingState):
    def commit_epoch(self, **kwargs):
        super().commit_epoch(**kwargs)
        if kwargs['next_epoch'] == 1: raise RuntimeError('simulated interruption after epoch commit')


class RealTrainerResumeTests(unittest.TestCase):
    def test_runner_defers_schedule_identity_to_audited_actual_loader(self):
        self.assertTrue(hasattr(run, 'make_training_state'), 'runner must let trainer audit actual schedule identity')
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(run.make_training_state(root, {'protocol': 'tiny'}).schedule_identity)

    def test_real_two_epoch_cpu_trainer_resume_matches_history_masks_and_state(self):
        torch.set_num_threads(1)
        config = trainer.TrainConfig(dataset='CMUMOSI', fold=1, device='cpu', epochs=2, batch_size=2,
            backbone_type='osram', training_objective='emotion-only', disable_unused_aux_modules=True,
            osram_bidirectional=False, latent_dim=8, hidden=8, osram_output_dim=10,
            osram_num_heads=2, osram_key_dim=3, osram_value_dim=4, dropout=.3,
            checkpoint_selection='test-oracle-per-rate', train_rate_mode='cyclic')
        identity = {'source': 'tiny-fixture', 'config': 'same-two-epochs'}
        with tempfile.TemporaryDirectory() as root, patch.object(trainer, 'get_loaders', side_effect=loaders):
            full, interrupted = Path(root, 'full'), Path(root, 'interrupted')
            expected_metrics = trainer.run_experiment(config, 'a', 't', 'v', full,
                training_state=TrainingState(full, identity=identity))
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                trainer.run_experiment(config, 'a', 't', 'v', interrupted,
                    training_state=InterruptAfterFirstCommit(interrupted, identity=identity))
            actual_metrics = trainer.run_experiment(config, 'a', 't', 'v', interrupted,
                training_state=TrainingState(interrupted, identity=identity))
            expected = torch.load(full / 'last_training.pt', weights_only=False)
            actual = torch.load(interrupted / 'last_training.pt', weights_only=False)
            self.assertEqual(actual['next_epoch'], 2)
            self.assertEqual(actual['history'], expected['history'])
            self.assertEqual(actual['selection_state'], expected['selection_state'])
            self.assertEqual(actual['schedule_identity'], expected['schedule_identity'])
            for key in expected['model']:
                self.assertTrue(torch.equal(actual['model'][key], expected['model'][key]), key)
            self.assertEqual(actual_metrics['test'], expected_metrics['test'])
            for i in range(8):
                name = f'predictions_miss_0p{i}.npz'
                with np.load(full / name) as first, np.load(interrupted / name) as second:
                    self.assertEqual(first.files, second.files)
                    for key in first.files:
                        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


if __name__ == '__main__': unittest.main()
