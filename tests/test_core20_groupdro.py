import tempfile
import unittest

import torch

from gcnet_missing_m3.train_gcnet import _emotion_loss
from gcnet_missing_m3.training_resume import TrainingState


class AuthorGroupDROTests(unittest.TestCase):
    def test_update_precedes_loss_and_absent_groups_keep_probability_mass(self):
        logits = torch.tensor([[[1.]], [[2.]]], requires_grad=True)
        labels = torch.zeros(1, 2)
        valid = torch.ones(1, 2, dtype=torch.bool)
        availability = torch.tensor([[[0., 0., 1.]], [[0., 1., 0.]]])
        weights = torch.ones(7, dtype=torch.double)
        loss, _ = _emotion_loss('CMUMOSI', logits, labels, valid, availability,
                               'pattern-groupdro-author', group_dro_weights=weights,
                               group_dro_eta=.1)
        expected = torch.softmax(torch.tensor([.1, .4, 0., 0., 0., 0., 0.], dtype=torch.double), 0)
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(loss, (expected[0] + 4 * expected[1]).float())
        loss.backward()
        torch.testing.assert_close(logits.grad.flatten(), (2 * expected[:2] * torch.tensor([1., 2.])).float())

    def test_resume_round_trips_auxiliary_objective_state(self):
        with tempfile.TemporaryDirectory() as output:
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters())
            state = TrainingState(output, identity={'test': 'groupdro'}).bind(model, optimizer)
            expected = {'group_dro_weights': torch.softmax(torch.arange(7, dtype=torch.double), 0)}
            state.commit_epoch(next_epoch=1, history=[{'epoch': 1}], selection_state={},
                               auxiliary_state=expected)
            restored = state.restore()
            torch.testing.assert_close(restored['auxiliary_state']['group_dro_weights'],
                                       expected['group_dro_weights'])


if __name__ == '__main__':
    unittest.main()
