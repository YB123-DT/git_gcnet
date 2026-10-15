"""CPU contract tests for the pre-extracted UTT MMIN adaptation."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.mmin import (
    MMIN, MMINConfig, Teacher, load_teacher, save_teacher,
    train_missing_step, train_teacher_step,
)


class MMINTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(66)
        torch.set_num_threads(1)
        self.config = MMINConfig((3, 4, 5), 7, embedding_dim=6,
                                 cls_layers=(8, 8), ae_layers=(9, 7, 4), n_blocks=2,
                                 dropout=0.0, teacher_dropout=0.0)
        self.teacher = Teacher(self.config)
        self.inputs = [torch.randn(6, d) for d in self.config.input_dims]
        self.visible = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                     [1, 1, 0], [1, 0, 1], [1, 1, 1.]])
        self.labels = torch.arange(6)

    def test_teacher_updates_and_missing_model_copies_encoders(self):
        before = self.teacher.netA.fc_out.weight.detach().clone()
        optim = torch.optim.Adam(self.teacher.parameters(), lr=1e-3)
        loss = train_teacher_step(self.teacher, optim, *self.inputs, self.labels)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, self.teacher.netA.fc_out.weight))
        model = MMIN(self.config, self.teacher)
        torch.testing.assert_close(model.netA.fc_out.weight, self.teacher.netA.fc_out.weight)
        self.assertIsNot(model.netA, self.teacher.netA)

    def test_eval_never_uses_hidden_features_or_teacher(self):
        model = MMIN(self.config, self.teacher).eval()
        reference = model(*self.inputs, self.visible)
        corrupted = [x.clone() for x in self.inputs]
        for index, value in enumerate(corrupted):
            value[self.visible[:, index] == 0] = float('nan')
        with torch.no_grad():
            for p in model.teacher.parameters():
                p.fill_(float('nan'))
        actual = model(*corrupted, self.visible)
        self.assertEqual(tuple(actual.shape), (6, 7))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(reference, actual)

    def test_all_missing_paths_update_and_teacher_stays_frozen(self):
        model = MMIN(self.config, self.teacher)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        optim = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        losses = train_missing_step(model, optim, *self.inputs, self.visible, self.labels)
        self.assertEqual(set(losses), {'total', 'ce', 'reconstruction', 'cycle'})
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        for prefix in ['netA.', 'netT.', 'netV.', 'netAE.', 'netAE_cycle.', 'netC.']:
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()
                                if k.startswith(prefix)), prefix)
        for k, v in model.state_dict().items():
            if k.startswith('teacher.'):
                torch.testing.assert_close(v, before[k])
        self.assertFalse(model.teacher.training)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))

    def test_teacher_signature_reuse_and_mismatch_rejection(self):
        signature = dict(dataset='MELD', fold=0, seed=66, feature_sha256='abc',
                         train_split_sha256='split', teacher_epochs=100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'teacher.pt'
            save_teacher(path, self.teacher, signature, epoch=100)
            restored, metadata = load_teacher(path, self.config, signature)
            self.assertEqual(metadata['epoch'], 100)
            for value, restored_value in zip(self.teacher.parameters(), restored.parameters()):
                torch.testing.assert_close(value, restored_value)
            # Missing rate is deliberately absent from the teacher identity.
            MMIN(self.config, restored)
            MMIN(self.config, restored)
            with self.assertRaises(ValueError):
                load_teacher(path, self.config, dict(signature, seed=67))
            with self.assertRaises(FileExistsError):
                save_teacher(path, self.teacher, signature, epoch=100)

    def test_one_utterance_and_batch_independence(self):
        model = MMIN(self.config, self.teacher).eval()
        full = model(*self.inputs, self.visible)
        one = model(*(x[:1] for x in self.inputs), self.visible[:1])
        self.assertEqual(tuple(one.shape), (1, 7))
        torch.testing.assert_close(one, full[:1], atol=1e-6, rtol=1e-5)

    def test_invalid_masks_rejected(self):
        model = MMIN(self.config, self.teacher)
        for mask in [torch.zeros(6, 3), torch.ones(6, 2), torch.full((6, 3), 0.5)]:
            with self.assertRaises(ValueError):
                model(*self.inputs, mask)

    def test_iemocap_official_initialization_and_clipping(self):
        config = replace(self.config, init_type='kaiming', init_gain=0.012,
                         teacher_clip=0.5, missing_clip=1.0)
        teacher = Teacher(config)
        # Kaiming fan-in std is sqrt(2/fan_in), not init_gain.
        self.assertGreater(float(teacher.netC.fc_out.weight.std()), 0.1)
        optim = torch.optim.SGD(teacher.parameters(), lr=0.0)
        train_teacher_step(teacher, optim, *(x * 10000 for x in self.inputs), self.labels)
        self.assertLessEqual(float(torch.nn.utils.clip_grad_norm_(teacher.netA.parameters(), float('inf'))), 0.50001)
        model = MMIN(config, teacher)
        optim = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.0)
        train_missing_step(model, optim, *(x * 10000 for x in self.inputs), self.visible, self.labels)
        for name in ['netA', 'netT', 'netV', 'netAE', 'netAE_cycle', 'netC']:
            norm = torch.nn.utils.clip_grad_norm_(getattr(model, name).parameters(), float('inf'))
            self.assertLessEqual(float(norm), 1.00001)

    def test_unsupported_initialization_rejected(self):
        with self.assertRaises(ValueError):
            Teacher(replace(self.config, init_type='invalid'))


if __name__ == '__main__':
    unittest.main()
