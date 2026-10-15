"""CPU-only state restoration tests; all artifacts are synthetic temp fixtures."""
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_one
from train_one import rng_state, restore_rng, save_torch
from common_data import random_masks
import adapters.mmin as mmin


class ResumeStateTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '',
                         'Run this test with CUDA_VISIBLE_DEVICES empty')
        self.assertFalse(torch.cuda.is_available())
        torch.set_num_threads(1)
        self.cfg = mmin.MMINConfig((3, 4, 5), 3, embedding_dim=6,
                                   cls_layers=(8, 8), ae_layers=(9, 7, 4), n_blocks=2,
                                   dropout=.3, teacher_dropout=.3)
        generator = torch.Generator().manual_seed(1234)
        self.features = [torch.randn(12, d, generator=generator) for d in self.cfg.input_dims]
        self.labels = torch.arange(12) % 3

    def seed(self, value=66):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        return np.random.default_rng(value)

    def model_and_optimizer(self):
        model = mmin.MMIN(self.cfg, mmin.Teacher(self.cfg))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                     lr=1e-3, betas=(.5, .999))
        return model, optimizer

    def advance(self, model, optimizer, rng, steps):
        trace = []
        for _ in range(steps):
            # Every persisted RNG contributes to the subsequent real training
            # step: Python selects MR, global numpy scales inputs, generator
            # chooses utterances/masks, torch drives dropout and perturbations.
            mr = random.choice([.1, .3, .6])
            scale = float(np.random.uniform(.8, 1.2))
            order = rng.permutation(len(self.labels))[:6]
            masks = random_masks(len(order), mr, rng)
            inputs = [x[order] * scale + torch.randn_like(x[order]) * .01 for x in self.features]
            losses = mmin.train_missing_step(model, optimizer, *inputs,
                                            torch.from_numpy(masks), self.labels[order])
            trace.append(dict(mr=mr, scale=scale, order=order.copy(),
                              masks=masks.copy(), losses=losses))
        return trace

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(left, right)

    def test_checkpoint_restores_adam_all_rng_and_exact_training_continuation(self):
        rng = self.seed()
        model, optimizer = self.model_and_optimizer()
        self.advance(model, optimizer, rng, 2)
        with tempfile.TemporaryDirectory(prefix='mmin-resume-test-') as temp:
            checkpoint = Path(temp) / 'last.pt'
            save_torch(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                       epoch=2, rng=rng_state(rng)))
            self.assertFalse(checkpoint.with_name('last.pt.tmp').exists())
            uninterrupted = self.advance(model, optimizer, rng, 3)
            expected_rng = rng_state(rng)
            # A fresh process would consume random numbers during model
            # construction. Restoration must occur after construction/loading.
            restored_rng = self.seed(987)
            restored, restored_optimizer = self.model_and_optimizer()
            saved = torch.load(checkpoint, map_location='cpu')
            restored.load_state_dict(saved['model'], strict=True)
            restored_optimizer.load_state_dict(saved['optimizer'])
            restore_rng(saved['rng'], restored_rng)
            self.assertEqual(saved['epoch'] + 1, 3)
            resumed = self.advance(restored, restored_optimizer, restored_rng, 3)
            self.assert_tree_equal(uninterrupted, resumed)
            self.assert_tree_equal(model.state_dict(), restored.state_dict())
            self.assert_tree_equal(optimizer.state_dict(), restored_optimizer.state_dict())
            self.assert_tree_equal(expected_rng, rng_state(restored_rng))

    def test_rng_restore_repeats_python_numpy_mask_and_torch_samples(self):
        rng = self.seed()
        before = rng_state(rng)
        def samples():
            return dict(python=[random.random() for _ in range(4)],
                        numpy=np.random.normal(size=7),
                        masks=random_masks(30, .6, rng),
                        order=rng.permutation(17), torch=torch.rand(9))
        first = samples()
        samples()
        restore_rng(before, rng)
        self.assert_tree_equal(first, samples())
        self.assertEqual(before['cuda'], [])

    def test_real_runner_teacher_cache_initialization_is_independent_of_mr(self):
        # Execute the real MMIN runner with tiny CPU fixtures, observing only
        # model construction. No datasets, production runs or GPU are touched.
        def record(name, offset):
            return dict(id=name, utterance_ids=['u0', 'u1', 'u2'],
                        a=np.arange(9, dtype=np.float32).reshape(3, 3) / 10 + offset,
                        t=np.arange(12, dtype=np.float32).reshape(3, 4) / 10 + offset,
                        v=np.arange(15, dtype=np.float32).reshape(3, 5) / 10 + offset,
                        labels=np.arange(3, dtype=np.int64))
        training, testing = [record('train', 0)], [record('test', .1)]
        data = dict(metadata=dict(dimensions=[3, 4, 5], selected_features_labels_sha256='synthetic-test-only'))
        captures = []
        real_class = mmin.MMIN
        def construct(config, teacher):
            model = real_class(config, teacher)
            captures.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return model
        with tempfile.TemporaryDirectory(prefix='mmin-cache-test-') as temp:
            scratch = Path(temp)
            outputs, teacher_paths = [], []
            for index, mr in enumerate([.3, .6, .3]):
                config = dict(model='MMIN', dataset='MELD', fold=0, seed=66,
                              n_classes=3, epochs=1, stage_epoch=0, teacher_epochs=2,
                              batch_size=3, lr=1e-3, teacher_betas=[.9, .999],
                              missing_betas=[.5, .999], missing_rate=mr,
                              eval_mask_seed=20261015, embedding_dim=6,
                              cls_layers=[8, 8], ae_layers=[9, 7, 4], n_blocks=2)
                run = scratch / ('run-' + str(index))
                run.mkdir()
                manifest = dict(result_type='smoke', expected_n_test=3,
                                expected_sample_ids=['test::u0', 'test::u1', 'test::u2'],
                                mask_key_records=[dict(id='test', utterance_ids=testing[0]['utterance_ids'])])
                train_one.atomic_json(run / 'config.json', config)
                rng = self.seed()
                with mock.patch.object(train_one, 'RUN_ROOT', scratch), \
                     mock.patch.object(mmin, 'MMIN', side_effect=construct), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train_one.run_mmin(types.SimpleNamespace(smoke=True), run, config, data,
                                       training, testing, torch.device('cpu'), rng, manifest)
                self.assertTrue(train_one.valid_result(run))
                outputs.append(torch.load(run / 'last.pt', map_location='cpu'))
                actual = json.loads((run / 'manifest.json').read_text())
                teacher_paths.append(actual['teacher_path'])
            self.assertEqual(len(set(teacher_paths)), 1)
            self.assertEqual(len(list((scratch / 'smoke_teachers').glob('*/teacher.pt'))), 1)
            teacher_history = Path(teacher_paths[0]).parent / 'history.jsonl'
            self.assertEqual(len(teacher_history.read_text().splitlines()), 2)
            self.assert_tree_equal(captures[0], captures[1])
            self.assert_tree_equal(captures[0], captures[2])
            # A cached rerun of the same MR reproduces full training state;
            # a different MR still starts with exactly identical parameters.
            self.assert_tree_equal(outputs[0]['model'], outputs[2]['model'])
            self.assert_tree_equal(outputs[0]['optimizer'], outputs[2]['optimizer'])
            self.assert_tree_equal(outputs[0]['rng'], outputs[2]['rng'])


if __name__ == '__main__':
    unittest.main()
