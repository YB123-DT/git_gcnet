import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'experiments/osram_frozen_memory_audit_20261009/probe.py'


def fixture(offset=0):
    rng = np.random.default_rng(7)
    n = 12
    memory = rng.normal(size=(n, 4, 512)).astype('float32')
    memory[::4] = 0
    memory[:, 1] = 0
    memory[:, 3] = 0
    return dict(local=rng.normal(size=(n, 256)).astype('float32'), memory=memory,
                availability=np.tile([1, 0, 1], (n, 1)).astype('float32'),
                labels=np.tile([-1., 0., 1., 2.], 3),
                conversation_ids=np.repeat([f'c{offset+i}' for i in range(3)], 4),
                utterance_indices=np.tile(np.arange(4), 3), speaker=np.zeros(n),
                prediction=np.zeros(n))


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if PATH.exists():
            spec = importlib.util.spec_from_file_location('frozen_probe', PATH)
            cls.p = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.p)
        else:
            cls.p = None

    def setUp(self):
        self.assertIsNotNone(self.p, 'probe implementation is missing')

    def test_train_only_normalizer_and_zero_slots(self):
        train = fixture()
        normalizer = self.p.fit_normalizer(train)
        before = {k: v.copy() for k, v in normalizer.items()}
        shifted = fixture(20)
        shifted['local'] += 100
        shifted['memory'] *= 100
        local, memory = self.p.transform(shifted, normalizer)
        self.assertGreater(float(local.mean()), 20)
        self.assertTrue(np.all(memory[:, 1] == 0))
        self.assertTrue(np.all(memory[:, 3] == 0))
        self.assertTrue(np.any(memory[1::4, 2] != 0))
        self.assertTrue(np.all(memory[::4] == 0))
        for key in before:
            np.testing.assert_array_equal(before[key], normalizer[key])

    def test_donors_match_cross_conversation_without_mutation(self):
        train = fixture()
        saved = {k: v.copy() for k, v in train.items()}
        donors, keep = self.p.choose_donors(train, train, 66)
        self.assertTrue(keep.all())
        for i, j in enumerate(donors):
            if i % 4 == 0:
                self.assertEqual(j, -1)
            else:
                self.assertNotEqual(train['conversation_ids'][i], train['conversation_ids'][j])
                self.assertGreater(train['utterance_indices'][j], 0)
                np.testing.assert_array_equal(train['availability'][i], train['availability'][j])
        for k in saved:
            np.testing.assert_array_equal(saved[k], train[k])
        lonely = {k: v[:4] for k, v in train.items()}
        _, keep = self.p.choose_donors(lonely, lonely, 66)
        np.testing.assert_array_equal(keep, [True, False, False, False])

    def test_targets_are_causal_without_skipping(self):
        data = fixture()
        t = self.p.make_targets(data)
        np.testing.assert_allclose(t['previous'][1:4], [-1, 0, 1])
        np.testing.assert_allclose(t['preceding3_mean'][1:4], [-1, -.5, 0])
        np.testing.assert_allclose(t['current_minus_previous'][1:4], [1, 1, 1])
        self.assertTrue(np.isnan(t['previous'][0]))
        altered = {k: v.copy() for k, v in data.items()}
        altered['labels'][3] = 100
        self.assertEqual(self.p.make_targets(altered)['preceding3_mean'][3], 0)
        gap = {k: np.delete(v, 1, axis=0) for k, v in data.items()}
        self.assertTrue(np.isnan(self.p.make_targets(gap)['previous'][1]))

    def test_finite_fit_artifacts_and_parameter_budgets(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable; execute this test in experiment environment')
        torch.set_num_threads(1)
        splits = dict(train=fixture(), test=fixture(20))
        saved = splits['train']['memory'].copy()
        with tempfile.TemporaryDirectory() as d:
            result = self.p.run_probes(splits, Path(d), seeds=(66,), epochs=2)
            self.assertEqual(result['parameter_counts'], {'A': 33409, 'B': 33089, 'C': 33089})
            self.assertEqual(len(result['runs']), 12)
            for run in result['runs']:
                self.assertEqual(run['selection_split'], 'test')
                self.assertTrue(np.isfinite(run['test']['mse']))
                self.assertTrue(Path(run['weights']).is_file())
                self.assertTrue(Path(run['predictions']).is_file())
            self.assertTrue((Path(d) / 'donors_seed66.json').exists())
        np.testing.assert_array_equal(saved, splits['train']['memory'])

    def test_reject_invalid_split_protocol(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'overlap'):
                self.p.run_probes(dict(train=fixture(), test=fixture()), Path(d), epochs=1)
            for mutation in ('label_shape', 'length', 'empty_availability', 'nonbinary'):
                train = fixture()
                if mutation == 'label_shape':
                    train['labels'] = train['labels'][:, None]
                elif mutation == 'length':
                    train['speaker'] = train['speaker'][:-1]
                elif mutation == 'empty_availability':
                    train['availability'][0] = 0
                else:
                    train['availability'][0, 0] = .5
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    self.p.run_probes(dict(train=train, test=fixture(20)), Path(d), epochs=1)

    def test_test_labels_do_not_change_training_weights(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        features = {'train': np.arange(30, dtype='float32').reshape(10, 3) / 30,
                    'test': np.ones((3, 3), dtype='float32')}
        targets = {'train': np.linspace(-1, 1, 10).astype('float32'),
                   'test': np.array([-1., 0., 1.], dtype='float32')}
        with tempfile.TemporaryDirectory() as d:
            first, _ = self.p._fit(features, targets, 4, 66, 1, 'cpu', Path(d) / 'first')
            changed = {**targets, 'test': targets['test'] + 100}
            second, _ = self.p._fit(features, changed, 4, 66, 1, 'cpu', Path(d) / 'second')
            one = torch.load(first['weights'], weights_only=True)['state_dict']
            two = torch.load(second['weights'], weights_only=True)['state_dict']
            for key in one:
                torch.testing.assert_close(one[key], two[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
