"""CPU-only guards for the alternate-feature Flat launcher."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

WORKER = Path(__file__).resolve().parents[1] / 'experiments/osram_meld_alternate_features_20261017/worker.py'


class AlternateWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = None
        if WORKER.exists():
            spec = importlib.util.spec_from_file_location('alternate_worker', WORKER)
            cls.worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.worker)

    def require_worker(self):
        self.assertIsNotNone(self.worker, 'alternate-feature worker must exist')
        return self.worker

    def test_config_requires_exact_flat_configuration(self):
        worker = self.require_worker()
        expected = {'seed': 66, 'epochs': 50, 'osram_meaningful_block': 'none'}
        worker.validate_config(dict(expected), expected)
        for invalid in ({**expected, 'seed': 67}, {**expected, 'extra': False}, {'seed': 66}):
            with self.assertRaisesRegex(ValueError, 'configuration'):
                worker.validate_config(invalid, expected)

    def test_environment_binds_alternate_data_before_import(self):
        worker = self.require_worker()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            data = Path(directory) / 'new.pkl'
            output = Path(directory) / 'run'
            worker.configure_environment(data, output, 'GPU-healthy')
            self.assertEqual(os.environ['GCNET_MELD_PICKLE'], str(data.resolve()))
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-healthy')
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '2')
            self.assertTrue(Path(os.environ['GCNET_CACHE_ROOT']).is_relative_to(output))

    def test_incomplete_cuda_rng_cannot_resume(self):
        worker = self.require_worker()
        for rng in (None, {}, {'cuda': None}, {'cuda': []}):
            with self.assertRaisesRegex(ValueError, 'CUDA RNG'):
                worker.require_cuda_rng({'rng': rng})
        worker.require_cuda_rng({'rng': {'cuda': [object()]}})

    def test_uuid_count_repair_also_updates_captured_random_module_alias(self):
        worker = self.require_worker()
        for initial_count in (0, 1):
            cuda = SimpleNamespace(device_count=lambda: initial_count,
                                   random=SimpleNamespace(device_count=lambda: 0))
            torch = SimpleNamespace(cuda=cuda, _C=SimpleNamespace(_cuda_getDeviceCount=lambda: 1))
            repair = worker.repair_uuid_enumeration(torch)
            self.assertEqual(cuda.device_count(), 1)
            self.assertEqual(cuda.random.device_count(), 1,
                             'Torch1.13 RNG functions capture a separate device_count alias')
            self.assertIs(cuda.random.device_count, cuda.device_count)
            self.assertTrue(repair['cuda_count_pinned'])

    def test_ua_covers_all_seven_classes_and_metrics_use_same_logits(self):
        worker = self.require_worker()
        metrics = worker.class_metrics([0, 0, 1], [[4, 0, 0, 0, 0, 0, 0],
                                                   [0, 4, 0, 0, 0, 0, 0],
                                                   [0, 4, 0, 0, 0, 0, 0]])
        self.assertAlmostEqual(metrics['ACC'], 200 / 3)
        self.assertAlmostEqual(metrics['UA'], 150 / 7)
        self.assertAlmostEqual(metrics['WF1'], 200 / 3)
        self.assertEqual(metrics['support'], [2, 1, 0, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            worker.class_metrics([0], [[float('nan')] * 7])

    def test_source_guard_rejects_changed_core_before_import(self):
        worker = self.require_worker()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'source'):
                worker.validate_source(Path(directory))

    def test_all_eight_best_metrics_preserve_their_selection_artifacts(self):
        import numpy as np
        worker = self.require_worker()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            availability = np.ones((2, 3), dtype=np.float32)
            labels = np.asarray([0, 1])
            logits = np.eye(7)[labels]
            selected = output / 'selection.npz'
            np.savez(selected, labels=labels, logits=logits, availability=availability,
                     selected_epoch=np.asarray(3), predictions=labels)
            checkpoint = dict(epoch=3, selection_accuracy=1., selection_rate=0.,
                selection_artifact=dict(path='selection.npz', sha256=worker.sha256(selected),
                    availability_sha256=worker.hashlib.sha256(availability.tobytes()).hexdigest()))
            for i in range(8):
                (output / f'best_miss_0p{i}.pt').write_bytes(b'checkpoint')
            fake_torch = SimpleNamespace(load=lambda path, **kwargs: {
                **checkpoint, 'selection_rate': int(path.stem[-1]) / 10})
            report = worker.selected_metrics(output, fake_torch)
            self.assertEqual(len(report['rates']), 8)
            self.assertEqual(report['mean']['ACC'], 100.)
            self.assertEqual(report['high']['WF1'], 100.)
            for i in range(8):
                self.assertEqual(worker.sha256(output / f'selected_predictions_miss_0p{i}.npz'),
                                 worker.sha256(selected))
            checkpoint['selection_artifact']['availability_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'mask'):
                worker.selected_metrics(output, fake_torch)

    def test_selection_epoch_and_rate_must_match_checkpoint(self):
        import numpy as np
        worker = self.require_worker()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            selected = output / 'selection.npz'
            np.savez(selected, labels=np.asarray([0]), logits=np.eye(7)[[0]],
                     availability=np.ones((1, 3)), selected_epoch=np.asarray(2))
            checkpoint = dict(epoch=3, selection_accuracy=1., selection_rate=0.,
                selection_artifact=dict(path='selection.npz', sha256=worker.sha256(selected),
                    availability_sha256=worker.hashlib.sha256(np.ones((1, 3)).tobytes()).hexdigest()))
            for i in range(8):
                (output / f'best_miss_0p{i}.pt').write_bytes(b'checkpoint')
            fake_torch = SimpleNamespace(load=lambda *args, **kwargs: checkpoint)
            with self.assertRaisesRegex(ValueError, 'epoch'):
                worker.selected_metrics(output, fake_torch)


if __name__ == '__main__':
    unittest.main()
