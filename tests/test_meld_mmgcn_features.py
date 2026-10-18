import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


FILE = Path(__file__).resolve().parents[1] / 'experiments/osram_meld_mmgcn_features_20261018/prepare.py'


def fixtures():
    ids = [0, 1, 2]
    speaker = {k: [[1, 0]] for k in ids}
    labels = {k: [k] for k in ids}
    text = {k: np.full((1, 4), k + 1, dtype=np.float32) for k in ids}
    audio = {k: np.full((1, 2), k + 1, dtype=np.float32) for k in ids}
    visual = {k: np.zeros((1, 3), dtype=np.float64) for k in ids}
    sentences = {k: [f'sentence {k}'] for k in ids}
    canonical = [speaker, labels, labels] + [text] * 8 + [audio, visual, sentences, [0], [2], [1]]
    source_text = {k: np.full((1, 5), k + 1, dtype=np.float32) for k in ids}
    source = [{k: [0] for k in ids}, speaker, labels, source_text, audio, visual, sentences, {0, 1}, {2}, None]
    return canonical, source


class MMGCNFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FILE.exists():
            raise AssertionError('MMGCN audit/conversion is not implemented')
        spec = importlib.util.spec_from_file_location('mmgcn_prepare', FILE)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_only_features_change_and_dev_is_preserved(self):
        c, s = fixtures()
        converted, audit = self.module.compare_and_convert(c, s)
        for field in [0, 1, 2, 13, 14, 15, 16]:
            self.assertIs(converted[field], c[field])
        np.testing.assert_array_equal(converted[3][0], s[3][0])
        self.assertEqual(audit['dimensions']['text'], {'source': 5, 'ecerc': 4})
        self.assertFalse(audit['all_training_float32_features_equal'])
        self.assertEqual(audit['split_counts'], {'train': 1, 'dev': 1, 'test': 1})
        for name in ['audio', 'visual']:
            self.assertEqual(audit['by_split']['train'][name]['equal_float32_N'], 1)

    def test_rejects_changed_metadata_or_missing_dialogue(self):
        c, s = fixtures()
        broken = copy.deepcopy(s)
        broken[2][0] = [6]
        with self.assertRaises(ValueError):
            self.module.compare_and_convert(c, broken)
        broken = copy.deepcopy(s)
        del broken[5][1]
        with self.assertRaises(ValueError):
            self.module.compare_and_convert(c, broken)
        broken = copy.deepcopy(s)
        broken[8] = {1, 2}
        with self.assertRaises(ValueError):
            self.module.compare_and_convert(c, broken)

    def test_identical_training_arrays_do_not_require_new_training(self):
        c, s = fixtures()
        s[3] = c[3]
        _, audit = self.module.compare_and_convert(c, s)
        self.assertTrue(audit['all_training_float32_features_equal'])
        s[4][0] = np.array([[np.nan, 0]])
        with self.assertRaises(ValueError):
            self.module.compare_and_convert(c, s)


if __name__ == '__main__':
    unittest.main()
