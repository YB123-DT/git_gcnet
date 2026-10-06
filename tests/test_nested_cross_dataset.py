import copy
import importlib
import unittest


class CrossDatasetProtocolTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.osram_nested_cross_dataset_20261006.run')

    def test_exactly_one_mosei_and_five_iemocap_six_folds(self):
        rows = self.module().tasks()
        self.assertEqual([(r['dataset'], r['fold']) for r in rows],
                         [('CMUMOSEI', 1)] + [('IEMOCAPSix', i) for i in range(1, 6)])
        self.assertTrue(all(r['seed'] == 66 for r in rows))

    def test_only_old_nested_switch_changes_reference(self):
        mod = self.module()
        old = dict(mod.REQUIRED, dataset='CMUMOSEI', fold=1, seed=66,
                   osram_meaningful_block='none', train_missing_rates=None)
        before = copy.deepcopy(old)
        new = mod.configuration(old, mod.tasks()[0])
        self.assertEqual(old, before)
        self.assertEqual({k: v for k, v in new.items() if v != old[k]},
                         {'osram_meaningful_block': 'nested_gnn_rooted_evidence'})

    def test_reference_protocol_and_fold_mismatch_rejected(self):
        mod = self.module()
        old = dict(mod.REQUIRED, dataset='CMUMOSEI', fold=1, seed=66)
        for update in ({'epochs': 99}, {'fold': 2}, {'osram_meaningful_block': 'nested_local8_evidence'},
                       {'paired_history_views': True}, {'train_missing_rates': [0.7]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                mod.configuration(dict(old, **update), mod.tasks()[0])


if __name__ == '__main__':
    unittest.main()
