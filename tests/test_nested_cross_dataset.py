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

    def test_three_seed_scope_has_27_new_runs_and_no_seed66_duplicates(self):
        mod = self.module()
        rows = mod.additional_tasks()
        self.assertEqual(len(rows), 27)
        self.assertEqual(len({(r['dataset'], r['seed'], r['fold']) for r in rows}), 27)
        self.assertEqual(sum(r['dataset'] == 'IEMOCAPFour' for r in rows), 15)
        self.assertEqual(sum(r['dataset'] == 'IEMOCAPSix' for r in rows), 10)
        self.assertEqual(sum(r['dataset'] == 'CMUMOSEI' for r in rows), 2)
        self.assertTrue(all(r['seed'] in (67, 68) for r in rows if r['dataset'] != 'IEMOCAPFour'))

    def test_reference_and_output_paths_use_exact_seed_and_classes(self):
        mod = self.module()
        task = dict(dataset='IEMOCAPFour', seed=68, fold=4)
        self.assertTrue(str(mod.reference_dir(task)).endswith('iemocap4/seed_68/fold_4'))
        self.assertTrue(str(mod.output_dir(task)).endswith('IEMOCAPFour/seed_68/fold_4'))


if __name__ == '__main__':
    unittest.main()
