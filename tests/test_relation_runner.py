import unittest


class RelationRunnerTests(unittest.TestCase):
    def test_only_relation_delta(self):
        from experiments.osram_current_history_relation_20261003.run import relation_config
        base = dict(seed=66, epochs=100, osram_readout_fusion='flat',
                    training_objective='emotion-only', completion_path='none')
        result = relation_config(base)
        self.assertEqual(base, {k: result[k] for k in base})
        self.assertEqual(set(result) - set(base), {
            'osram_relation_block', 'osram_relation_mode',
            'osram_relation_dim', 'osram_relation_out_dim'})
        self.assertTrue(result['osram_relation_block'])
        self.assertEqual(result['osram_relation_mode'], 'pairwise')
        self.assertNotIn('osram_relation_block', base)

    def test_reject_protocol_changes(self):
        from experiments.osram_current_history_relation_20261003.run import relation_config
        base = dict(seed=66, epochs=100, osram_readout_fusion='flat',
                    training_objective='emotion-only', completion_path='none')
        for key, value in [('seed', 67), ('epochs', 50),
                           ('paired_history_views', True),
                           ('osram_readout_fusion', 'memory-shift-residual')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                relation_config(dict(base, **{key: value}))
