from copy import deepcopy
import unittest

from experiments.osram_meld_b3_structure_hpo_20261016.run import make_config
from experiments.osram_meld_block2_auto_20261014.replicate import make_jobs


class ReplicationTests(unittest.TestCase):
    def source(self):
        from experiments.osram_meld_b3_default43_3seed_20261016.run import TRAINING_SHA
        config = make_config(4, 3)
        config.setdefault('osram_disable_local_skip', False)
        return dict(status='complete', method='B3', config=config,
                    provenance={'git_sha': TRAINING_SHA})

    def test_only_two_new_seeds_and_no_configuration_drift(self):
        from experiments.osram_meld_b3_default43_3seed_20261016.run import prepare
        source = self.source()
        original = deepcopy(source)
        manifest = prepare(source, 'original/SUMMARY.json')
        jobs = make_jobs(manifest['configurations'], '/new')
        self.assertEqual([j['config']['seed'] for j in jobs], [67, 68])
        for job in jobs:
            config = deepcopy(job['config'])
            config['seed'] = 66
            self.assertEqual(config, original['config'])
        self.assertEqual(source, original)

    def test_rejects_other_structure_training_protocol_or_source(self):
        from experiments.osram_meld_b3_default43_3seed_20261016.run import prepare
        for key, value in [('osram_mbt_bottlenecks', 1), ('osram_mbt_rounds', 2),
                           ('learning_rate', .0003), ('osram_disable_local_skip', True),
                           ('missing_mask_protocol', 'conversation-mask-v2')]:
            source = self.source()
            source['config'][key] = value
            with self.assertRaises(ValueError):
                prepare(source, 'original/SUMMARY.json')
        source = self.source()
        source['provenance']['git_sha'] = 'wrong'
        with self.assertRaises(ValueError):
            prepare(source, 'original/SUMMARY.json')


if __name__ == '__main__':
    unittest.main()
