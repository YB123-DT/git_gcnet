import unittest
from pathlib import Path


class PreparationTests(unittest.TestCase):
    def test_only_implemented_source_accepted_ids_can_be_selected(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / 'experiments/osram_v3_120_20261004/prepare.py').is_file())
        from experiments.osram_v3_120_20261004.prepare import select
        name = 'routing_predinet_bound_predicates'
        result = select(root, [name])
        self.assertEqual(result['cards'][0]['id'], name)
        self.assertEqual(result['status'], 'source_accepted')
        for names in ([], [name, name], ['not_a_real_method'],
                      ['routing_reformer_multihash']):
            with self.assertRaises(ValueError):
                select(root, names)


if __name__ == '__main__':
    unittest.main()
