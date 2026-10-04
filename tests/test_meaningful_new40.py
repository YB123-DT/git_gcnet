"""Catalog guard; numerical contracts reuse test_meaningful_input.py."""
import importlib.util
import unittest
import json
from pathlib import Path


class NewFortyCatalogTests(unittest.TestCase):
    def test_selection_keeps_training_cap(self):
        self.assertIsNotNone(importlib.util.find_spec(
            'experiments.osram_new40_20261004.prepare'))
        from experiments.osram_new40_20261004.prepare import select
        root = Path(__file__).resolve().parents[1]
        catalog = json.loads((root / 'experiments/osram_new40_20261004/CATALOG.json').read_text())
        registry = json.loads((root / 'experiments/osram_method_registry.json').read_text())
        registry['max_distinct_trained_methods'] = 60
        registry.pop('training_budget_override', None)
        registry['rounds'].pop('4', None)
        names = [c['id'] for c in catalog['cards']]
        manifest = select(catalog, names[:1], registry)
        self.assertEqual([c['id'] for c in manifest['cards']], names[:1])
        for invalid in ([], names[:21], [names[0], names[0]], ['unknown']):
            with self.assertRaises(ValueError):
                select(catalog, invalid, registry)
        registry['rounds']['3'] = {'methods': [{'id': name} for name in names[:20]]}
        with self.assertRaises(ValueError):
            select(catalog, names[20:21], registry)
        self.assertEqual(len(select(catalog, names[:1], registry)['cards']), 1)
        registry['max_distinct_trained_methods'] = 69
        with self.assertRaises(ValueError):
            select(catalog, names[20:21], registry)
        registry['training_budget_override'] = {'max_distinct_trained_methods': 69,
            'user_request': 'GPU2 and GPU6 each eleven experiments, 2026-10-04'}
        self.assertEqual(len(select(catalog, names[20:29], registry)['cards']), 9)
        with self.assertRaises(ValueError):
            select(catalog, names[20:30], registry)

    def test_forty_distinct_real_factories(self):
        self.assertIsNotNone(importlib.util.find_spec(
            'gcnet_missing_m3.meaningful_new40_registry'))
        from gcnet_missing_m3.meaningful_new40_registry import NEW40_METHODS
        from gcnet_missing_m3.meaningful_input import INPUT_FAMILIES, MeaningfulInputAdapter
        from gcnet_missing_m3.meaningful_blocks import ROUND1_METHODS
        old = set(ROUND1_METHODS)
        for family, methods in INPUT_FAMILIES.items():
            if family != 'new40':
                old.update(methods)
        self.assertEqual(len(NEW40_METHODS), 40)
        self.assertEqual(len(set(NEW40_METHODS)), 40)
        self.assertFalse(old.intersection(NEW40_METHODS))
        self.assertEqual(tuple(INPUT_FAMILIES['new40']), tuple(NEW40_METHODS))
        for method in NEW40_METHODS:
            with self.subTest(method=method):
                model = MeaningfulInputAdapter(256, 1024, 1600, method, 8, 64)
                self.assertGreater(sum(p.numel() for p in model.parameters()), 0)


if __name__ == '__main__':
    unittest.main()
