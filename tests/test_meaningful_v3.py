"""Use the existing adapter contract for the incremental source-grounded batch."""
import importlib.util
import unittest
from unittest.mock import patch


class IncrementalCatalogTests(unittest.TestCase):
    def test_registry_and_common_contract(self):
        self.assertIsNotNone(importlib.util.find_spec(
            'gcnet_missing_m3.meaningful_v3_registry'))
        from gcnet_missing_m3.meaningful_v3_registry import V3_METHODS
        from gcnet_missing_m3.meaningful_input import INPUT_FAMILIES
        from tests.test_meaningful_input import InputAdapterTests
        self.assertEqual(tuple(INPUT_FAMILIES['v3']), V3_METHODS)
        self.assertEqual(len(set(V3_METHODS)), len(V3_METHODS))
        self.assertTrue(V3_METHODS)
        old = {name for family, names in INPUT_FAMILIES.items()
               if family != 'v3' for name in names}
        self.assertFalse(old.intersection(V3_METHODS))
        with patch('gcnet_missing_m3.meaningful_input.INPUT_METHODS', V3_METHODS):
            InputAdapterTests('test_shared_contract').test_shared_contract()


if __name__ == '__main__':
    unittest.main()
