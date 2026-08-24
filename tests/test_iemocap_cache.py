import tempfile
import unittest
from pathlib import Path

from dataloader_iemocap import load_or_create_cache


class IEMOCAPCacheTests(unittest.TestCase):
    def test_second_load_reuses_the_atomic_pickle(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "dataset.pkl"

            def factory():
                calls.append("called")
                return {"values": [1, 2, 3]}

            first = load_or_create_cache(cache_path, factory, fingerprint="abc")
            second = load_or_create_cache(cache_path, factory, fingerprint="abc")

        self.assertEqual(first, second)
        self.assertEqual(calls, ["called"])


if __name__ == "__main__":
    unittest.main()
