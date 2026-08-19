from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gcnet_modality_jepa.dataloader_iemocap import (
    build_feature_path_index,
    load_or_create_cache,
)


class IEMOCAPCacheTest(unittest.TestCase):
    def test_feature_index_scans_directory_once_and_matches_exact_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "utt_1.npy").touch()
            (root / "utt_10.npy").touch()

            index = build_feature_path_index(root, ["utt_1", "utt_10"])

            self.assertEqual(index["utt_1"].name, "utt_1.npy")
            self.assertEqual(index["utt_10"].name, "utt_10.npy")

    def test_cache_factory_runs_only_once(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "dataset.pkl"

            first = load_or_create_cache(cache_path, lambda: calls.append(1) or {"x": 3})
            second = load_or_create_cache(cache_path, lambda: calls.append(2) or {"x": 4})

        self.assertEqual(first, {"x": 3})
        self.assertEqual(second, {"x": 3})
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
