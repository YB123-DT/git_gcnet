from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gcnet_modality_jepa.dataloader_iemocap import (
    build_feature_path_index,
    load_or_create_cache,
)
from gcnet_modality_jepa.dataloader_cmumosi import load_cmumosi_dataset


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

    def test_cmu_dataset_factory_is_cached_by_source_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = root / "labels.pkl"
            label.write_bytes(b"labels")
            feature_roots = []
            for name in ("audio", "text", "visual"):
                path = root / name
                path.mkdir()
                feature_roots.append(str(path))
            sentinel = {"cached": True}
            with mock.patch.dict(
                os.environ, {"GCNET_CACHE_ROOT": str(root / "cache")}
            ), mock.patch(
                "gcnet_modality_jepa.dataloader_cmumosi.CMUMOSIDataset",
                return_value=sentinel,
            ) as constructor:
                first = load_cmumosi_dataset(
                    str(label), *feature_roots, dataset_name="CMUMOSI"
                )
                second = load_cmumosi_dataset(
                    str(label), *feature_roots, dataset_name="CMUMOSI"
                )

        self.assertEqual(first, sentinel)
        self.assertEqual(second, sentinel)
        constructor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
