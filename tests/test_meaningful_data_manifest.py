import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class DataManifestTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.osram_meaningful20_20261003.manifest')

    def build(self, root, output):
        with patch('sys.argv',['manifest','data','--dataset-root',str(root),'--output',str(output)]):
            self.module().main()

    def fixture(self, root):
        for name in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT'):
            path=root/'CMUMOSI'/'features'/name
            path.mkdir(parents=True)
            (path/'a.bin').write_bytes(b'feature')
        (root/'CMUMOSI'/'CMUMOSI_features_raw_2way.pkl').write_bytes(b'labels and splits')

    def test_build_validates_hashes_and_never_overwrites(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'dataset'; self.fixture(root)
            output=Path(tmp)/'manifest.json'
            self.build(root,output)
            data=json.loads(output.read_text())
            self.assertEqual(len(data['files']),4)
            self.assertEqual(len(data['feature_roots']),3)
            from experiments.osram_meaningful20_20261003.run import validate_data_manifest
            validate_data_manifest(output)
            with self.assertRaises(FileExistsError): self.build(root,output)
            Path(data['split_files'][0]).write_bytes(b'changed')
            with self.assertRaises(ValueError): validate_data_manifest(output)

    def test_missing_canonical_file_rejected_without_output(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'dataset'; self.fixture(root)
            (root/'CMUMOSI'/'CMUMOSI_features_raw_2way.pkl').unlink()
            output=Path(tmp)/'manifest.json'
            with self.assertRaises(ValueError): self.build(root,output)
            self.assertFalse(output.exists())

    def test_symlink_escape_rejected(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'dataset'; self.fixture(root)
            outside=Path(tmp)/'outside'; outside.write_bytes(b'outside')
            (root/'CMUMOSI'/'features'/'manet_UTT'/'escape').symlink_to(outside)
            with self.assertRaises(ValueError): self.build(root,Path(tmp)/'manifest.json')


if __name__=='__main__': unittest.main()
