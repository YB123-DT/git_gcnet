"""CPU regression tests for delivery-only resume and teacher provenance guards."""
import dataclasses
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dispatch
import train_one


class DeliveryResumeTests(unittest.TestCase):
    def identity(self):
        train = [dict(id='train', utterance_ids=['u0', 'u1'])]
        test = [dict(id='test', utterance_ids=['u0'])]
        meta = dict(selected_features_labels_sha256='content-A',
                    source_fingerprint=dict(mtime_ns=1))
        return meta, train, test

    def test_changed_feature_content_or_split_is_rejected_before_manifest_replacement(self):
        meta, train, test = self.identity()
        previous = dict(code_snapshot='frozen', data_identity=train_one.data_identity(meta, train, test))
        for altered_meta, altered_train, altered_test in [
            (dict(meta, selected_features_labels_sha256='content-B'), train, test),
            (meta, [dict(id='other-train', utterance_ids=['u0', 'u1'])], test),
            (meta, train, [dict(id='test', utterance_ids=['different'])]),
        ]:
            with self.subTest(data=altered_meta, train=altered_train, test=altered_test):
                current = dict(code_snapshot='frozen', data_identity=train_one.data_identity(altered_meta, altered_train, altered_test))
                with self.assertRaisesRegex(RuntimeError, 'data|split'):
                    train_one.validate_resume_identity(previous, current)
        meta['source_fingerprint']['mtime_ns'] = 999
        current = dict(code_snapshot='frozen', data_identity=train_one.data_identity(meta, train, test))
        train_one.validate_resume_identity(previous, current)
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            train_one.validate_resume_identity(dict(code_snapshot='frozen'), current)

    def test_teacher_only_interruption_uses_original_snapshot_and_rejects_config_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; run.mkdir()
            old = root/'old-snapshot'; old.mkdir(); (old/'snapshot.json').write_text('{}')
            new = root/'new-snapshot'; new.mkdir()
            config = dict(model='MMIN', seed=66, teacher_epochs=100)
            config_path = root/'config.json'; config_path.write_text(json.dumps(config))
            launch = dict(config=config, code_snapshot=str(old))
            (run/'launch.json').write_text(json.dumps(launch))
            (run/'manifest.json').write_text(json.dumps(dict(config=dict(config, device='cuda:0', seq_len=None), code_snapshot=str(old))))
            task = dict(run_dir=str(run), config_path=str(config_path), config=config)
            # No missing-model last.pt exists during interrupted teacher training.
            self.assertFalse((run/'last.pt').exists())
            with patch.object(dispatch, 'RUN_ROOT', root/'runs'), \
                 patch.object(dispatch.subprocess, 'Popen', return_value=types.SimpleNamespace(pid=123)) as start:
                dispatch.launch(task, 'GPU-test-only', new)
            self.assertEqual(start.call_args.args[0][1], str(old/'supervise.py'))
            self.assertEqual(start.call_args.kwargs['env']['BASELINE_RUN_ROOT'],str(root/'runs'))
            resumed = json.loads((run/'launch.json').read_text())
            self.assertEqual(resumed['code_snapshot'], str(old))
            task['config'] = dict(config, teacher_epochs=200)
            config_path.write_text(json.dumps(task['config']))
            with patch.object(dispatch.subprocess, 'Popen') as start:
                with self.assertRaisesRegex(RuntimeError, 'config'):
                    dispatch.launch(task, 'GPU-test-only', new)
                start.assert_not_called()

    def test_teacher_cache_identity_covers_optimizer_dependencies_schedule_and_reuses_mr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'adapters/mmin/networks').mkdir(parents=True)
            for path in [root/'train_one.py', root/'runtime.py', root/'common_data.py',
                         root/'adapters/mmin/__init__.py', root/'adapters/mmin/networks/classifier.py']:
                path.write_text('# initial implementation\n')
            meta, train, _ = self.identity()
            config = dict(dataset='MELD', fold=0, seed=66, lr=.001,
                          batch_size=256, teacher_betas=[.9, .999], missing_rate=.3)
            cfg = types.SimpleNamespace()
            with patch.object(train_one, 'ROOT', root), \
                 patch.object(dataclasses, 'asdict', return_value=dict(embedding_dim=128, teacher_dropout=.3)):
                signature = train_one.teacher_signature(config, cfg, meta, train, 100, False)
                other_mr = train_one.teacher_signature(dict(config, missing_rate=.6), cfg, meta, train, 100, False)
                self.assertEqual(signature, other_mr)
                optimizer_change = train_one.teacher_signature(dict(config, teacher_betas=[.5, .999]), cfg, meta, train, 100, False)
                self.assertNotEqual(signature, optimizer_change)
                (root/'adapters/mmin/networks/classifier.py').write_text('# changed classifier\n')
                dependency_change = train_one.teacher_signature(config, cfg, meta, train, 100, False)
                self.assertNotEqual(signature, dependency_change)
                with patch.object(train_one, 'mmin_lr', return_value=.002):
                    schedule_change = train_one.teacher_signature(config, cfg, meta, train, 100, False)
                self.assertNotEqual(dependency_change, schedule_change)

    def test_import_is_host_independent_but_training_entry_rejects_nonlocal_host(self):
        root = Path(train_one.__file__).parent
        result = subprocess.run([sys.executable, '-c',
            "import socket; socket.gethostname=lambda: 'another-host'; import train_one"],
            cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        with patch.object(train_one.socket, 'gethostname', return_value='another-host'), \
             patch.object(train_one.Path, 'mkdir') as mkdir:
            with self.assertRaisesRegex(RuntimeError, 'restricted to local host'):
                train_one.main()
            mkdir.assert_not_called()


if __name__ == '__main__':
    unittest.main()
