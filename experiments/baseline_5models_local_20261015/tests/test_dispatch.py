"""Local queue boundary tests, using CPU subprocesses and mocked GPU queries only.

Run directly with multimodalerc310 Python; pytest is not needed. Persisted result
fixtures below are synthetic verifier contracts and never reported as experiments.
"""
import contextlib
import json
import os
from pathlib import Path
import socket
import select
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dispatch
import runtime
from common_data import fixed_masks


class LocalQueueBoundary(unittest.TestCase):
    def setUp(self):
        # Keep scratch data on the experiment disk and remove only this test's data.
        scratch = runtime.RUN_ROOT.parent
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='dispatch-unittest-', dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def write_json(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def queue(self):
        source = self.base/'code'; source.mkdir()
        runs = self.base/'runs'; runs.mkdir()
        config = dict(model='GCNet', dataset='MELD', seed=66, fold=0,
                      epochs=3, stage_epoch=0, missing_rate=.3)
        config_path = source/'configs'/'task.json'
        self.write_json(config_path, config)
        task = dict(id='synthetic-task', priority=1, gate='approved', phase='formal',
                    run_dir=str(runs/'task'), config=config, config_path=str(config_path))
        self.write_json(source/'experiment_registry.json', {'tasks':[task]})
        self.write_json(source/'gates.json', {'approved':{'ready':True}})
        return source, runs, task

    @contextlib.contextmanager
    def queue_environment(self, source, runs):
        with mock.patch.multiple(dispatch, ROOT=source, RUN_ROOT=runs, GPUS=['GPU-test-a','GPU-test-b']), \
             mock.patch.object(dispatch,'enforce_local'), \
             mock.patch.object(dispatch,'alive',return_value=False), \
             mock.patch.object(dispatch,'budget_decision',return_value=False), \
             mock.patch.object(sys,'argv',['dispatch.py','--once']):
            yield

    def test_nonlocal_host_rejected_before_gpu_or_environment_mutation(self):
        with mock.patch.object(runtime.socket,'gethostname',return_value='not-authorized-local-host'), \
             mock.patch.object(runtime.subprocess,'check_output') as query, \
             mock.patch.dict(os.environ,{'WANDB_MODE':'original-value'}):
            with self.assertRaises(RuntimeError):
                runtime.gpu_free('GPU-test-a')
            query.assert_not_called()
            self.assertEqual(os.environ['WANDB_MODE'],'original-value')

    def test_gpu_uuid_occupied_by_any_pid_is_not_idle(self):
        rows = '12345, GPU-test-a\n333, GPU-test-b\n'
        with mock.patch.object(runtime,'enforce_local'), \
             mock.patch.object(runtime.subprocess,'check_output',return_value=rows) as query:
            self.assertFalse(runtime.gpu_free('GPU-test-a'))
            self.assertEqual(query.call_count,1)

    def test_high_memory_or_utilization_without_reported_compute_pid_is_not_idle(self):
        for row in ('300, 0','0, 80'):
            with self.subTest(row=row), mock.patch.object(runtime,'enforce_local'), \
                 mock.patch.object(runtime.subprocess,'check_output',side_effect=['',row]):
                self.assertFalse(runtime.gpu_free('GPU-test-a'))

    def test_busy_uuids_never_launch_even_an_approved_task(self):
        source,runs,task = self.queue()
        with self.queue_environment(source,runs), \
             mock.patch.object(dispatch,'valid_result',return_value=False), \
             mock.patch.object(dispatch,'gpu_free',return_value=False), \
             mock.patch.object(dispatch,'launch') as launch, \
             mock.patch.object(dispatch,'snapshot') as snapshot:
            dispatch.main()
        launch.assert_not_called(); snapshot.assert_not_called()
        state=json.loads((runs/'dispatcher.json').read_text())
        self.assertEqual(state['state'],'WAITING')
        self.assertEqual(state['remaining'],1)

    def test_incompatible_completed_config_cannot_be_silently_skipped(self):
        source,runs,task = self.queue()
        wrong=dict(task['config'],epochs=99)
        self.write_json(Path(task['run_dir'])/'config.json',wrong)
        with self.queue_environment(source,runs), \
             mock.patch.object(dispatch,'valid_result',return_value=True), \
             mock.patch.object(dispatch,'launch') as launch:
            with self.assertRaisesRegex(RuntimeError,'incompatible'):
                dispatch.main()
        launch.assert_not_called()

    def test_consumed_adapter_json_changes_snapshot_identity_and_copied_content(self):
        source=self.base/'source';source.mkdir()
        (source/'train_one.py').write_text('raise SystemExit(0)\n')
        metadata=source/'adapters'/'mmin'/'teacher_config.json'
        self.write_json(metadata,{'encoder_dims':[3,4,2]})
        runs=self.base/'runs';runs.mkdir()
        signature=lambda root=None:runtime.code_signature(source if root is None else root)
        error=subprocess.CalledProcessError(1,['git'])
        with mock.patch.multiple(dispatch,ROOT=source,RUN_ROOT=runs), \
             mock.patch.object(dispatch,'code_signature',side_effect=signature), \
             mock.patch.object(dispatch.subprocess,'check_output',side_effect=error):
            snapshot_a,key_a=dispatch.snapshot()
            self.write_json(metadata,{'encoder_dims':[3,4,3]})
            snapshot_b,key_b=dispatch.snapshot()
        self.assertNotEqual(key_a,key_b,'Runtime-consumed JSON must invalidate the frozen source identity')
        self.assertEqual(json.loads((snapshot_a/'adapters/mmin/teacher_config.json').read_text()),{'encoder_dims':[3,4,2]})
        self.assertEqual(json.loads((snapshot_b/'adapters/mmin/teacher_config.json').read_text()),{'encoder_dims':[3,4,3]})
        self.assertIn('adapters/mmin/teacher_config.json',json.loads((snapshot_a/'snapshot.json').read_text())['files'])

    def start_owned_cpu_process(self, run):
        process=subprocess.Popen([sys.executable,'-c','import time;print("READY",flush=True);time.sleep(20)','--run-dir',str(run)],
                                 stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
        def cleanup():
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            process.stdout.close()
        self.addCleanup(cleanup)
        ready,_,_=select.select([process.stdout],[],[],5)
        self.assertTrue(ready,'CPU test child did not reach exec readiness')
        self.assertEqual(process.stdout.readline().strip(),'READY')
        return process

    def test_alive_requires_this_run_and_rejects_dead_pid(self):
        run=self.base/'task'
        child=self.start_owned_cpu_process(run)
        self.assertTrue(dispatch.alive(child.pid,run))
        self.assertFalse(dispatch.alive(child.pid,self.base/'unrelated'))
        child.terminate();child.wait(timeout=5)
        self.assertFalse(dispatch.alive(child.pid,run))

    def test_run_path_prefix_does_not_own_a_sibling_process(self):
        child=self.start_owned_cpu_process(self.base/'task-other')
        self.assertFalse(dispatch.alive(child.pid,self.base/'task'),
                         'Substring ownership would incorrectly reserve the sibling run and GPU')

    def test_supervisor_persists_distinct_child_pid_and_nonzero_exit(self):
        if socket.gethostname()!=runtime.HOST:
            self.skipTest('Supervisor intentionally refuses this non-local host')
        run=self.base/'owned';run.mkdir()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-test-no-device',PYTHONDONTWRITEBYTECODE='1')
        command=[sys.executable,str(ROOT/'supervise.py'),'--run-dir',str(run),'--',
                 sys.executable,'-c','import time;time.sleep(1);raise SystemExit(7)','--run-dir',str(run)]
        process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        def cleanup():
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=5)
        self.addCleanup(cleanup)
        deadline=time.monotonic()+5
        while not (run/'child.json').exists() and process.poll() is None and time.monotonic()<deadline:
            time.sleep(.02)
        self.assertTrue((run/'child.json').exists(),'Supervisor did not record its child')
        child=json.loads((run/'child.json').read_text())
        self.assertEqual(child['supervisor_pid'],process.pid)
        self.assertNotEqual(child['pid'],process.pid)
        self.assertTrue(dispatch.alive(process.pid,run))
        self.assertTrue(dispatch.alive(child['pid'],run))
        process.communicate(timeout=10)
        self.assertEqual(process.returncode,7)
        self.assertEqual(json.loads((run/'exit.json').read_text())['exit_code'],7)
        status=json.loads((run/'status.json').read_text())
        self.assertEqual(status['state'],'FAILED')
        self.assertEqual(status['gpu_uuid'],'GPU-test-no-device')
        self.assertFalse(dispatch.alive(child['pid'],run))

    def completed_contract(self):
        """Minimal synthetic artifact contract, not an actual trained checkpoint."""
        run=self.base/'artifact';run.mkdir()
        config=dict(dataset='MELD',model='ComP',model_version='causal',seed=66,fold=0,
                    missing_rate=.3,eval_mask_seed=20261015,n_classes=2,epochs=3,stage_epoch=1,
                    checkpoint_selection='test_best_acc',evaluation_protocol='test_oracle')
        utterances=['u0','u1','u2']
        labels=np.array([0,1,0]);prediction=np.array([0,1,1])
        ids=np.array(['dialogue::'+u for u in utterances])
        masks=fixed_masks('MELD','dialogue',utterances,.3,20261015)
        np.savez(run/'predictions.npz',labels=labels,predictions=prediction,sample_ids=ids,masks=masks)
        (run/'best.pt').write_bytes(b'synthetic-unittest-contract-only'*10)
        scores=runtime.compute_metrics(labels,prediction,2)
        manifest=dict(result_type='formal',config=config,expected_n_test=3,
                      expected_sample_ids=ids.tolist(),mask_key_records=[dict(id='dialogue',utterance_ids=utterances)])
        artifacts={'config.json':config,'manifest.json':manifest,
                   'status.json':dict(state='COMPLETED',exit_code=0,epochs=3,result_type='formal'),
                   'metrics.json':dict(scores,best_epoch=2,n_test=3,checkpoint_selection='test_best_acc',evaluation_protocol='test_oracle'),
                   'best.json':dict(epoch=2,checkpoint_sha256=runtime.digest(run/'best.pt'),predictions_sha256=runtime.digest(run/'predictions.npz')),
                   'checkpoint_replay.json':dict(verified=True,epoch=2)}
        for name,value in artifacts.items():self.write_json(run/name,value)
        self.assertTrue(runtime.valid_result(run),'Baseline synthetic fixture no longer matches the verifier contract; update fixture explicitly')
        return run,artifacts

    def test_claimed_completed_dynamic_or_tampered_mask_is_rejected(self):
        run,artifacts=self.completed_contract()
        with np.load(run/'predictions.npz') as saved:arrays={k:saved[k] for k in saved.files}
        arrays['masks']=arrays['masks'].copy();arrays['masks'][0]=[0,0,0]
        np.savez(run/'predictions.npz',**arrays)
        # Refresh checksum so only semantic fixed-mask validity determines refusal.
        artifacts['best.json']['predictions_sha256']=runtime.digest(run/'predictions.npz')
        self.write_json(run/'best.json',artifacts['best.json'])
        self.assertFalse(runtime.valid_result(run))

    def test_claimed_completed_without_checkpoint_replay_is_rejected(self):
        run,artifacts=self.completed_contract()
        (run/'checkpoint_replay.json').unlink()
        self.assertFalse(runtime.valid_result(run))

    def test_claimed_completed_with_wrong_replay_epoch_is_rejected(self):
        run,artifacts=self.completed_contract()
        self.write_json(run/'checkpoint_replay.json',dict(verified=True,epoch=1))
        self.assertFalse(runtime.valid_result(run))

    def test_claimed_completed_with_partial_training_is_rejected(self):
        run,artifacts=self.completed_contract()
        self.write_json(run/'status.json',dict(artifacts['status.json'],epochs=2))
        self.assertFalse(runtime.valid_result(run))

    def test_best_epoch_cannot_exceed_the_completed_training_budget(self):
        run,artifacts=self.completed_contract()
        for name,field in [('metrics.json','best_epoch'),('best.json','epoch'),('checkpoint_replay.json','epoch')]:
            artifacts[name][field]=4;self.write_json(run/name,artifacts[name])
        self.assertFalse(runtime.valid_result(run),'An epoch beyond config.epochs cannot be a real selected checkpoint')


if __name__=='__main__':
    unittest.main(verbosity=2)
