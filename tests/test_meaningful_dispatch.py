import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.osram_meaningful20_round2_20261004 import dispatch


class DispatchTests(unittest.TestCase):
    def controller(self):
        self.assertTrue(hasattr(dispatch, 'Dispatch'), 'missing simultaneous-cohort controller')
        return dispatch.Dispatch(dict(bundle_ids=[f'b{i}' for i in range(12)], other_ids=[f'o{i}' for i in range(8)],
            bundle_gpu_index='3', bundle_gpu_uuid='GPU-three', other_gpu_indices=['0'],
            bundle_concurrency_limit=12, other_concurrency_limit=3, formal_concurrency_limit=15))

    def test_all_twelve_profiles_required_before_bundle_admission(self):
        controller = self.controller()
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(readiness_root=Path(temporary))
            state = dict(jobs={}, phase='screening')
            plan = controller.plan(args, state, {}, [])
            self.assertFalse(plan['ready'])
            self.assertIn('12', state['dispatch_waiting']['bundle'])
            self.assertEqual(controller.limits, dict(bundle=12, other=3, total=15))

    def test_aggregate_peak_and_pending_allocations_are_reserved(self):
        controller = self.controller()
        profiles = {f'b{i}': dict(peak_mib=1000, artifact_gib=1) for i in range(12)}
        plan = dict(ready=True, profiles=profiles, pending=list(profiles), aggregate_peak_mib=12000)
        args = SimpleNamespace(root=Path('/campaign'))
        state = dict(phase='screening')
        gpu = dict(uuid='GPU-three', free_mib=13000, utilization=0, temperature=50)
        with patch.object(dispatch, 'allocated_memory', return_value={}):
            ok, reason = controller.admit('b0', '3', args, state, [], profiles['b0'], plan, gpu, 100)
            self.assertFalse(ok); self.assertIn('aggregate', reason)
            gpu['free_mib'] = 30000
            self.assertTrue(controller.admit('b0', '3', args, state, [], profiles['b0'], plan, gpu, 100)[0])
            active = [dict(candidate=f'b{i}', gpu_index='3', gpu_uuid='GPU-three', pid=i+100, profile=profiles[f'b{i}']) for i in range(6)]
            plan['pending'] = [f'b{i}' for i in range(6,12)]
            gpu['free_mib'] = 20000
            self.assertFalse(controller.admit('b6', '3', args, state, active, profiles['b6'], plan, gpu, 100)[0])
        with patch.object(dispatch, 'allocated_memory', return_value={i+100:('GPU-three',1000) for i in range(6)}):
            self.assertTrue(controller.admit('b6', '3', args, state, active, profiles['b6'], plan, gpu, 100)[0])
        gpu.update(utilization=99, free_mib=30000)
        with patch.object(dispatch, 'allocated_memory', return_value={}):
            ok, reason = controller.admit('b0', '3', args, state, [], profiles['b0'], plan, gpu, 100)
            self.assertFalse(ok); self.assertIn('saturation', reason)

    def test_queue_reaches_twelve_before_any_first_epoch_without_hidden_six_cap(self):
        from experiments.osram_meaningful20_20261003 import queue as q, manifest as m
        controller = self.controller()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = controller.policy['bundle_ids'] + controller.policy['other_ids']
            cards = [dict(id=name, accepted=True, design_sha256=f'{i:064x}',
                          source=dict(paper='paper', code='code')) for i, name in enumerate(ids)]
            manifest, audit, data = (root / name for name in ('ROUND.json','AUDIT.json','DATA.json'))
            m.write(manifest, dict(round_id='round2',status='source_accepted',cards=cards))
            m.write(audit, dict(runs=[dict(seed=seed,mean8=.8) for seed in (66,67,68)])); m.write(data,{})
            profile = dict(peak_mib=1000,artifact_gib=1)
            profiles = {name: profile for name in ids[:12]}
            for name in ids[:12]: m.write(root / 'ready' / name / 'READY.json', dict(snapshot_root=str(root)))
            args = SimpleNamespace(root=root/'queue',manifest=manifest,baseline_audit=audit,data_manifest=data,
                reference_root=root,readiness_root=root/'ready',python='python',max_concurrent=15,once=False,poll_seconds=1)
            gpu = {'3':dict(uuid='GPU-three',free_mib=100000,utilization=0,temperature=50),
                   '4':dict(uuid='GPU-bad',free_mib=100000,utilization=0,temperature=50)}
            def plan(unused_args,state,unused_cards,active):
                state['throughput']={'GPU-three':dict(concurrency_cap=6)}
                running={job['candidate'] for job in active}
                return dict(ready=True,profiles=profiles,pending=[name for name in profiles if name not in running])
            polls = []
            def stop_after_twelve(unused):
                polls.append(1)
                if len(polls)==12: raise StopIteration
            with patch.object(controller,'plan',side_effect=plan), \
                 patch.object(dispatch,'allocated_memory',return_value={}), \
                 patch.object(q,'query_gpus',return_value=gpu), \
                 patch.object(q,'verify_snapshot',return_value={'source_sha256':{'model.py':'hash'}}), \
                 patch.object(q,'validate_readiness',return_value=profile), \
                 patch.object(q.os,'getloadavg',return_value=(0,0,0)), \
                 patch.object(q.shutil,'disk_usage',return_value=SimpleNamespace(free=1000*1024**3)), \
                 patch.object(q.subprocess,'Popen',return_value=SimpleNamespace(pid=os.getpid(),poll=lambda:None)) as popen, \
                 patch.object(q.time,'sleep',side_effect=stop_after_twelve):
                with self.assertRaises(StopIteration):
                    q.coordinate(args,gpu_filter=lambda candidate,index,args:index=='3',dispatch=controller)
            state=m.read(args.root/'QUEUE.json')
            self.assertEqual(popen.call_count,12)
            self.assertEqual(sum(job['status']=='running' for job in state['jobs'].values()),12)
            self.assertEqual(state['dispatch_limits'],dict(bundle=12,other=3,total=15))


if __name__ == '__main__': unittest.main()
