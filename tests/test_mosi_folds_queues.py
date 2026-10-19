"""Finite scope, resource admission, dependency and paired-report contracts."""
import json

import pytest


GPUS = ('GPU-a', 'GPU-b', 'GPU-c')


def test_mosi_plan_exactly_six_new_gcnet_jobs(tmp_path):
    from experiments.osram_mosi_gcnet_router_3seed_20261019.run import plan_jobs
    jobs = plan_jobs(tmp_path, 'sha', GPUS)
    assert len(jobs) == 6
    assert {(j['seed'], j['mode']) for j in jobs} == {(s, m) for s in (66, 67, 68) for m in ('flat', 'joint')}
    assert len({j['output'] for j in jobs}) == 6
    assert all(j['mask_protocol'] == 'gcnet-random-mask-v1' for j in jobs)
    assert all(j['dataset'] == 'CMUMOSI' for j in jobs)
    assert all(j['command'][j['command'].index('--gpu-uuid') + 1] == j['gpu_uuid'] for j in jobs)
    assert all(sum(k['gpu_uuid'] == g for k in jobs) == 2 for g in GPUS)


def test_iemocap_plan_reuses_fold5_and_adds_only_24(tmp_path):
    from experiments.osram_iemocap6_router_fivefold_20261019.run import plan_jobs, fold5_references
    jobs = plan_jobs(tmp_path, 'sha', GPUS)
    assert len(jobs) == 24
    assert {j['fold'] for j in jobs} == {1, 2, 3, 4}
    assert len({j['output'] for j in jobs}) == 24
    assert all('--fold' in j['command'] for j in jobs)
    refs = fold5_references()
    assert len(refs) == 6 and all(j['fold'] == 5 for j in refs)


def test_dependency_fails_closed_and_only_complete_admits(tmp_path):
    from experiments.osram_mosi_gcnet_router_3seed_20261019.run import dependency_state
    path = tmp_path/'QUEUE.json'
    assert dependency_state(path) == 'waiting'
    path.write_text(json.dumps({'status': 'training'}))
    assert dependency_state(path) == 'waiting'
    path.write_text(json.dumps({'status': 'failed'}))
    with pytest.raises(RuntimeError, match='upstream'):
        dependency_state(path)
    path.write_text(json.dumps({'status': 'complete'}))
    assert dependency_state(path) == 'ready'


def test_gpu4_is_forbidden_and_capacity_can_reduce(monkeypatch):
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import run
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *a, **k: '4, GPU-a, 0, 32768\n')
    with pytest.raises(ValueError, match='forbidden'):
        run.devices(['GPU-a'])
    found = {g: dict(total_mib=32768, used_mib=0) for g in GPUS}
    assert run.gpu_capacities(found, GPUS, 6000) == {g: 2 for g in GPUS}
    found['GPU-b']['used_mib'] = 20000
    assert run.gpu_capacities(found, GPUS, 6000)['GPU-b'] == 1
    found['GPU-b']['used_mib'] = 31000
    assert run.gpu_capacities(found, GPUS, 6000)['GPU-b'] == 0


def test_completion_requires_valid_summary(tmp_path):
    from experiments.osram_mosi_gcnet_router_3seed_20261019.run import validate_summary
    job = dict(output=str(tmp_path), dataset='MOSI', seed=66, mode='flat', phase='train', source_commit='sha')
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_summary(job)
    payload = dict(status='complete', phase='train', dataset='MOSI', seed=66, mode='flat', source_commit='sha', aggregate={}, evaluation={})
    (tmp_path/'SUMMARY.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='eight'):
        validate_summary(job)


def test_report_rejects_incomplete_and_mismatched_pairs(tmp_path):
    from experiments.osram_mosi_gcnet_router_3seed_20261019.report import validate_pair, write_report
    with pytest.raises(ValueError, match='six'):
        write_report(tmp_path, {'jobs': []})
    a = dict(initial_backbone_sha256='a', identity={'config': {}, 'split': {}, 'source_hashes': {}},
             mask_hashes={'test': {}, 'validation': {}})
    b = dict(a, initial_backbone_sha256='b')
    with pytest.raises(ValueError, match='initial'):
        validate_pair(a, b, [], [])


def fake_run(job, *, epochs):
    """Minimal real on-disk predictions+history, not an invented finished queue."""
    import numpy as np
    from experiments.osram_mosi_gcnet_router_3seed_20261019.report import CORE, RATES
    output = __import__('pathlib').Path(job['output'])
    output.mkdir(parents=True)
    evaluation = output/'evaluation'
    evaluation.mkdir()
    keys = ('ACC', 'WF1', 'MSE') if job['dataset'] == 'CMUMOSI' else ('ACC', 'WF1', 'UA', 'CE')
    score = 50 + job['seed']/100 + (job.get('fold') or 0)/10 + (job['mode'] == 'joint')
    metrics = {key: score for key in keys}
    masks = {rate: 'mask' for rate in RATES}
    identity = dict(config=dict(seed=job['seed'], fold=job.get('fold') or 1,
                               epochs=epochs, missing_mask_protocol='gcnet-random-mask-v1'),
                    split={'fixed': True}, source_hashes={name: 'core' for name in CORE},
                    validation_mask_hashes=masks, test_masks=masks)
    summary = dict(status='complete', phase='train', seed=job['seed'], dataset=job['dataset'],
                   mode=job['mode'], source_commit='sha', epoch=epochs, selected_epoch=3,
                   initial_backbone_sha256=f'init{job["seed"]}', identity=identity,
                   aggregate={'router': {'mean': metrics, 'high': metrics}},
                   evaluation={rate: {'router': metrics} for rate in RATES})
    if job['dataset'] == 'CMUMOSI':
        summary['mask_hashes'] = dict(validation=masks, test=masks)
    (output/'SUMMARY.json').write_text(json.dumps(summary))
    (output/'history.json').write_text(json.dumps([{'mask_trace_sha256': 'same'}]*epochs))
    for rate in RATES:
        np.savez(evaluation/f'random_{rate.replace(".", "p")}.npz', labels=np.array([1, 0]),
                 speakers=np.array([0, 1]), conversation_ids=np.array(['a', 'a']),
                 utterance_indices=np.array([0, 1]), availability=np.ones((2, 3)))
    job['status'] = 'complete'
    return summary


def test_mosi_complete_report_and_cross_seed_masks(tmp_path):
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import run, report
    jobs = run.plan_jobs(tmp_path, 'sha', GPUS)
    for job in jobs:
        fake_run(job, epochs=150)
        run.validate_summary(job)  # actual worker's CMUMOSI tag
    report.write_report(tmp_path, {'jobs': jobs})
    value = json.loads((tmp_path/'SUMMARY.json').read_text())
    assert value['aggregate']['joint']['mean']['WF1']['mean'] - value['aggregate']['flat']['mean']['WF1']['mean'] == 1
    assert len((tmp_path/'METRICS.csv').read_text().splitlines()) == 49
    for job in jobs:
        if job['seed'] == 68:
            path = __import__('pathlib').Path(job['output'])/'SUMMARY.json'
            summary = json.loads(path.read_text())
            summary['mask_hashes']['test']['0.1'] = 'changed'
            path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='across seeds'):
        report.write_report(tmp_path, {'jobs': jobs})


def test_iemocap_complete_report_reuses_30_runs(tmp_path):
    from experiments.osram_iemocap6_router_fivefold_20261019 import run, report
    jobs = run.plan_jobs(tmp_path, 'sha', GPUS)
    refs = []
    for seed in (66, 67, 68):
        for mode in ('flat', 'joint'):
            refs.append(dict(dataset='IEMOCAPSix', fold=5, seed=seed, mode=mode, reused=True,
                             output=str(tmp_path/'reused'/f'seed_{seed}'/mode)))
    for job in jobs + refs:
        fake_run(job, epochs=100)
    report.write_report(tmp_path, {'jobs': jobs, 'references': refs})
    value = json.loads((tmp_path/'SUMMARY.json').read_text())
    assert len(value['reused_fold5']) == 6
    assert len((tmp_path/'METRICS.csv').read_text().splitlines()) == 241
    assert value['aggregate']['joint']['mean']['ACC']['mean'] - value['aggregate']['flat']['mean']['ACC']['mean'] == 1
    assert len(value['runs']) == 5


def test_controller_runs_one_joint_smoke_and_bounded_six_jobs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import run
    events, running, peak_running = [], {}, {g: 0 for g in GPUS}
    found = {g: dict(index=i, total_mib=32768, used_mib=0) for i, g in enumerate(GPUS)}
    monkeypatch.setattr(run, 'devices', lambda uuids: found)
    monkeypatch.setattr(run, 'available_gib', lambda: 500)
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(run.time, 'sleep', lambda delay: None)
    monkeypatch.setattr(run, 'validate_summary', lambda job: dict(peak_memory_mib=1000))

    class Process:
        def __init__(self, job):
            self.pid = len(events) + 100
            self.job = job

        def poll(self):
            running[self.job['gpu_uuid']] -= 1
            return 0

    def launch(job):
        assert job['command'][2] == run.MODULE  # smoke module must not be '--phase'
        events.append((job['phase'], job['mode']))
        uuid = job['gpu_uuid']
        running[uuid] = running.get(uuid, 0) + 1
        peak_running[uuid] = max(peak_running[uuid], running[uuid])
        job.update(status='running', pid=len(events) + 100)
        return Process(job)

    monkeypatch.setattr(run, 'launch', launch)
    args = SimpleNamespace(output=tmp_path/'queue', resume=False, source_commit='sha', gpus=list(GPUS),
                           after_queue=None, poll_seconds=1)
    reports = []
    run.run_queue(args, run.plan_jobs(args.output, 'sha', GPUS), lambda output, state: reports.append(state['jobs']))
    assert events[0] == ('smoke', 'joint') and len(events) == 7
    assert all(peak == 2 for peak in peak_running.values())
    assert len(reports) == 1
    state = json.loads((args.output/'QUEUE.json').read_text())
    assert state['status'] == 'complete'


def test_controller_upstream_failure_never_launches(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import run
    upstream = tmp_path/'upstream.json'
    upstream.write_text(json.dumps({'status': 'failed'}))
    monkeypatch.setattr(run, 'launch', lambda job: pytest.fail('must not launch'))
    args = SimpleNamespace(output=tmp_path/'queue', resume=False, source_commit='sha', gpus=list(GPUS),
                           after_queue=upstream, poll_seconds=1)
    with pytest.raises(RuntimeError, match='upstream'):
        run.run_queue(args, run.plan_jobs(args.output, 'sha', GPUS), lambda *a: None)
    assert json.loads((args.output/'QUEUE.json').read_text())['status'] == 'failed'
