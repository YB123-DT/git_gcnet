import importlib
from types import SimpleNamespace

import pytest

PACKAGE = 'experiments.osram_frozen_nested_cross_20261025'


def test_exactly_one_seed_three_datasets_five_loso_folds():
    queue = importlib.import_module(PACKAGE+'.scheduler')
    smoke, jobs = queue.jobs_for('smoke'), queue.jobs_for('train')
    assert len(smoke) == 3 and {j['dataset'] for j in smoke} == {'CMUMOSEI','MELD','IEMOCAPSix'}
    assert len(jobs) == len({j['id'] for j in jobs}) == 56
    assert all(j['seed'] == 66 for j in jobs)
    assert sum(j['dataset'] == 'IEMOCAPSix' for j in jobs) == 40
    assert {(j['fold'],j['rate']) for j in jobs if j['dataset'] == 'IEMOCAPSix'} == {
        (f,str(r/10)) for f in range(1,6) for r in range(8)}


def test_launch_uses_the_dataset_numeric_snapshot_and_exact_uuid(tmp_path,monkeypatch):
    queue = importlib.import_module(PACKAGE+'.scheduler')
    launched = []
    def popen(command,**kwargs):
        launched.append((command,kwargs));return SimpleNamespace(pid=123)
    monkeypatch.setattr(queue.subprocess,'Popen',popen)
    monkeypatch.setattr(queue,'source_root',lambda d:'/pinned/'+d)
    args = SimpleNamespace(root=tmp_path,source_commit='locked',resume=False)
    for job in queue.jobs_for('smoke'):
        queue.launch(args,job,'GPU-test','smoke')
        cmd,kw = launched[-1]
        assert PACKAGE+'.run' in cmd and '--smoke' in cmd
        assert cmd[cmd.index('--dataset')+1] == job['dataset']
        assert cmd[cmd.index('--fold')+1] == str(job['fold'])
        assert kw['cwd'] == '/pinned/'+job['dataset']
        assert kw['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-test' and kw['start_new_session']


def test_partial_results_do_not_fabricate_fold_or_seed_means(tmp_path):
    report = importlib.import_module(PACKAGE+'.summarize')
    result = report.summarize(tmp_path,'locked',[dict(id='waiting',status='pending')])
    assert result['status'] == 'incomplete' and result['expected_jobs'] == 56
    assert not result['aggregate'] and result['failures']


def test_gpu4_is_forbidden_and_both_physical_and_uuid_maps_are_checked():
    queue = importlib.import_module(PACKAGE+'.scheduler')
    assert queue.parse_gpu_map('0=GPU-a,1=GPU-b,2=GPU-c') == {0:'GPU-a',1:'GPU-b',2:'GPU-c'}
    for value in ('4=GPU-bad','0=GPU-a,2=GPU-a','3=GPU-d'):
        with pytest.raises(ValueError):queue.parse_gpu_map(value)


def test_concurrency_expands_only_after_real_cyclic_coverage(tmp_path):
    import json
    queue = importlib.import_module(PACKAGE+'.scheduler')
    output = tmp_path/'job';output.mkdir()
    job = dict(dataset='MELD',status='running',output=str(output))
    peaks = {'MELD':dict(gpu_mib=500.,rss_gib=2.)}
    (output/'SUMMARY.json').write_text(json.dumps(dict(peak_memory_mib=900.,peak_rss_gib=3.)))
    (output/'history.json').write_text(json.dumps([dict(training_rates=[0.,.1])]))
    assert not queue.refresh_measurements([job],peaks)['MELD']
    assert peaks['MELD'] == dict(gpu_mib=900.,rss_gib=3.)
    (output/'history.json').write_text(json.dumps([dict(training_rates=[i/10 for i in range(8)])]))
    assert queue.refresh_measurements([job],peaks)['MELD']
