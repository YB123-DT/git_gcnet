from pathlib import Path
from types import SimpleNamespace
import json

import pytest


def test_only_missing_24_runs_no_seed66_duplicates():
    from experiments.osram_v3_three_seed_20261021.run import plan_jobs
    jobs = plan_jobs(Path('/tmp/supplement'))
    assert len(jobs) == 24
    assert {j['seed'] for j in jobs} == {67, 68}
    for seed in (67, 68):
        same = [j for j in jobs if j['seed'] == seed]
        assert len(same) == 12
        assert sum(j['dataset'] == 'MELD' for j in same) == 1
        assert sum(j['dataset'] == 'CMUMOSEI' for j in same) == 1
        for dataset in ('IEMOCAPFour', 'IEMOCAPSix'):
            assert {j['fold'] for j in same if j['dataset'] == dataset} == {1, 2, 3, 4, 5}
    assert len({j['output'] for j in jobs}) == 24


def test_per_dataset_original_python_and_actual_seed():
    from experiments.osram_v3_three_seed_20261021.run import worker_command, plan_jobs
    args = SimpleNamespace(source_commit='frozen')
    jobs = plan_jobs(Path('/tmp/supplement'))
    for dataset, python in (('MELD', '/data2/yb/conda_envs/cmerc-repro/bin/python'),
            ('CMUMOSEI', '/data2/yb/reproduction_workspace/envs/s0/bin/python')):
        job = next(j for j in jobs if j['dataset'] == dataset and j['seed'] == 68)
        cmd = worker_command(args, job, 'GPU-healthy')
        assert cmd[0] == python
        assert cmd[cmd.index('--seed')+1] == '68'
        assert cmd[cmd.index('--output')+1] == job['output']


def test_do_not_overlap_legacy_controller_caps():
    from experiments.osram_v3_three_seed_20261021.run import dependency_complete
    assert not dependency_complete({'status': 'running', 'jobs': []})
    assert not dependency_complete({'status': 'complete', 'jobs': [{'status': 'running'}]})
    assert dependency_complete({'status': 'complete', 'jobs': [{'status': 'complete'}]})
    with pytest.raises(RuntimeError):
        dependency_complete({'status': 'failed', 'jobs': []})


def test_smoke_records_reused_without_new_training():
    from experiments.osram_v3_three_seed_20261021.run import measured_peaks
    rows = [{'dataset': d, 'status': 'complete', 'returncode': 0, 'peak_memory_mib': n}
        for d, n in (('MELD', 1600), ('IEMOCAPFour', 5300), ('IEMOCAPSix', 7200))]
    peaks = measured_peaks({'smoke': rows}, {'smoke': [dict(dataset='CMUMOSEI',
        status='complete', returncode=0, peak_memory_mib=2300)]})
    assert peaks['CMUMOSEI'] == 2300 and peaks['IEMOCAPSix'] == 7200
    with pytest.raises(ValueError):
        measured_peaks({'smoke': []}, {'smoke': []})


def test_report_requires_all_three_complete_seeds(tmp_path):
    from experiments.osram_v3_three_seed_20261021.report import summarize, RATES
    jobs=[]
    for seed in (66,67,68):
        output=tmp_path/f'seed_{seed}';output.mkdir()
        summary=dict(status='complete' if seed!=68 else 'running', phase='train',
            dataset='MELD',seed=seed,epoch=50,config=dict(osram_source_role_v3=True,
                dataset='MELD',seed=seed),evaluation={r:dict(router=dict(ACC=seed,UA=30.,WF1=50.,CE=.8)) for r in RATES},
            best_by_rate={r:dict(epoch=1) for r in RATES})
        (output/'SUMMARY.json').write_text(json.dumps(summary))
        jobs.append(dict(dataset='MELD',fold=1,seed=seed,output=str(output),status=summary['status']))
    state=dict(status='running',jobs=jobs[1:])
    result=summarize(tmp_path,state,dict(jobs=jobs[:1]),dict(jobs=[]))
    assert not result['three_seed'] and {r['seed'] for r in result['per_seed']} == {66,67}
    path=tmp_path/'seed_68/SUMMARY.json';last=json.loads(path.read_text());last['status']='complete';path.write_text(json.dumps(last))
    result=summarize(tmp_path,state,dict(jobs=jobs[:1]),dict(jobs=[]))
    assert len(result['three_seed'])==2
    assert result['three_seed'][0]['ACC_mean']==67
