import copy
import importlib


def test_scope_reduction_preserves_living_mosei_and_never_restarts_it():
    m = importlib.import_module('experiments.osram_nested_cmerc_tuning_20261023.rescope')
    methods = ('none', 'nested_gnn_rooted_evidence', 'nested_dim32', 'nested_dim128',
        'mosei_nested_depth2', 'mosei_nested_mlp128', 'mosei_nested_groups4')
    jobs = [dict(dataset=d, method=method, status='pending', command=['python', '--source-commit', 'oldsha'])
        for method in methods for d in ('CMUMOSEI', 'IEMOCAPFour', 'IEMOCAPSix')]
    jobs[0].update(status='running', pid=12345)
    jobs[1].update(status='running', pid=23456)
    state = dict(jobs=jobs, source_commit='oldsha', maximum_jobs=21)
    old = copy.deepcopy(state)
    reduced = m.reduce_state(state)
    assert state == old
    assert reduced['jobs'][0] == state['jobs'][0]
    assert len(m.active_jobs(reduced)) == 14
    assert {j['dataset'] for j in m.active_jobs(reduced)} == {'CMUMOSEI', 'IEMOCAPSix'}
    removed = [j for j in reduced['jobs'] if j['dataset'] == 'IEMOCAPFour']
    assert len(removed) == 7 and all(j['status'] == 'cancelled_by_user' for j in removed)
    assert reduced['maximum_jobs'] == 14 and reduced['source_commit'] == 'oldsha'
    assert all('--resume' not in j['command'] for j in m.active_jobs(reduced))
