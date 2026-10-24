"""Small retained artifacts exercise the complete Flat128 report without a GPU."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import pytest


MODULE = 'experiments.osram_mosei_flat128_nested_20261024.report'
BASE_CONFIG = dict(dataset='CMUMOSEI', seed=66, fold=1, epochs=60,
    learning_rate=1e-4, batch_size=32, weight_decay=1e-5, optimizer='adam',
    lr_schedule='constant', missing_mask_protocol='gcnet-random-mask-v1',
    training_objective='emotion-only', train_rate_mode='cyclic',
    train_missing_rates=[i/10 for i in range(8)], checkpoint_selection='test-oracle-per-rate',
    checkpoint_selection_metric='weighted_f1', evaluate_test=True,
    osram_adapter_hidden_dim=128, osram_output_dim=1600, osram_meaningful_block='none',
    osram_num_heads=8, osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
    osram_disable_local_skip=False, osram_bidirectional=False, osram_readout_fusion='flat',
    paired_history_views=False, osram_source_role_v3=False, osram_tasr=False,
    osram_source_role=False, osram_source_relation=False, osram_source_tracking=False,
    completion_path='none', initial_backbone_checkpoint=None)
BASE_CONFIG.update(mosi_task_mode='regression', task_regression_loss='mse')


def report():
    return importlib.import_module(MODULE)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def artifact_hashing_without_torch(monkeypatch):
    helper = types.ModuleType('experiments.osram_tasr_20261018.runner_helpers')
    helper.sha = digest
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    if MODULE in sys.modules:
        sys.modules[MODULE]._CACHE.clear()


def make_run(path, variant='F66', *, score=80., parameters=100):
    module = report()
    seed = int(variant[1:]) if variant.startswith('F') else 66
    method = 'none' if variant.startswith('F') else module.method_name(variant)
    cfg = dict(BASE_CONFIG, seed=seed, osram_meaningful_block=method)
    path.mkdir(parents=True)
    history = [dict(epoch=i+1, mask_trace_sha256=f'{seed}-trace-{i}', optimizer_steps=10,
                    learning_rates=[1e-4], test={}) for i in range(60)]
    best, evaluation = {}, {}
    for index, rate in enumerate(module.RATES):
        metrics = dict(router=dict(ACC=score+index/10, WF1=score+index/10, MSE=.5))
        names = dict(checkpoint=f'best_{index}.pt', predictions=f'pred_{index}.npz')
        for name in names.values():
            (path/name).write_bytes(f'{variant}-{name}'.encode())
        best[rate] = dict(**names, checkpoint_sha256=digest(path/names['checkpoint']),
            prediction_sha256=digest(path/names['predictions']), epoch=1,
            metrics=metrics, same_evaluation_call=True)
        (path/'evaluation').mkdir(exist_ok=True)
        target = path/'evaluation'/f"random_{rate.replace('.', 'p')}.npz"
        target.write_bytes((path/names['predictions']).read_bytes())
        evaluation[rate] = dict(metrics, selected_epoch=1, output_sha256=digest(target))
        for row in history:
            row['test'][rate] = copy.deepcopy(metrics)
    aggregate = {scope: {metric: sum(evaluation[r]['router'][metric] for r in rates)/len(rates)
                        for metric in module.METRICS}
                 for scope, rates in [('mean', module.RATES), ('high', module.RATES[5:])]}
    summary = dict(status='complete', phase='train', dataset='CMUMOSEI', fold=1,
        epoch=60, method=method, seed=seed, config=cfg, source_commit='commit',
        identity=dict(config=cfg, seed=seed, method=method,
            source_hashes={'gcnet_missing_m3/model.py': 'same',
                'gcnet_missing_m3/meaningful_new40_registry.py': 'old' if variant=='F66' else 'new'},
            split={'accepted': 66}, mask_reference={'test': 'fixed'}, data={'files': ['same']},
            rng_repair='uuid-repaired', seed_protocol=dict(training_seed=seed, split_seed=66,
                evaluation_seed=66, train_loader={'seed': seed})),
        environment={'name': 'cmerc-repro'}, initial_backbone_sha256=f'init-{seed}',
        total_parameters=4746977+(0 if variant.startswith('F') else parameters),
        nested_parameters=0 if variant.startswith('F') else parameters,
        seconds=120., peak_memory_mib=1234., best_by_rate=best, evaluation=evaluation,
        aggregate={'router': aggregate})
    (path/'SUMMARY.json').write_text(json.dumps(summary))
    (path/'history.json').write_text(json.dumps(history))
    return summary, history


def make_state(tmp_path, monkeypatch, complete=()):
    module = report()
    monkeypatch.setattr(module, 'verify_registry_extension', lambda *args: {
        'registry_sha256': 'new', 'base_registry_sha256': 'old'})
    baseline = tmp_path/'F66'
    make_run(baseline)
    monkeypatch.setattr(module, 'REUSED_CONTROL', baseline)
    jobs = []
    for variant in ('F67', 'F68', *module.VARIANTS):
        path = tmp_path/variant
        if variant in complete:
            score = 79. if variant=='F67' else (81. if variant=='F68' else 82.)
            make_run(path, variant, score=score, parameters=101 if variant=='N01' else 100)
        jobs.append(dict(variant=variant, output=str(path), status='complete' if variant in complete else 'pending',
                         **module.run_spec(variant)))
    return dict(source_commit='commit', source_extension_proof={'verified': True}, jobs=jobs)


def test_report_entrypoint_exists():
    assert (Path(__file__).resolve().parents[1]/MODULE.replace('.', '/')).with_suffix('.py').exists()


def test_reused_control_requires_exact_protocol_and_retained_bytes(tmp_path):
    module = report()
    path = tmp_path/'F66'
    summary, _ = make_run(path)
    proof = module.validate_reused_control(path)
    assert proof['summary_sha256'] == digest(path/'SUMMARY.json')
    assert proof['total_parameters'] == 4746977
    (path/summary['best_by_rate']['0.0']['checkpoint']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        module.validate_reused_control(path)


def test_seed_exception_allows_only_flat_seed_and_train_mask_changes(tmp_path):
    module = report()
    baseline, old = make_run(tmp_path/'F66')
    candidate, history = make_run(tmp_path/'F67', 'F67')
    module.validate_pair(candidate, history, baseline, old, variant='F67')
    history[0]['optimizer_steps'] += 1
    with pytest.raises(ValueError, match='trace|steps'):
        module.validate_pair(candidate, history, baseline, old, variant='F67')


@pytest.mark.parametrize('mutation', ['initial', 'seed', 'split', 'mask', 'source', 'environment',
    'rng', 'trace', 'lr', 'config', 'seed_protocol', 'budget'])
def test_nested_rejects_every_pairing_drift(tmp_path, mutation):
    module = report()
    baseline, old = make_run(tmp_path/'F66')
    candidate, history = make_run(tmp_path/'N01', 'N01')
    if mutation=='initial': candidate['initial_backbone_sha256'] = 'wrong'
    elif mutation=='seed': candidate['config']['seed'] = 67
    elif mutation=='split': candidate['identity']['split'] = 'wrong'
    elif mutation=='mask': candidate['identity']['mask_reference'] = 'wrong'
    elif mutation=='source': candidate['identity']['source_hashes']['gcnet_missing_m3/model.py'] = 'wrong'
    elif mutation=='environment': candidate['environment'] = {'name': 's0'}
    elif mutation=='rng': candidate['identity']['rng_repair'] = 'old'
    elif mutation=='trace': history[0]['mask_trace_sha256'] = 'wrong'
    elif mutation=='lr': history[0]['learning_rates'] = [1e-4, 1e-4]
    elif mutation=='config': candidate['config']['osram_adapter_hidden_dim'] = 256
    elif mutation=='seed_protocol': candidate['identity']['seed_protocol']['training_seed'] = 67
    else: history.pop()
    with pytest.raises(ValueError):
        module.validate_pair(candidate, history, baseline, old, variant='N01')


def test_progress_ranks_only_complete_nested_and_has_no_unpaired_seed_gain(tmp_path, monkeypatch):
    module = report()
    state = make_state(tmp_path, monkeypatch, complete=('F67', 'F68', 'N01', 'N02'))
    output = tmp_path/'report'
    result = module.summarize(output, state)
    assert len(result['states']) == 24
    assert len(result['runs']) == 5
    assert result['nested_ranking'] == ['N02', 'N01']
    assert result['flat_three_seed']['mean']['WF1']['mean'] == pytest.approx(80.35)
    assert result['flat_three_seed']['mean']['WF1']['sample_sd'] == pytest.approx(1.)
    assert result['runs']['N01']['delta_vs_flat66']['mean']['WF1'] == pytest.approx(2.)
    assert 'delta_vs_flat_three_seed' not in result['runs']['N01']
    assert len((output/'PER_RATE.csv').read_text().splitlines()) == 41
    assert not (output/'RESULT.json').exists()
    assert 'pending' in (output/'RESULT.md').read_text()
    with pytest.raises(ValueError, match='24|complete'):
        module.summarize(output, state, final=True)


def test_final_requires_exact_23_jobs_and_returns_all_24_results(tmp_path, monkeypatch):
    module = report()
    state = make_state(tmp_path, monkeypatch, complete=('F67', 'F68', *module.VARIANTS))
    result = module.summarize(tmp_path/'report', state, final=True)
    assert result['complete'] and len(result['runs']) == 24
    assert len((tmp_path/'report'/'PER_RATE.csv').read_text().splitlines()) == 193
    assert len(result['runs']['N01']['epochs']) == 60
    assert result['runs']['N01']['source_hashes']['gcnet_missing_m3/model.py'] == 'same'
    state['jobs'][-1]['variant'] = state['jobs'][0]['variant']
    with pytest.raises(ValueError, match='23|duplicate|variant'):
        module.summarize(tmp_path/'report', state)


@pytest.mark.parametrize('mutation', ['tie', 'metric', 'aggregate', 'published', 'nan'])
def test_selected_validation_rejects_incorrect_earliest_or_same_call_artifacts(tmp_path, monkeypatch, mutation):
    module = report()
    state = make_state(tmp_path, monkeypatch, complete=('N01',))
    path = tmp_path/'N01'/'SUMMARY.json'
    summary = json.loads(path.read_text())
    if mutation=='tie':
        summary['best_by_rate']['0.0']['epoch'] = 2
        summary['evaluation']['0.0']['selected_epoch'] = 2
    elif mutation=='metric': summary['best_by_rate']['0.0']['same_evaluation_call'] = False
    elif mutation=='aggregate': summary['aggregate']['router']['mean']['WF1'] += .1
    elif mutation=='published': (tmp_path/'N01'/'evaluation'/'random_0p0.npz').write_bytes(b'wrong')
    else: summary['aggregate']['router']['mean']['WF1'] = float('nan')
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        module.summarize(tmp_path/'report', state)


def test_completed_validation_is_cached_until_summary_changes_but_forced_final(tmp_path, monkeypatch):
    module = report()
    state = make_state(tmp_path, monkeypatch, complete=('F67', 'F68', *module.VARIANTS))
    output = tmp_path/'report'
    module.summarize(output, state)
    (tmp_path/'N01'/'best_0.pt').write_bytes(b'corrupt')
    module.summarize(output, state)  # Immutable completed artifacts; cheap polling.
    with pytest.raises(ValueError, match='integrity'):
        module.summarize(output, state, final=True)


@pytest.mark.parametrize('mutation', ['registry', 'commit', 'job_seed', 'duplicate_output'])
def test_report_binds_artifacts_to_verified_source_and_job_identity(tmp_path, monkeypatch, mutation):
    module = report()
    state = make_state(tmp_path, monkeypatch, complete=('N01',))
    job = next(j for j in state['jobs'] if j['variant']=='N01')
    job['source_commit'] = 'commit'
    path = Path(job['output'])/'SUMMARY.json'
    summary = json.loads(path.read_text())
    if mutation=='registry':
        summary['identity']['source_hashes']['gcnet_missing_m3/meaningful_new40_registry.py'] = 'unverified'
    elif mutation=='commit': summary['source_commit'] = 'other'
    elif mutation=='job_seed': job['seed'] = 67
    else: state['jobs'][1]['output'] = state['jobs'][0]['output']
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        module.summarize(tmp_path/'report', state)


def test_running_summary_with_partial_aggregate_never_enters_scores(tmp_path, monkeypatch):
    module = report()
    state = make_state(tmp_path, monkeypatch, complete=('N01',))
    path = tmp_path/'N01'/'SUMMARY.json'
    summary = json.loads(path.read_text())
    summary.update(status='running', epoch=59)
    path.write_text(json.dumps(summary))
    result = module.summarize(tmp_path/'report', state)
    assert result['nested_ranking'] == []
    assert set(result['runs']) == {'F66'}
    assert result['states']['N01']['epoch'] == 59


def test_summary_change_invalidates_completed_hash_cache(tmp_path, monkeypatch):
    module = report()
    state = make_state(tmp_path, monkeypatch, complete=('N01',))
    output = tmp_path/'report'
    module.summarize(output, state)
    (tmp_path/'N01'/'best_0.pt').write_bytes(b'corrupt')
    path = tmp_path/'N01'/'SUMMARY.json'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='integrity'):
        module.summarize(output, state)


@pytest.mark.parametrize('mutation', ['loss', 'task', 'epochs'])
def test_reused_baseline_self_pair_does_not_hide_protocol_drift(tmp_path, mutation):
    module = report()
    path = tmp_path/'F66'
    summary, history = make_run(path)
    if mutation=='loss': summary['config']['task_regression_loss'] = 'smooth_l1'
    elif mutation=='task': summary['config']['mosi_task_mode'] = 'classification'
    else:
        history[0]['epoch'] = 60
        (path/'history.json').write_text(json.dumps(history))
    (path/'SUMMARY.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        module.validate_reused_control(path)


@pytest.mark.parametrize('mutation', ['initialization', 'loader', 'masks'])
def test_fresh_flat_seed_cannot_silently_reuse_seed66_randomness(tmp_path, mutation):
    module = report()
    baseline, old = make_run(tmp_path/'F66')
    candidate, history = make_run(tmp_path/'F67', 'F67')
    if mutation=='initialization': candidate['initial_backbone_sha256'] = baseline['initial_backbone_sha256']
    elif mutation=='loader': candidate['identity']['seed_protocol']['train_loader'] = baseline['identity']['seed_protocol']['train_loader']
    else:
        for row, reference in zip(history, old):
            row['mask_trace_sha256'] = reference['mask_trace_sha256']
    with pytest.raises(ValueError):
        module.validate_pair(candidate, history, baseline, old, variant='F67')


def test_historical_missing_epoch_timing_is_explicit(tmp_path, monkeypatch):
    module = report()
    state = make_state(tmp_path, monkeypatch)
    result = module.summarize(tmp_path/'report', state)
    timing = result['runs']['F66']['timing']
    assert timing['wall_seconds_per_epoch'] == 2.
    assert timing['training_seconds_by_epoch'] is None
    assert 'unavailable' in timing['epoch_timing_note']
