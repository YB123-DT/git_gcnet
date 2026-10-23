"""Lightweight report regression tests: no Torch, model loading or GPU needed."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

MODULE = 'experiments.osram_iemocap6_nested128_3seed_20261024.report'


def report_module():
    return importlib.import_module(MODULE)


def fixture_run(path, seed=66, fold=5):
    module = report_module()
    path.mkdir(parents=True)
    config = dict(module.REQUIRED_CONFIG, seed=seed, fold=fold)
    source = {'gcnet_missing_m3/osram.py': 'osram-hash',
              'gcnet_missing_m3/model.py': 'model-hash'}
    environment = dict(module.REQUIRED_ENVIRONMENT)
    history = []
    masks, best, evaluation = {}, {}, {}
    for epoch in range(1, 101):
        history.append(dict(epoch=epoch, test={}, learning_rates=[.001]))
    for i, rate in enumerate(module.RATES):
        metric = dict(ACC=60 + seed - 66 + fold + i / 10,
                      UA=55 + seed - 66 + fold, WF1=59 + fold, CE=1.1)
        mask_hash = hashlib.sha256(f'fold{fold}-{rate}'.encode()).hexdigest()
        masks[rate] = dict(mask_sha256=mask_hash, path='reference', sha256='reference-hash')
        names = dict(checkpoint=f'best_{i}.pt', predictions=f'pred_{i}.npz')
        for name in names.values():
            (path / name).write_bytes(f'{seed}-{fold}-{name}'.encode())
        selected = dict(router=metric, mask_sha256=mask_hash)
        best[rate] = dict(**names, checkpoint_sha256=module.sha256(path / names['checkpoint']),
            prediction_sha256=module.sha256(path / names['predictions']), epoch=1,
            metrics=selected, same_evaluation_call=True,
            criterion='TEST accuracy; strict improvement; earliest tie')
        (path / 'evaluation').mkdir(exist_ok=True)
        (path / 'evaluation' / f"random_{rate.replace('.', 'p')}.npz").write_bytes(
            (path / names['predictions']).read_bytes())
        evaluation[rate] = dict(selected, selected_epoch=1, output_sha256=best[rate]['prediction_sha256'])
        for row in history:
            row['test'][rate] = copy.deepcopy(selected)
    aggregate = {'rates': {r: v['router'] for r, v in evaluation.items()}}
    for scope, rates in [('mean', module.RATES), ('high', module.RATES[5:])]:
        aggregate[scope] = {m: sum(evaluation[r]['router'][m] for r in rates) / len(rates)
                            for m in module.METRICS}
    summary = dict(status='complete', phase='train', dataset='IEMOCAPSix', method='nested_dim128',
        seed=seed, fold=fold, epoch=100, source_commit='commit', config=config,
        identity=dict(config=config, source_hashes=source, environment=environment, mask_reference=masks),
        environment=environment, aggregate=dict(router=aggregate), best_by_rate=best,
        evaluation=evaluation, total_parameters=13950441, nested_parameters=432387,
        seconds=120., peak_memory_mib=1500.)
    (path / 'SUMMARY.json').write_text(json.dumps(summary))
    (path / 'history.json').write_text(json.dumps(history))
    (path / 'config.json').write_text(json.dumps(config))
    (path / 'test_masks.pt').write_bytes(b'retained mask artifact')
    (path / 'last.pt').write_bytes(b'retained full recovery artifact')
    return summary


def test_complete_reuse_validates_without_model_loading(tmp_path):
    module = report_module()
    path = tmp_path / 'reuse'
    fixture_run(path)
    assert module.validate_reused(path)['nested_parameters'] == 432387
    (path / 'pred_0.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        module.validate_reused(path)


@pytest.mark.parametrize('mutation', ['budget', 'mask', 'metric', 'mode', 'nan', 'epoch'])
def test_reject_protocol_or_selected_record_drift(tmp_path, mutation):
    module = report_module()
    path = tmp_path / 'run'
    summary = fixture_run(path)
    if mutation == 'budget':
        summary['epoch'] = 99
    elif mutation == 'mask':
        summary['evaluation']['0.0']['mask_sha256'] = 'wrong'
    elif mutation == 'metric':
        summary['evaluation']['0.0']['router']['UA'] += 1
    elif mutation == 'mode':
        summary['config']['osram_source_role_v3'] = True
    elif mutation == 'nan':
        summary['evaluation']['0.0']['router']['CE'] = float('nan')
    else:
        summary['best_by_rate']['0.0']['epoch'] = 2  # earliest tie must stay epoch1
        summary['evaluation']['0.0']['selected_epoch'] = 2
    (path / 'SUMMARY.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        module.load_run(path, 66, 5)


def complete_state(tmp_path):
    jobs = []
    reused = None
    for seed in (66, 67, 68):
        for fold in range(1, 6):
            path = tmp_path / f'{seed}_{fold}'
            fixture_run(path, seed, fold)
            row = dict(seed=seed, fold=fold, dataset='IEMOCAPSix', method='nested_dim128',
                       status='complete', output=str(path), source_commit='commit')
            if (seed, fold) == (66, 5):
                reused = dict(row, status='reused_complete')
            else:
                jobs.append(row)
    return dict(source_commit='commit', jobs=jobs, reused=reused)


def test_requires_all_fifteen_then_equal_fold_and_seed_statistics(tmp_path):
    module = report_module()
    state = complete_state(tmp_path)
    output = tmp_path / 'report'
    incomplete = copy.deepcopy(state)
    incomplete['jobs'].pop()
    with pytest.raises(ValueError, match='15|14'):
        module.write_report(output, incomplete)
    module.write_report(output, state)
    result = json.loads((output / 'RESULT.json').read_text())
    assert len(result['runs']) == 15
    assert result['seeds']['66']['mean']['ACC'] == pytest.approx(63.35)
    assert result['seeds']['67']['mean']['ACC'] == pytest.approx(64.35)
    assert result['three_seed']['mean']['ACC']['mean'] == pytest.approx(64.35)
    assert result['three_seed']['mean']['ACC']['sample_sd'] == pytest.approx(1.)
    assert len((output / 'PER_FOLD_RATE.csv').read_text().splitlines()) == 121
    assert 'INTERNAL TEST-ORACLE' in (output / 'RESULT.md').read_text()


def test_different_numerical_model_source_never_mixes_in_report(tmp_path):
    module = report_module()
    state = complete_state(tmp_path)
    path = Path(state['jobs'][0]['output']) / 'SUMMARY.json'
    s = json.loads(path.read_text())
    s['identity']['source_hashes']['gcnet_missing_m3/osram.py'] = 'changed'
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError, match='source'):
        module.write_report(tmp_path / 'report', state)
