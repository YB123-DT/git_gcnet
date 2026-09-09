import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/osram_write_step_grid_20260909/summarize_grid.py'


def module():
    assert SCRIPT.exists(), 'grid summarizer is not implemented'
    spec = importlib.util.spec_from_file_location('grid_summary', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(tmp_path):
    root, inherited = tmp_path / 'grid', tmp_path / 'inherited'
    modes = ('reference', 'fixed0.95', 'fixed0.9', 'fixed0.8')
    for dataset in ('iemocap4', 'mosi'):
        for seed in range(66, 71):
            data = dict(seed=seed, dataset=dataset, checkpoint=f'/{dataset}/{seed}.pt',
                        checkpoint_sha256=f'hash-{dataset}-{seed}',
                        config=dict(seed=seed, dataset=dataset), weights_unchanged=True,
                        all_mode_masks_equal=True, results=[])
            for rate in (0., .1, .3, .5, .7):
                for i, mode in enumerate(modes):
                    data['results'].append(dict(rate=rate, mode=mode, mask_sha256=f'{seed}-{rate}',
                        task_metrics={'weighted_f1': .7 + (0., .01, .02, .015)[i] + (seed-66)*.001},
                        retention={} if rate == 0 else {'audio': {'counts': {'retention': 2, 'NO_HISTORY': 3},
                            'metrics': {'err_decay': {'count': 2, 'mean': i+.25}, 'err_post': {'count': 2, 'mean': 999}}}},
                        current_observed_write_fit={'audio': {'metrics': {'err_after': {'count': seed, 'mean': i+.5}}}}))
            path = root / dataset / f'seed{seed}' / 'metadata.json'
            path.parent.mkdir(parents=True)
            if dataset == 'iemocap4':
                old = inherited / f'iemocap4_seed{seed}' / 'metadata.json'
                old.parent.mkdir(parents=True)
                old.write_text(json.dumps(data))
                data['results'] = [r for r in data['results'] if r['mode'] in ('fixed0.95', 'fixed0.8')]
            path.write_text(json.dumps(data))
    return root, inherited


def test_summary_preserves_pairing_and_reports_gate(tmp_path):
    mod = module()
    root, inherited = fixture(tmp_path)
    rows = mod.summarize(root, inherited)
    row = next(r for r in rows if r['dataset']=='iemocap4' and r['rate']=='all'
               and r['mode']=='fixed0.9' and r['metric']=='weighted_f1')
    assert row['mean'] == pytest.approx(.722)
    assert row['n_seeds'] == 5
    assert (root / 'gate.json').exists()
    assert json.loads((root / 'gate.json').read_text())['gate_passed'] is True
    assert 'NO_HISTORY' in (root / 'SUMMARY.md').read_text()
    assert str(inherited) in (root / 'provenance.csv').read_text()
    assert (root / 'per_seed_shape.csv').exists()
    assert not any(r['metric']=='err_post' for r in rows)


@pytest.mark.parametrize('field,value,match', [
    ('checkpoint_sha256', 'wrong', 'checkpoint'),
    ('config', {'seed':66, 'dataset':'iemocap4', 'changed':True}, 'config'),
])
def test_reject_inherited_identity_mismatch(tmp_path, field, value, match):
    mod = module()
    root, inherited = fixture(tmp_path)
    path = root / 'iemocap4/seed66/metadata.json'
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        mod.summarize(root, inherited)


def test_reject_cross_source_mask_mismatch(tmp_path):
    mod = module()
    root, inherited = fixture(tmp_path)
    path = root / 'iemocap4/seed66/metadata.json'
    data = json.loads(path.read_text())
    data['results'][0]['mask_sha256'] = 'different'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='mask'):
        mod.summarize(root, inherited)


@pytest.mark.parametrize('case', ['tie_at_080', 'only_two_positive'])
def test_gate_requires_both_datasets_and_strict_shape(tmp_path, case):
    mod = module()
    root, inherited = fixture(tmp_path)
    for seed in range(66, 71):
        path = root / 'mosi' / f'seed{seed}' / 'metadata.json'
        data = json.loads(path.read_text())
        for result in data['results']:
            baseline = .7+(seed-66)*.001
            if case == 'tie_at_080' and result['mode']=='fixed0.8':
                result['task_metrics']['weighted_f1'] = baseline+.02
            if case == 'only_two_positive' and result['mode']=='fixed0.9':
                result['task_metrics']['weighted_f1'] = baseline+(.1 if seed<68 else -.01)
        path.write_text(json.dumps(data))
    mod.summarize(root, inherited)
    gate = json.loads((root / 'gate.json').read_text())
    assert gate['datasets']['iemocap4']['passed'] is True
    assert gate['gate_passed'] is False


def test_retention_excludes_no_history_and_uses_actual_read_error():
    mod = module()
    result = dict(task_metrics={'weighted_f1': .7}, retention={'audio': {
        'counts': {'retention': 2, 'NO_HISTORY': 8},
        'metrics': {'err_decay': {'count': 2, 'mean': .25}, 'err_post': {'count': 2, 'mean': 999}}}})
    values = mod.flatten(result)
    assert values['retention', 'audio', 'err_decay'] == .25
    assert ('retention', 'audio', 'err_post') not in values
    result['retention']['audio']['metrics']['err_decay']['count'] = 10
    with pytest.raises(ValueError, match='NO_HISTORY'):
        mod.flatten(result)


def test_run_means_have_equal_seed_weight(tmp_path):
    mod = module()
    root, inherited = fixture(tmp_path)
    for seed in range(66, 71):
        path = root / 'mosi' / f'seed{seed}' / 'metadata.json'
        data = json.loads(path.read_text())
        for result in data['results']:
            result['current_observed_write_fit']['audio']['metrics']['err_after'] = {
                'mean': seed-66, 'count': 100000 if seed==70 else 1}
        path.write_text(json.dumps(data))
    rows = mod.summarize(root, inherited)
    row = next(r for r in rows if r['dataset']=='mosi' and r['rate']=='all'
               and r['mode']=='reference' and r['metric']=='err_after')
    assert row['mean'] == 2
