import csv
import importlib.util
import json
from pathlib import Path

import pytest

from test_write_step_grid_summary import fixture as old_fixture


SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/osram_write_step_range_20260909/summarize_range.py'


def module():
    assert SCRIPT.exists(), 'range summarizer is not implemented'
    spec = importlib.util.spec_from_file_location('range_summary', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(tmp_path, peak=.4):
    grid, inherited = old_fixture(tmp_path)
    root = tmp_path / 'range'
    for dataset in ('iemocap4', 'mosi'):
        for seed in range(66, 71):
            data = json.loads((grid / dataset / f'seed{seed}/metadata.json').read_text())
            template = [r for r in data['results'] if r['mode'] == 'fixed0.8']
            data['results'] = []
            for eta in (.6, .4, .2, .0):
                for old in template:
                    result = json.loads(json.dumps(old))
                    result['mode'] = f'fixed{eta:.1f}'
                    result['task_metrics']['weighted_f1'] = .8 - .1 * (eta-peak)**2 + (seed-66)*.001
                    data['results'].append(result)
            path = root / dataset / f'seed{seed}/metadata.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(data))
    return root, grid, inherited


def test_interior_peak_bracket_and_paired_support(tmp_path):
    mod = module()
    root, grid, inherited = fixture(tmp_path)
    rows = mod.summarize(root, grid, inherited)
    peak = next(r for r in rows if r['dataset']=='iemocap4' and r['rate']=='all' and r['mode']=='fixed0.4' and r['metric']=='weighted_f1')
    assert peak['mean'] == pytest.approx(.802)
    brackets = list(csv.DictReader((root / 'peak_brackets.csv').open()))
    row = next(r for r in brackets if r['dataset']=='iemocap4' and r['rate']=='all')
    assert float(row['lower_eta']) == .2
    assert float(row['higher_eta']) == .6
    assert row['positive_vs_lower'] == '5'
    assert row['positive_vs_higher'] == '5'
    assert len(list(csv.DictReader((root / 'provenance.csv').open()))) == 400
    assert not any(r['metric']=='err_post' for r in rows)
    assert not any(r['metric']=='err_decay' and r['rate']=='all' for r in rows)
    report = (root / 'SUMMARY.md').read_text()
    assert 'AFTER' in report and 'NO_HISTORY' in report
    assert not (root / 'gate.json').exists()


def test_zero_endpoint_is_not_interior_optimum(tmp_path):
    mod = module()
    root, grid, inherited = fixture(tmp_path, peak=0)
    mod.summarize(root, grid, inherited)
    rows = list(csv.DictReader((root / 'peak_brackets.csv').open()))
    assert all(r['location']=='off-memory boundary; no interior optimum established' for r in rows)
    assert all(r['lower_eta']=='' and float(r['higher_eta'])==.2 for r in rows)
    assert 'not a trained Local model' in (root / 'SUMMARY.md').read_text()


@pytest.mark.parametrize('field,value', [('checkpoint_sha256', 'bad'), ('epoch', 999), ('config', {'seed':66, 'bad':True})])
def test_reject_inherited_identity_mismatch(tmp_path, field, value):
    mod = module()
    root, grid, inherited = fixture(tmp_path)
    path = root / 'iemocap4/seed66/metadata.json'
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=field):
        mod.summarize(root, grid, inherited)


def test_reject_cross_source_mask_mismatch(tmp_path):
    mod = module()
    root, grid, inherited = fixture(tmp_path)
    path = root / 'mosi/seed66/metadata.json'
    data = json.loads(path.read_text())
    data['results'][0]['mask_sha256'] = 'bad'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='mask'):
        mod.summarize(root, grid, inherited)
