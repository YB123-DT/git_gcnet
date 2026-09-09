import csv
import importlib.util
import json
from pathlib import Path

import pytest

from test_write_step_range_summary import fixture as range_fixture


SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/osram_write_step_train_20260909/summarize.py'


def module():
    assert SCRIPT.exists(), 'cross summarizer is not implemented'
    spec = importlib.util.spec_from_file_location('cross_summary', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(tmp_path):
    range_root, grid_root, inherited_root = range_fixture(tmp_path)
    root = tmp_path / 'train'
    for base in (range_root, grid_root, inherited_root):
        for path in base.rglob('metadata.json'):
            data = json.loads(path.read_text())
            data.update(evaluation_only=True, new_checkpoint_selection=False,
                        epoch=33, selection_protocol='8-rate-mean-test-oracle')
            path.write_text(json.dumps(data))
    for dataset in ('iemocap4', 'mosi'):
        for seed in range(66, 71):
            old = json.loads((range_root / dataset / f'seed{seed}/metadata.json').read_text())
            for label, eta, f1 in [('C_train06_test06', .6, .82), ('D_train06_test1', 1., .81)]:
                data = json.loads(json.dumps(old))
                data.update(checkpoint=f'new-{dataset}-{seed}', checkpoint_sha256=f'newhash-{dataset}-{seed}',
                            training_write_step=.6, evaluation_write_step=eta, epoch=55)
                data['config'].update(osram_write_step=.6, osram_forward_slot_reuse=False)
                data['results'] = [r for r in data['results'] if r['mode']=='fixed0.6']
                for result in data['results']:
                    result.update(mode='reference')
                    result['task_metrics']['weighted_f1'] = f1+(seed-66)*.001
                path = root / dataset / f'seed_{seed}' / label / 'metadata.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
    return root, grid_root, range_root, inherited_root


def test_cross_summary_preserves_pairing_and_legacy_defaults(tmp_path):
    mod = module()
    paths = fixture(tmp_path)
    rows = mod.summarize(*paths)
    row = next(r for r in rows if r['dataset']=='mosi' and r['rate']=='all' and r['arm']=='C' and r['metric']=='weighted_f1')
    assert row['mean']==pytest.approx(.822)
    delta = next(r for r in csv.DictReader((paths[0]/'paired_deltas.csv').open()) if r['dataset']=='mosi' and r['rate']=='all' and r['comparison']=='C-A')
    assert float(delta['mean'])==pytest.approx(.12)
    assert delta['positive_seeds']=='5'
    report=(paths[0]/'SUMMARY.md').read_text()
    assert 'eight-rate' in report and 'five-rate' in report and 'NO_HISTORY' in report
    assert not any(r['metric']=='err_post' for r in rows)
    assert len(list(csv.DictReader((paths[0]/'provenance.csv').open())))==200


@pytest.mark.parametrize('field,value', [('checkpoint_sha256','bad'), ('epoch',999), ('evaluation_write_step',.6), ('new_checkpoint_selection',True)])
def test_reject_crossed_checkpoint_or_protocol_mismatch(tmp_path, field, value):
    mod=module(); paths=fixture(tmp_path)
    path=paths[0]/'mosi/seed_66/D_train06_test1/metadata.json'
    data=json.loads(path.read_text()); data[field]=value; path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match=field): mod.summarize(*paths)


def test_reject_extra_training_config_change(tmp_path):
    mod=module(); paths=fixture(tmp_path)
    path=paths[0]/'mosi/seed_66/C_train06_test06/metadata.json'
    data=json.loads(path.read_text()); data['config']['learning_rate']=.5; path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='config'): mod.summarize(*paths)


def test_reject_mask_mismatch(tmp_path):
    mod=module(); paths=fixture(tmp_path)
    path=paths[0]/'iemocap4/seed_66/C_train06_test06/metadata.json'
    data=json.loads(path.read_text()); data['results'][0]['mask_sha256']='bad'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='mask'): mod.summarize(*paths)


def test_missing_result_is_pending_without_fake_summary(tmp_path):
    mod=module(); paths=fixture(tmp_path)
    (paths[0]/'mosi/seed_70/D_train06_test1/metadata.json').unlink()
    with pytest.raises(ValueError,match='PENDING'): mod.summarize(*paths)
    assert not (paths[0]/'SUMMARY.md').exists()
