import importlib
import json

import pytest


def module():
    return importlib.import_module('experiments.osram_checkpoint_cleanup_20261030.cleanup')


def test_only_explicit_families_and_model_filenames():
    m = module()
    assert len(m.FAMILIES) == 7
    assert 'osram_frozen_nested_3seed_20261026' not in m.FAMILIES
    for name in ('fixed_masks.pt', 'selected_masks.pt', 'train_support.pt'):
        assert not m.common.MODEL.fullmatch(name)


def test_keep_controls_winners_and_head48(tmp_path):
    m = module()
    for family, run in (
        ('osram_mosi_memory_grid_20261022', 'execution_v1/runs/head_48'),
        ('osram_mosi_memory_grid_20261022', 'execution_v1/runs/baseline'),
        ('osram_mosi_memory_grid_20261022', 'execution_v1/runs/step_0p4'),
        ('osram_source_role_direct_cross_20261020', 'execution_v1/runs/CMUMOSEI/fold_1/seed_66/direct'),
        ('osram_nested_cmerc_tuning_20261023', 'execution_v1/runs/CMUMOSEI/fold_1/seed_66/none'),
        ('osram_nested_cmerc_tuning_20261023', 'execution_v1/runs/IEMOCAPSix/fold_5/seed_66/nested_dim128'),
        ('osram_reg_only_hparam_sweep_20260919_100', 'screen_seed66/cfg84_out1600_kv64'),
    ):
        assert m.preserve_run(family, tmp_path / family / run)
    assert not m.preserve_run('osram_mosi_memory_grid_20261022', tmp_path / 'runs/step_1p0')


def test_saved_score_required():
    m = module()
    assert m.has_scores({'status': 'complete', 'best_by_rate': {'0.1': {'epoch': 3, 'metrics': {'ACC': 50}}}})
    assert m.has_scores({'selected_weighted_f1_by_rate': {'0.1': 0.5}, 'selected_epoch_by_rate': {'0.1': 2}})
    assert not m.has_scores({'status': 'running'})
    assert not m.has_scores({'status': 'failed', 'best_by_rate': {'0.1': {'epoch': 3, 'metrics': {'ACC': 50}}}})


def test_dependency_scan_ignores_inventory_but_keeps_runtime_parent(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, 'BASE', tmp_path)
    run = tmp_path / next(iter(m.FAMILIES)) / 'run'; run.mkdir(parents=True)
    old_inventory = tmp_path / 'old_checkpoint_cleanup'; old_inventory.mkdir()
    (old_inventory / 'PLAN.json').write_text(json.dumps({'retained': str(run / 'last.pt')}))
    snapshot = tmp_path / 'code_snapshots'; snapshot.mkdir()
    (snapshot / 'config.json').write_text(json.dumps({'parent': str(run / 'last.pt')}))
    assert not m.external_dependencies({run})[0]
    runtime = tmp_path / 'later_training'; runtime.mkdir()
    (runtime / 'config.json').write_text(json.dumps({'parent': str(run / 'last.pt')}))
    protected, records = m.external_dependencies({run})
    assert str(run) in protected
    assert len(records) == 1


@pytest.mark.parametrize('folder', ['source', 'dev_source', 'preflight_source', 'experiments'])
def test_do_not_traverse_archived_source_snapshots(tmp_path, folder):
    m = module()
    snapshot = tmp_path / folder / 'nested'; snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{}')
    assert all(directory != snapshot for directory, _ in m.walk(tmp_path))


def test_exact_execution_preserves_mask_and_other_weights(tmp_path, monkeypatch):
    m = module()
    root = tmp_path / next(iter(m.FAMILIES)); run = root / 'run'; run.mkdir(parents=True)
    weight = run / 'last.pt'; weight.write_bytes(b'obsolete')
    mask = run / 'fixed_masks.pt'; mask.write_bytes(b'mask')
    kept = run / 'best_epoch_1.pt'; kept.write_bytes(b'protected')
    monkeypatch.setattr(m, 'BASE', tmp_path)
    monkeypatch.setattr(m.common, 'live_paths', lambda: ([], set()))
    row = m.common.target_row(weight, root, run)
    plan = {'status': 'planned', 'executor_sha256': m.common.sha(m.__file__), 'targets': [row],
            'retained': [m.fingerprint(mask), m.fingerprint(kept)], 'dependency_records': []}
    manifest = tmp_path / 'manifest.json'; manifest.write_text(json.dumps(plan))
    m.execute(manifest, tmp_path / 'receipt')
    assert not weight.exists()
    assert mask.read_bytes() == b'mask'
    assert kept.read_bytes() == b'protected'
    assert json.loads((tmp_path / 'receipt/RECEIPT.json').read_text())['deleted_count'] == 1


def test_retained_evidence_change_blocks_all_deletion(tmp_path, monkeypatch):
    m = module()
    root = tmp_path / next(iter(m.FAMILIES)); run = root / 'run'; run.mkdir(parents=True)
    weight = run / 'last.pt'; weight.write_bytes(b'obsolete')
    mask = run / 'fixed_masks.pt'; mask.write_bytes(b'mask')
    monkeypatch.setattr(m, 'BASE', tmp_path)
    monkeypatch.setattr(m.common, 'live_paths', lambda: ([], set()))
    plan = {'status': 'planned', 'executor_sha256': m.common.sha(m.__file__),
            'targets': [m.common.target_row(weight, root, run)],
            'retained': [m.fingerprint(mask)], 'dependency_records': []}
    manifest = tmp_path / 'manifest.json'; manifest.write_text(json.dumps(plan))
    mask.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        m.execute(manifest, tmp_path / 'receipt')
    assert weight.exists()
