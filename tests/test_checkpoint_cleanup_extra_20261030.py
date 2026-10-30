import importlib
import json


def module():
    return importlib.import_module('experiments.osram_checkpoint_cleanup_extra_20261030.cleanup')


def test_scope_does_not_expand_current_models():
    m = module()
    assert len(m.FAMILIES) == 8
    assert 'osram_frozen_nested_3seed_20261026' not in m.FAMILIES
    assert 'osram_conditional_reread_20261023' not in m.FAMILIES


def test_baseline_ancestors_and_chosen_capacity_are_protected(tmp_path):
    m = module()
    for path in ('cfg84_out1600_kv64/seed_68', 'cfg11_capacity_baseline/seed_66',
                 'cfg01_baseline/seed_67', 'runs/f512_plain', 'runs/f1024_plain',
                 'runs/small_flat', 'runs/f512_v128'):
        assert m.preserve_run('any', tmp_path / path)
    assert not m.preserve_run('any', tmp_path / 'runs/f256_v128')


def test_scope_restores_legacy_globals_on_error():
    m = module(); old_families = m.legacy.FAMILIES; old_rule = m.legacy.preserve_run
    try:
        with m.scope():
            assert m.legacy.FAMILIES == m.FAMILIES
            assert m.legacy.preserve_run is m.preserve_run
            raise RuntimeError('simulate failure')
    except RuntimeError:
        pass
    assert m.legacy.FAMILIES is old_families
    assert m.legacy.preserve_run is old_rule


def test_exact_cleanup_keeps_masks_and_original_scope(tmp_path, monkeypatch):
    m = module()
    root = tmp_path / next(iter(m.FAMILIES)); run = root / 'run'; run.mkdir(parents=True)
    target = run / 'last.pt'; target.write_bytes(b'obsolete')
    mask = run / 'fixed_masks.pt'; mask.write_bytes(b'mask')
    monkeypatch.setattr(m.legacy, 'BASE', tmp_path)
    monkeypatch.setattr(m.legacy.common, 'live_paths', lambda: ([], set()))
    plan = dict(status='planned', executor_sha256=m.legacy.common.sha(m.legacy.__file__),
        driver_sha256=m.legacy.common.sha(m.__file__), families=sorted(m.FAMILIES),
        targets=[m.legacy.common.target_row(target, root, run)],
        retained=[m.legacy.fingerprint(mask)], dependency_records=[])
    manifest = tmp_path / 'PLAN.json'; manifest.write_text(json.dumps(plan))
    m.execute(manifest, tmp_path / 'receipt')
    assert not target.exists()
    assert mask.read_bytes() == b'mask'
    assert len(m.legacy.FAMILIES) == 7
