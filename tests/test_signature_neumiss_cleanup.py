import importlib
import json
from pathlib import Path

import pytest

MODULE = 'experiments.osram_signature_neumiss_cleanup_20261026.cleanup'


def module():
    return importlib.import_module(MODULE)


def test_filename_whitelist_excludes_masks_and_unrelated_state():
    m = module()
    for name in ('best_epoch_12.pt', 'best_miss_0p7.pt', 'best_miss_0p3_epoch_75.pt', 'last.pt'):
        assert m.MODEL.fullmatch(name)
    for name in ('fixed_masks.pt', 'train_support.pt', 'projection.pt', 'model.py', 'last.pt.tmp'):
        assert not m.MODEL.fullmatch(name)


def test_method_scoping_does_not_delete_flat_csdi_or_ridge(tmp_path):
    m = module()
    for mode in ('signature', 'signature_noarea', 'neumiss'):
        assert m.allowed_method(dict(mode=mode), tmp_path/'run', 'mixed')
    for mode in ('flat', 'csdi', 'ridge', 'nested'):
        assert not m.allowed_method(dict(mode=mode), tmp_path/'run', 'mixed')
    assert m.allowed_method(dict(mode='serial'), tmp_path/'run', 'noarea')
    assert not m.allowed_method({}, tmp_path/'run', 'mixed')


def test_target_cannot_escape_or_change_after_inventory(tmp_path):
    m = module()
    root = tmp_path/'allowed'; run = root/'run'; run.mkdir(parents=True)
    target = run/'last.pt'; target.write_bytes(b'weights')
    row = m.target_row(target, root, run)
    m.validate_target(row, {root})
    with pytest.raises(ValueError):
        m.validate_target(row, {tmp_path/'different'})
    target.write_bytes(b'new weights')
    with pytest.raises(ValueError):
        m.validate_target(row, {root})


def test_symlink_and_hardlink_are_not_safe_targets(tmp_path):
    import os
    m = module()
    run = tmp_path/'run'; run.mkdir()
    file = run/'best_epoch_1.pt'; file.write_bytes(b'weights')
    link = run/'last.pt'; link.symlink_to(file)
    with pytest.raises(ValueError):
        m.target_row(link, tmp_path, run)
    link.unlink(); os.link(file, link)
    with pytest.raises(ValueError):
        m.target_row(file, tmp_path, run)


def test_deferred_removal_requires_complete_matching_nested_collector(tmp_path):
    m = module()
    q = dict(status='train', source_commit='source', train=[dict(status='complete')]*128)
    result = dict(status='complete', source_commit='source', validated_rate_results=192, failures=[])
    (tmp_path/'QUEUE.json').write_text(json.dumps(q))
    (tmp_path/'RESULT.json').write_text(json.dumps(result))
    assert not m.nested_finished(tmp_path, 'source')
    q['status'] = 'complete'; (tmp_path/'QUEUE.json').write_text(json.dumps(q))
    assert m.nested_finished(tmp_path, 'source')
    assert not m.nested_finished(tmp_path, 'different')
    result['validated_rate_results'] = 191; (tmp_path/'RESULT.json').write_text(json.dumps(result))
    assert not m.nested_finished(tmp_path, 'source')


def test_exact_execution_preserves_masks_and_deferred_weights(tmp_path, monkeypatch):
    m = module()
    root = tmp_path/'authorized'; run = root/'run'; run.mkdir(parents=True)
    deleted, kept, mask = run/'last.pt', run/'best_epoch_1.pt', run/'fixed_masks.pt'
    deleted.write_bytes(b'obsolete model'); kept.write_bytes(b'dependency'); mask.write_bytes(b'mask')
    monkeypatch.setattr(m, 'ROOTS', {root})
    monkeypatch.setattr(m, 'BASE', tmp_path)
    monkeypatch.setattr(m, 'live_paths', lambda: ([], set()))
    plan = dict(status='planned', source_sha256=m.sha(Path(m.__file__)),
        immediate=[m.target_row(deleted, root, run)], deferred=[m.target_row(kept, root, run)],
        retained_evidence=[dict(path=str(mask), sha256=m.sha(mask))])
    manifest = tmp_path/'plan.json'; manifest.write_text(json.dumps(plan))
    m.execute(manifest, 'immediate', tmp_path/'receipt')
    assert not deleted.exists()
    assert kept.read_bytes() == b'dependency'
    assert mask.read_bytes() == b'mask'
    assert json.loads((tmp_path/'receipt/RECEIPT.json').read_text())['deleted_count'] == 1
