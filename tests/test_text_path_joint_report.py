"""Reports reject incomplete jobs and unequal evaluation cohorts/masks."""
import json

import numpy as np
import pytest

from experiments.osram_text_path_router_joint_20261019.report import aggregate


def _runs(tmp_path):
    score = dict(ACC=50., UA=40., WF1=45., CE=1., N=2)
    arrays = dict(labels=np.array([0, 1]), speakers=np.array([0, 1]),
                  conversation_ids=np.array(['a', 'a']), utterance_indices=np.array([0, 1]),
                  availability=np.array([[1., 0., 1.], [0., 1., 0.]]))
    for dataset in ('MELD', 'IEMOCAPSix'):
        for mode in ('flat', 'joint'):
            output = tmp_path / 'runs' / dataset / mode
            (output / 'evaluation').mkdir(parents=True)
            summary = dict(status='complete', phase='train', mode=mode, selected_epoch=2,
                           initial_backbone_sha256='same-initialization', validation_mask_hashes={'0.3':'same'},
                           evaluation={'0.3': {'router': score}},
                           aggregate={'router': {'mean': score, 'high': score}})
            (output / 'SUMMARY.json').write_text(json.dumps(summary))
            (output / 'history.json').write_text(json.dumps([{'mask_trace_sha256':'same-training-mask'}]))
            np.savez(output / 'evaluation' / 'random_0p3.npz', **arrays)
    return tmp_path / 'runs'


def test_report_checks_masks_and_preserves_protocol_warning(tmp_path):
    root = _runs(tmp_path)
    aggregate(root, tmp_path / 'report')
    text = (tmp_path / 'report' / 'RESULT.md').read_text()
    assert 'Not per-rate Test-best' in text
    assert 'Within Joint' in text
    assert text.endswith('\n') and not text.endswith('\n\n')
    assert len((tmp_path / 'report' / 'METRICS.csv').read_text().splitlines()) == 5


def test_report_does_not_rank_incomplete_training(tmp_path):
    root = _runs(tmp_path)
    path = root / 'MELD' / 'joint' / 'SUMMARY.json'
    summary = json.loads(path.read_text())
    summary['status'] = 'running'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Incomplete'):
        aggregate(root, tmp_path / 'report')


def test_report_rejects_same_seed_different_mask(tmp_path):
    root = _runs(tmp_path)
    path = root / 'MELD' / 'joint' / 'evaluation' / 'random_0p3.npz'
    with np.load(path) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays['availability'] = arrays['availability'].copy()
    arrays['availability'][0] = [0., 1., 0.]
    np.savez(path, **arrays)
    with pytest.raises(AssertionError):
        aggregate(root, tmp_path / 'report')


def test_finite_queue_rejects_forbidden_and_duplicate_gpu():
    from experiments.osram_text_path_router_joint_20261019.orchestrate import BAD_UUID, devices
    for values in ([BAD_UUID, 'GPU-good'], ['GPU-same', 'GPU-same']):
        with pytest.raises(ValueError, match='healthy'):
            devices(values)


def test_finite_queue_resolves_uuid_not_logical_cuda_index(monkeypatch):
    from experiments.osram_text_path_router_joint_20261019 import orchestrate
    monkeypatch.setattr(orchestrate.subprocess, 'check_output', lambda *args, **kwargs:
                        '0, GPU-first, 0, 32768\n2, GPU-second, 512, 32768\n')
    found = orchestrate.devices(['GPU-second', 'GPU-first'])
    assert found['GPU-second']['index'] == 2
    assert found['GPU-first']['used_mib'] == 0
