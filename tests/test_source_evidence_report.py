"""Reports consume completed frozen statistics, never model or NPZ artifacts."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.osram_12_source_evidence_audit_20261019.analysis import analyze_arrays
from experiments.osram_12_source_evidence_audit_20261019.report import summarize_run


def create_run(root):
    full = np.array([[2., 0.], [0., 2.]])
    arrays = dict(labels=np.array([0, 1]), logits_full=full,
        logits_removed=np.broadcast_to(full[:, None, None], (2, 4, 3, 2)),
        logits_coalitions=np.broadcast_to(full[:, None], (2, 8, 2)),
        source_reads=np.ones((2, 4, 3, 1)), row_active=np.array([[1, 0, 1, 0]] * 2),
        trajectory=np.broadcast_to([1., .5, 1., 1.], (2, 3, 4)),
        state_source_norms=np.ones((2, 3)), state_joint_norm=np.ones(2),
        availability=np.array([[1, 0, 1]] * 2), conversation_ids=np.array(['a', 'b']),
        utterance_indices=np.array([1, 1]))
    stats = analyze_arrays(arrays, 2)
    conditions = {}
    names = [('random', f'{i / 10:.1f}') for i in range(8)]
    names += [('persistent', pattern) for pattern in ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')]
    root.mkdir()
    for index, (group, name) in enumerate(names):
        stats['full_metrics'].update(N=100+index, ACC=60+index, UA=30+index, WF1=55+index, CE=1+.1*index)
        key = group+'_'+name.replace('.', 'p')
        checks = dict(scans=1, source_sum_max_abs_error=index*1e-6,
                      diagnostics=[dict(valid=2, memory_sum_max_abs_error=index*1e-6,
                                        memory_sum_max_relative_error=index*1e-7,
                                        gap_sum_max_abs_error=2e-6,
                                        base_sum_max_abs_error=3e-6,
                                        prewrite_memory_sum_max_abs_error=4e-6)])
        result = dict(source_commit='source-sha', anchor_sha256='checkpoint-sha',
                      group=group, condition=name, checks=checks, statistics=stats)
        (root/(key+'.json')).write_text(json.dumps(result))
        conditions[key] = dict(path=key+'.json', group=group, condition=name,
                               full_metrics=dict(stats['full_metrics']), checks=checks)
    summary = dict(status='complete', dataset='MELD', source_commit='source-sha',
                   anchor=dict(checkpoint_sha256='checkpoint-sha', checkpoint='/protected/best.pt', seed=66, fold=1),
                   conditions=conditions, no_training=True, backbone_unchanged=True,
                   seconds=20., peak_memory_mib=100.)
    (root/'SUMMARY.json').write_text(json.dumps(summary))
    return summary


def test_report_uses_equal_condition_means_and_reports_reconstruction_maxima(tmp_path):
    root = tmp_path/'run'
    create_run(root)
    paths = summarize_run(root, tmp_path/'reports')
    machine = json.loads(Path(paths['json']).read_text())
    assert machine['aggregates']['random8']['ACC'] == pytest.approx(63.5)
    assert machine['aggregates']['high_missing']['ACC'] == pytest.approx(66.)
    assert machine['aggregates']['persistent6']['ACC'] == pytest.approx(70.5)
    assert machine['reconstruction_error_maxima']['memory_sum_max_abs_error'] == pytest.approx(14e-6)
    markdown = Path(paths['result']).read_text()
    assert 'INTERNAL DIAGNOSTIC ONLY' in markdown
    assert 'not pure modality' in markdown
    assert 'No confidence intervals' in markdown
    assert 'Persistent ATV' in markdown
    assert 'Gap-T' in markdown
    assert 'N/A' in markdown
    import csv
    with Path(paths['slots']).open() as handle:
        slots = list(csv.DictReader(handle))
    assert len(slots) == 15*12
    gap_a = next(row for row in slots if row['row'] == 'Gap-A')
    assert gap_a['N'] == '0' and gap_a['delta_CE'] == ''
    with Path(paths['grouped_slots']).open() as handle:
        grouped = list(csv.DictReader(handle))
    assert any(row['stratum_family'] == 'availability' and row['stratum'] == 'AV' for row in grouped)
    with Path(paths['shapley']).open() as handle:
        assert len(list(csv.DictReader(handle))) == 45


def test_incomplete_runs_and_missing_conditions_fail_closed_without_reports(tmp_path):
    root = tmp_path/'run'
    summary = create_run(root)
    summary['status'] = 'running'
    (root/'SUMMARY.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='complete'):
        summarize_run(root)
    assert not (root/'RESULT.md').exists()
    summary['status'] = 'complete'
    summary['conditions'].pop('random_0p7')
    (root/'SUMMARY.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='15'):
        summarize_run(root)


def test_mixed_source_or_checkpoint_versions_are_rejected(tmp_path):
    root = tmp_path/'run'
    create_run(root)
    path = root/'persistent_A.json'
    condition = json.loads(path.read_text())
    condition['source_commit'] = 'changed-version'
    path.write_text(json.dumps(condition))
    with pytest.raises(ValueError, match='identity'):
        summarize_run(root)
