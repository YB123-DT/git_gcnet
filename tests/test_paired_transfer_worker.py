import hashlib
import importlib

import pytest
import torch


def module():
    return importlib.import_module('experiments.osram_paired_meld_iemocap6_20261018.worker')


def test_identity_excludes_only_inert_training_projector():
    m=module()
    model=torch.nn.Module()
    model.core=torch.nn.Linear(3,2)
    before=m.backbone_identity(model)
    model.history_contrast_projector=torch.nn.Linear(2,9)
    assert m.backbone_identity(model)==before
    with torch.no_grad():model.core.weight.add_(1)
    assert m.backbone_identity(model)!=before


def test_epoch_diagnostics_enforce_task_only_weighted_loss():
    m=module()
    row=dict(loss=2.,paired_history=dict(task_view1=1.,task_view2=3.,
        task_view2_weight=.5,observed_before=10,observed_dropped=2,anchor_count=3))
    assert m.check_epoch(row)['actual_drop_ratio']==.2
    row['loss']=3.
    with pytest.raises(ValueError,match='task'):
        m.check_epoch(row)


def test_queue_waits_for_both_real_first_epochs(tmp_path):
    import json
    m=importlib.import_module('experiments.osram_paired_meld_iemocap6_20261018.run')
    assert not m.integration_ready(tmp_path)
    for name in m.GATE_IDS:
        p=tmp_path/name/'FIRST_EPOCH_INTEGRATION.json'
        p.parent.mkdir()
        p.write_text(json.dumps(dict(status='complete')))
    assert m.integration_ready(tmp_path)
    report=m.render(dict(status='running',datasets={}),m.baseline_scores())
    assert 'Status: running' in report and 'INTERNAL TEST-ORACLE' in report
