import importlib
from types import SimpleNamespace

import pytest

PACKAGE = 'experiments.osram_frozen_nested_cross_20261025'


def test_only_authorized_56_cells_and_constant_learning_rates():
    p = importlib.import_module(PACKAGE + '.protocol')
    cells = [(d, f, r) for d in p.DATASETS
             for f in (range(1, 6) if d == 'IEMOCAPSix' else (1,)) for r in p.RATES]
    assert len(cells) == 56
    for d, f, r in cells:
        assert p.validate_scope(d, 66, f, r) == r
        assert [p.learning_rate(d, e) for e in range(p.budget(d))] == [3e-4] * p.budget(d)
    assert [p.budget(d) for d in ('CMUMOSEI', 'MELD', 'IEMOCAPSix')] == [60, 50, 100]
    for args in [('CMUMOSI', 66, 1, '.3'), ('MELD', 67, 1, '.3'),
                 ('CMUMOSEI', 66, 2, '.3'), ('IEMOCAPSix', 66, 6, '.3'),
                 ('MELD', 66, 1, True), ('MELD', 66, 1, '.8')]:
        with pytest.raises(ValueError): p.validate_scope(*args)
    with pytest.raises(ValueError): p.learning_rate('MELD', 50)


def test_reference_names_and_strict_same_epoch_metric_selection():
    p = importlib.import_module(PACKAGE + '.protocol')
    assert p.reference_dir('IEMOCAPSix', 5, '.7').name == 'IEMOCAPSix_66_fold5_0.7'
    assert p.reference_dir('MELD', 1, '.3').name == 'MELD_66_0.3'
    assert p.source_root('IEMOCAPSix').name == 'source_21e46f2'
    assert p.source_commit('MELD').startswith('e0ca94c')
    for dataset in p.DATASETS:
        key = 'WF1' if dataset == 'CMUMOSEI' else 'ACC'
        assert p.selection_metric(dataset) == key
        assert p.improves(dataset, {key: 2}, {key: 1})
        assert not p.improves(dataset, {key: 2}, {key: 2})


@pytest.mark.parametrize('dataset,classes', [('IEMOCAPSix', 6), ('MELD', 7)])
def test_evaluation_preserves_class_logits_and_all_array_fields(dataset, classes):
    import numpy as np
    import torch
    run = importlib.import_module(PACKAGE + '.run')
    logits = torch.zeros(2, 2, classes)
    logits[0, 0, 1] = 3
    logits[1, 1, 2] = 4
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Linear(1, 1)
        def predict_frame(self, frame): return frame.prediction
    frame = SimpleNamespace(valid=torch.eye(2, dtype=torch.bool), prediction=logits)
    frame.to = lambda device: frame
    rows = [dict(frame=frame, availability=np.ones((2, 3)),
                 metadata=dict(labels=np.array([1, 2]), utterance=np.array(['a', 'b'])))]
    metrics, arrays = run.evaluate(dataset, Model(), rows)
    assert arrays['prediction'].shape == (2, classes)
    assert metrics['ACC'] == 100
    assert set(arrays) == {'prediction', 'base_prediction', 'labels', 'utterance', 'availability'}


def test_reference_checks_complete_parent_config_fold_mask_and_source():
    p = importlib.import_module(PACKAGE + '.protocol')
    parent, config = {'parent_id': 'accepted'}, {'dataset': 'MELD', 'fold': 1}
    identity = dict(mode='signature', dataset='MELD', seed=66, evaluation_rate='0.3',
                    source_commit=p.numerical_source('MELD'), smoke=False, stage2_epochs=50,
                    cyclic_training=True, training_rates=list(p.RATES),
                    mask_protocol='gcnet-random-mask-v1', effective_stage2_config=config,
                    test_mask_sha256='mask', parent=parent)
    reference = dict(status='complete', epoch=50, total_parameters=281856,
                     parent_id='accepted', identity=identity)
    p.validate_reference(reference, parent, config, 'mask', 'MELD', 1, '0.3')
    for key, wrong in [('source_commit', 'new'), ('smoke', True),
                       ('parent', {'parent_id': 'other'}), ('test_mask_sha256', 'other')]:
        changed = dict(reference, identity=dict(identity, **{key: wrong}))
        with pytest.raises(ValueError):
            p.validate_reference(changed, parent, config, 'mask', 'MELD', 1, '0.3')


def test_trace_and_recovery_metadata_are_preserved():
    p = importlib.import_module(PACKAGE + '.protocol')
    run = importlib.import_module(PACKAGE + '.run')
    trace = dict(epoch=50, optimizer_steps=1, training_rates=[.1], mask_trace_sha256='mask',
                 N=3, learning_rate=3e-4)
    p.validate_epoch_trace('MELD', trace, trace)
    with pytest.raises(ValueError):
        p.validate_epoch_trace('MELD', dict(trace, mask_trace_sha256='wrong'), trace)
    previous = dict(seconds=14., peak_memory_mib=900., peak_rss_gib=2.,
                    baseline_artifact={'sha256': 'accepted'})
    summary = run.starting_summary(previous, 'MELD', 66, 1, '.3', 'source')
    assert summary['baseline_artifact'] == previous['baseline_artifact']
    assert run.costs(previous, 5., 800., 3.) == dict(seconds=19., peak_memory_mib=900., peak_rss_gib=3.)


def test_mosei_binds_verified_manifest_before_trainer_import(tmp_path, monkeypatch):
    import os
    run = importlib.import_module(PACKAGE + '.run')
    imports, configured = [], []
    def imported(name):
        imports.append(name)
        return SimpleNamespace(load_manifest=lambda: ({'dataset_root': '/accepted/mosei'}, {}))
    monkeypatch.setattr(run.importlib, 'import_module', imported)
    monkeypatch.setenv('GCNET_MELD_PICKLE', '/wrong/meld')
    monkeypatch.setenv('GCNET_DATASET_ROOT', '/wrong/dataset')
    monkeypatch.setenv('GCNET_CACHE_ROOT', '/wrong/cache')
    prepare = SimpleNamespace(configure_environment=lambda *args: configured.append(args))
    run.configure_environment('CMUMOSEI', prepare, tmp_path/'run')
    assert os.environ['GCNET_DATASET_ROOT'] == '/accepted/mosei'
    assert os.environ['GCNET_CACHE_ROOT'] == str(tmp_path/'feature_cache'/'CMUMOSEI')
    assert 'GCNET_MELD_PICKLE' not in os.environ
    assert imports == ['experiments.osram_mosei_gcnet_router_20261020.worker']
    assert configured == [('CMUMOSEI', tmp_path/'run')]


def test_mosei_metrics_exclude_neutral_only_from_classification():
    p = importlib.import_module(PACKAGE + '.protocol')
    scores = p.measured_metrics('CMUMOSEI', [-1., 0., 1.], [-1., 3., 1.], [-1., 0., 1.])
    assert scores['N'] == 2 and scores['ACC'] == 100
    assert scores['MSE'] == 3.


@pytest.mark.parametrize('dataset,classes', [('IEMOCAPSix', 6), ('MELD', 7)])
def test_real_original_nested_classification_ce_updates_only_block(dataset, classes):
    import torch
    from experiments.osram_frozen_nested_gcnet_20261025.model import FrozenNestedModel
    from experiments.osram_recent_noarea_serial_20261025 import stage2
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from tests.test_frozen_nested_gcnet_model import pretrained_parent, parent_call
    from tests.test_three_block_joint_model import view
    from tests.test_recent_noarea_serial_stage2 import toy_bundle
    torch.set_num_threads(1)
    torch.manual_seed(47)
    parent = pretrained_parent()
    parent.smax_fc = torch.nn.Linear(1600, classes)
    model = FrozenNestedModel(parent).train()
    data = dict(view(2), labels=torch.tensor([[0, 1, 2], [2, 1, 0]]), conversation_ids=['a', 'b'])
    expected = parent_call(parent, data)[0]
    assert torch.equal(model(data)['prediction'], expected)
    bundle = toy_bundle()
    bundle['cfg'].dataset = dataset
    def loss(task_dataset, prediction, labels, umask, *args):
        assert task_dataset == dataset and prediction.shape == (3, 2, classes)
        valid = umask.bool()
        return torch.nn.functional.cross_entropy(prediction.transpose(0, 1)[valid], labels[valid])
    bundle['trainer'] = SimpleNamespace(
        _schedules=lambda cfg, split: {r: r for r in cfg.train_missing_rates},
        _move_batch=lambda raw, device: raw,
        _prepare_view=lambda *args: data, _task_loss=loss)
    digest = h.backbone_digest(parent)
    optimizer = stage2.build_optimizer(model)
    with torch.backends.mkldnn.flags(enabled=False):
        row = stage2.train_epoch(bundle, model, optimizer, 0, '0.3', True)
    assert row['live_scan_calls'] == [1]
    assert row['updated_parameters'] > 0 and row['nonzero_gradient_batches'] == 1
    assert h.backbone_digest(parent) == digest
    assert all(p.grad is None for p in parent.parameters())
