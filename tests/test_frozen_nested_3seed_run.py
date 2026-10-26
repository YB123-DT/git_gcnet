import importlib
import json

import pytest


MODULE = 'experiments.osram_frozen_nested_3seed_20261026.run'


def test_regression_and_classification_shapes_remain_distinct():
    from types import SimpleNamespace
    import numpy as np
    import torch
    run = importlib.import_module(MODULE)

    class Model:
        block = torch.nn.Linear(1, 1)
        def eval(self):
            return self
        def predict_frame(self, frame):
            return frame.prediction

    class Frame(SimpleNamespace):
        def to(self, device):
            return self

    for dataset, width in [('CMUMOSI', 1), ('CMUMOSEI', 1), ('MELD', 7), ('IEMOCAPSix', 6)]:
        frame = Frame(prediction=torch.ones(2, 1, width), valid=torch.ones(2, 1, dtype=torch.bool))
        labels = np.array([1, 1])
        _, arrays = run.evaluate(dataset, Model(), [dict(frame=frame,
            metadata=dict(labels=labels), availability=np.ones((2, 3), dtype=bool))])
        assert arrays['prediction'].shape == ((2,) if width == 1 else (2, width))


def test_resume_preserves_artifact_provenance_and_cumulative_cost():
    run = importlib.import_module(MODULE)
    old = dict(status='failed', baseline_artifact={'sha256': 'protected'}, seconds=10,
        peak_memory_mib=50, peak_rss_gib=2, error='interrupted')
    current = run.starting_summary(old, 'MELD', 68, 1, '0.7', 'source')
    assert current['baseline_artifact'] == old['baseline_artifact']
    assert 'error' not in current
    assert current['seed'] == 68
    assert run.costs(old, 5, 40, 3) == dict(seconds=15, peak_memory_mib=50, peak_rss_gib=3)


def test_optional_reference_is_explicit_not_substituted(tmp_path, monkeypatch):
    run = importlib.import_module(MODULE)
    monkeypatch.setattr(run, 'reference_dir', lambda *a: None)
    assert run.load_reference('MELD', 67, 1, '0.3') is None
    monkeypatch.setattr(run, 'reference_dir', lambda *a: tmp_path)
    with pytest.raises(FileNotFoundError):
        run.load_reference('CMUMOSI', 67, 1, '0.3')


def test_reference_requires_complete_contiguous_history(tmp_path, monkeypatch):
    run = importlib.import_module(MODULE)
    monkeypatch.setattr(run, 'reference_dir', lambda *a: tmp_path)
    (tmp_path/'SUMMARY.json').write_text(json.dumps(dict(status='complete', epoch=150)))
    (tmp_path/'history.json').write_text(json.dumps([dict(epoch=1)]))
    with pytest.raises(ValueError, match='history'):
        run.load_reference('CMUMOSI', 67, 1, '0.3')
