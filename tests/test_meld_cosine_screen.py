import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]


def runner():
    try:
        return importlib.import_module('experiments.osram_meld_cosine_20261017.run')
    except ModuleNotFoundError:
        pytest.fail('Single MELD cosine screen is not implemented')


def test_only_existing_schedule_switch_changes():
    base = json.loads((ROOT/'experiments/osram_meld_memory_sensitivity_20261017/FROZEN_SUMMARY.json').read_text())['source_config']
    original = copy.deepcopy(base)
    actual = runner().make_config(base)
    assert base == original
    assert actual == dict(original, lr_schedule='cosine')
    assert actual['warmup_ratio'] == .05 and actual['epochs'] == 50
    with pytest.raises(ValueError):
        runner().make_config(dict(base, missing_mask_protocol='conversation-mask-v2'))


def test_curve_is_existing_two_epoch_warmup_then_cosine_to_zero():
    import ast
    import math
    curve = runner().expected_lrs(.001, 50, .05)
    assert len(curve) == 50
    assert curve[:2] == [.0005, .001] and curve[-1] == 0.
    assert all(a >= b for a, b in zip(curve[1:], curve[2:]))
    path = ROOT/'gcnet_missing_m3/train_gcnet.py'
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == '_apply_epoch_learning_rate']
    namespace = dict(math=math, torch=torch, TrainConfig=object)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    optimizer = SimpleNamespace(param_groups=[dict(lr=.001)])
    config = SimpleNamespace(epochs=50,warmup_ratio=.05,lr_schedule='cosine')
    for epoch,expected in enumerate(curve):
        namespace['_apply_epoch_learning_rate'](optimizer,config,epoch)
        assert math.isclose(optimizer.param_groups[0]['lr'], expected, rel_tol=1e-12, abs_tol=1e-15)


def capture(epoch, ua_better=False):
    # Real metrics are evaluated from stored logits, never an extra forward.
    from experiments.osram_meld_write_tuning_20261017.capture import _metrics
    logits = np.zeros((7, 7), dtype=np.float32)
    logits[:, 0] = 1.
    if ua_better:
        logits = np.eye(7, dtype=np.float32)*2
    artifacts = dict(logits=logits, labels=np.arange(7), availability=np.ones((7, 3)),
                     predictions=logits.argmax(1), conversation_ids=np.array(['one']*7),
                     utterance_indices=np.arange(7), speakers=np.zeros(7))
    return SimpleNamespace(latest={str(i/10): dict(epoch=epoch, artifacts=artifacts,
                           scores=_metrics(artifacts)) for i in range(8)})


def test_ua_selection_keeps_same_forward_checkpoint_and_first_tie(tmp_path):
    run = runner()
    from gcnet_missing_m3.training_resume import TrainingState
    model = torch.nn.Linear(3, 7)
    optimizer = torch.optim.Adam(model.parameters())
    observer = run.UASelection(TrainingState, tmp_path/'ua_best', {'test': 'identity'})
    observer.bind(model, optimizer)
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    observer.update(capture(1), 1, {'lr_schedule': 'cosine'})
    observer.update(capture(2), 2, {'lr_schedule': 'cosine'})
    assert all(v['epoch'] == 1 for v in observer.scores.values())
    observer.update(capture(3, True), 3, {'lr_schedule': 'cosine'})
    assert all(v['epoch'] == 3 for v in observer.scores.values())
    assert torch.equal(torch_rng, torch.get_rng_state())
    now = np.random.get_state()
    assert np.array_equal(numpy_rng[1], now[1]) and numpy_rng[2:] == now[2:]
    checkpoint = torch.load(tmp_path/'ua_best/best_miss_0p0.pt', weights_only=False)
    assert checkpoint['selection_metric'] == 'ua'
    assert checkpoint['selected_metrics']['UA'] == 100.
    with np.load(tmp_path/'ua_best'/checkpoint['selection_artifact']['path']) as saved:
        assert int(saved['selected_epoch']) == 3
        np.testing.assert_array_equal(saved['logits'], capture(3, True).latest['0.0']['artifacts']['logits'])
    assert len(observer.rows) == 3
    observer.prune()
    clone = run.UASelection(TrainingState, tmp_path/'ua_best', {'test': 'identity'})
    clone.bind(model, optimizer)
    clone.restore(observer.state_dict())
    assert clone.scores == observer.scores and clone.rows == observer.rows


def test_incomplete_epoch_cannot_update_ua_state(tmp_path):
    from gcnet_missing_m3.training_resume import TrainingState
    observer = runner().UASelection(TrainingState, tmp_path, {'test': True})
    model = torch.nn.Linear(3, 7)
    observer.bind(model, torch.optim.Adam(model.parameters()))
    row = capture(1)
    del row.latest['0.7']
    with pytest.raises(ValueError, match='eight|epoch'):
        observer.update(row, 1, {})
    assert observer.scores == {}


def test_scheduler_authorizes_one_job_on_gpu5_only(tmp_path):
    scheduler = runner().CosineScheduler(tmp_path/'queue', source=tmp_path/'b4', control=tmp_path/'W0')
    assert scheduler.gpus == (5,) and scheduler.capacity == {5: 1}
    command = scheduler.command({'id':'C0', 'output':str(tmp_path/'C0')}, tmp_path/'config.json', 'GPU-safe')
    assert '--worker' in command
    assert command[command.index('--gpu-uuid')+1] == 'GPU-safe'
    assert command[command.index('--control')+1] == str(tmp_path/'W0')
