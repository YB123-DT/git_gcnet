import ast
import copy
from dataclasses import asdict, make_dataclass, replace
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def function(name):
    path = ROOT/'experiments/osram_nested_low_lr_extend150_20261010/run.py'
    assert path.exists(), 'Low-LR continuation wrapper missing'
    definitions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(definitions) == 1
    namespace = {'replace': replace}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_only_epochs_and_learning_rate_change():
    raw = json.loads((ROOT/'experiments/osram_nested_training_gradients_20261010/nested_FINAL_PROVENANCE.json').read_text())['effective_config']
    cfg = make_dataclass('Config', [(key, object) for key in raw])(**raw)
    actual = asdict(function('continued_config')(cfg, .0001))
    assert actual == dict(raw, epochs=150, learning_rate=.0001)


def test_other_seed_is_preserved_in_continued_config():
    raw = json.loads((ROOT/'experiments/osram_nested_training_gradients_20261010/nested_FINAL_PROVENANCE.json').read_text())['effective_config']
    raw['seed'] = 67
    cfg = make_dataclass('Config', [(key, object) for key in raw])(**raw)
    assert asdict(function('continued_config')(cfg, .0001)) == dict(raw, epochs=150, learning_rate=.0001)


def test_full_state_retained_except_optimizer_learning_rate():
    state = dict(next_epoch=100, history=[{'epoch': i} for i in range(1,101)],
                 optimizer={'state': {0: {'step': 200, 'exp_avg': [1.,2.], 'exp_avg_sq': [3.,4.]}},
                            'param_groups': [{'lr': .001, 'params': [0], 'betas': (.9,.999)}]},
                 rng={'torch': [7,8,9]}, model={'weights': [4,5]},
                 scheduler=None, scaler=None, best_references={str(i/10): i for i in range(8)},
                 selection_state={'best': .8}, schedule_state={'epoch': 100})
    before = copy.deepcopy(state)
    result = function('lower_optimizer_lr')(state, .0001)
    assert result == [.001]
    expected = copy.deepcopy(before)
    expected['optimizer']['param_groups'][0]['lr'] = .0001
    assert state == expected
