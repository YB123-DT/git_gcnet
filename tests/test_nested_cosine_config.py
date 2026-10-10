import ast
from dataclasses import asdict, make_dataclass, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(functions) == 1, f'Missing requested function {name}'
    namespace = {'replace': replace, 'math': math, 'TrainConfig': object,
                 'torch': SimpleNamespace(optim=SimpleNamespace(Optimizer=object))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_only_schedule_and_nested_switch_change():
    select = load_function(ROOT/'experiments/osram_nested_training_gradients_20261010/run.py', 'select_config')
    raw = json.loads((ROOT/'experiments/osram_nested_training_gradients_20261010/flat_FINAL_PROVENANCE.json').read_text())['effective_config']
    cfg = make_dataclass('Config', [(key, object) for key in raw])(**raw)
    assert asdict(select(cfg, 'flat', 'constant')) == raw
    for model in ['flat', 'nested']:
        actual = asdict(select(cfg, model, 'cosine'))
        expected = dict(raw, lr_schedule='cosine')
        if model == 'nested':
            expected['osram_meaningful_block'] = 'nested_gnn_rooted_evidence'
        assert actual == expected
        assert actual['warmup_ratio'] == .05


def test_existing_cosine_actual_epoch_values_and_group_ratios():
    apply = load_function(ROOT/'gcnet_missing_m3/train_gcnet.py', '_apply_epoch_learning_rate')
    cfg = SimpleNamespace(epochs=100, warmup_ratio=.05, lr_schedule='cosine')
    optimizer = SimpleNamespace(param_groups=[{'lr': .001}, {'lr': .0005}])
    rates = []
    for epoch in range(100):
        apply(optimizer, cfg, epoch)
        rates.append(optimizer.param_groups[0]['lr'])
        assert optimizer.param_groups[1]['lr'] == .5*rates[-1]
    assert rates[0] == .0002 and rates[4] == .001 and rates[-1] == 0.
    assert all(a >= b for a, b in zip(rates[4:], rates[5:]))
