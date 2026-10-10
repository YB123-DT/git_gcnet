import ast
from dataclasses import asdict, make_dataclass, replace
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_seed65_pair_changes_only_seed_and_nested_switch():
    path = ROOT/'experiments/osram_nested_training_gradients_20261010/run.py'
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == 'select_config']
    namespace = {'replace': replace}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    raw = json.loads((ROOT/'experiments/osram_nested_training_gradients_20261010/flat_FINAL_PROVENANCE.json').read_text())['effective_config']
    cfg = make_dataclass('Config', [(key, object) for key in raw])(**raw)
    select = namespace['select_config']
    assert asdict(select(cfg, 'flat', 'constant', seed=65)) == dict(raw, seed=65)
    assert asdict(select(cfg, 'nested', 'constant', seed=65)) == dict(
        raw, seed=65, osram_meaningful_block='nested_gnn_rooted_evidence')
    assert asdict(select(cfg, 'flat', 'constant')) == raw
