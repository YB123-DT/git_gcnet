import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / 'experiments/osram_neumiss_stage2_3seed_20261025/pause_signature.py'
spec = importlib.util.spec_from_file_location('pause_signature', PATH)
pause = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pause)


def test_exact_target_matching(tmp_path):
    output = str(pause.ROOT / 'train' / 'IEMOCAPSix_66_fold5_0.0')
    args = ['python', '-m', pause.WORKER, '--output', output, '--source-commit', pause.SHA]
    assert pause.matches(args, pause.WORKER, '--output', output)
    assert not pause.matches(args, pause.WORKER + '.wrong', '--output', output)
    assert not pause.matches(args, pause.WORKER, '--output', output + '_other')
    assert not pause.matches(args[:-1] + ['wrong'], pause.WORKER, '--output', output)


def test_full_state_validation():
    state = dict(block={'w': 1}, optimizer={'state': {}, 'param_groups': [{}]}, epoch=2,
                 history=[{}, {}], identity={'source_commit': pause.SHA, 'parent': {'checkpoint': 'p'}},
                 rng={name: object() for name in ('python', 'numpy', 'torch', 'cuda')},
                 best={'epoch': 1, 'checkpoint': 'best.pt', 'predictions': 'pred.npz',
                       'checkpoint_sha256': 'x', 'prediction_sha256': 'y'})
    pause.validate_state(state)
    for key in ('block', 'optimizer', 'rng', 'identity', 'history', 'best'):
        broken = dict(state)
        del broken[key]
        with pytest.raises(ValueError):
            pause.validate_state(broken)
    with pytest.raises(ValueError):
        pause.validate_state(dict(state, epoch=3))
    with pytest.raises(ValueError):
        pause.validate_state(dict(state, rng={'torch': 1}))
