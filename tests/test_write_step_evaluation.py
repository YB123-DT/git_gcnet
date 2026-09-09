import pytest

from gcnet_missing_m3 import evaluate_write_intervention as evaluator


def test_explicit_step_override_is_recordable_without_mutating_saved_config():
    args = evaluator.build_parser().parse_args([
        '--checkpoint', 'best.pt', '--feature-root', 'features',
        '--output-dir', 'out', '--modes', 'reference', '--evaluation-write-step', '1',
    ])
    assert args.evaluation_write_step == 1.
    assert evaluator.resolve_evaluation_step(.6, None, ['reference']) == .6
    assert evaluator.resolve_evaluation_step(.6, 1., ['reference']) == 1.


def test_legacy_interventions_cannot_accidentally_double_scale_native_step():
    with pytest.raises(ValueError):
        evaluator.resolve_evaluation_step(.6, None, ['fixed0.6'])
    with pytest.raises(ValueError):
        evaluator.resolve_evaluation_step(.6, 1., ['fixed0.6'])
    assert evaluator.resolve_evaluation_step(1., None, ['fixed0.6']) == 1.
    for bad in (float('nan'), -1., 1.1):
        with pytest.raises(ValueError):
            evaluator.resolve_evaluation_step(1., bad, ['reference'])
