"""Frozen CHM diagnostics: isolate historical writer gradients and replay reads."""
import importlib

import pytest
import torch


def core():
    return importlib.import_module('experiments.osram_current_pattern_history_audit_20261022.core')


@pytest.fixture
def case():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    torch.manual_seed(66)
    torch.set_num_threads(1)
    cfg = TrainConfig(dataset='CMUMOSI', backbone_type='osram',
        training_objective='emotion-only', disable_unused_aux_modules=True,
        latent_dim=8, osram_output_dim=16, osram_num_heads=2,
        osram_key_dim=4, osram_value_dim=4, osram_bidirectional=False,
        osram_write_step=.6)
    model = _build_model(cfg, (4, 5, 6), torch.device('cpu')).requires_grad_(True).eval()
    # A trained checkpoint has a nonzero adapter. Random OSRAM's zero final
    # adapter intentionally disconnects sentiment from all Memory parameters.
    torch.nn.init.normal_(model.osram.emotion_adapter[-1].weight, std=.1)
    complete = torch.randn(4, 2, 15)
    availability = torch.tensor([[[1., 1., 0.], [0., 1., 1.]],
                                 [[1., 0., 1.], [1., 0., 0.]],
                                 [[0., 1., 0.], [1., 1., 1.]],
                                 [[1., 1., 1.], [0., 0., 0.]]])
    mask = torch.tensor([[1., 1., 1., 1.], [1., 1., 1., 0.]])
    expanded = torch.repeat_interleave(availability.bool(), torch.tensor([4, 5, 6]), -1)
    view = dict(complete=complete, incomplete=torch.where(expanded, complete, 0.),
        availability=availability, qmask=torch.zeros(2, 4),
        umask=mask, labels=torch.tensor([[1., -1., 1., -.5], [-1., 1., -.5, 0.]]),
        lengths=[4, 3], conversation_ids=['a', 'b'])
    return model, view


def test_prefix_patterns_preserve_full_history_and_source_tensors(case):
    _, background = case
    module = core()
    assert module.PATTERNS == ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')
    for pattern in module.PATTERNS:
        selected = module.prefix_for_pattern(background, 1, 2, (4, 5, 6), pattern)
        assert selected['incomplete'].shape == (3, 1, 15)
        assert selected['lengths'] == [3] and selected['conversation_ids'] == ['b']
        assert torch.equal(selected['incomplete'][:-1], background['incomplete'][:2, 1:2])
        assert torch.equal(selected['availability'][:-1], background['availability'][:2, 1:2])
        assert torch.equal(selected['qmask'], background['qmask'][1:2, :3])
        assert torch.equal(selected['labels'], background['labels'][1:2, :3])
        assert selected['availability'][-1, 0].tolist() == list(module.pattern_bits(pattern))
    with pytest.raises(ValueError, match='valid'):
        module.prefix_for_pattern(background, 1, 3, (4, 5, 6), 'A')
    with pytest.raises(ValueError):
        module.pattern_bits('TA')


def test_masked_nan_current_features_are_safely_zeroed(case):
    _, background = case
    background['complete'][-1, 0, 4:] = float('nan')
    selected = core().prefix_for_pattern(background, 0, 3, (4, 5, 6), 'A')
    assert torch.isfinite(selected['incomplete']).all()
    assert not selected['incomplete'][-1, 0, 4:].any()


def test_seven_modes_have_exact_same_past_and_finite_historical_writer_gradients(case):
    model, background = case
    module = core()
    named = module.write_parameters(model)
    assert named and all(n.startswith(('osram.key_projectors.', 'osram.value_projectors.')) for n, _ in named)
    assert len({id(p) for _, p in named}) == len(named)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    results = []
    for pattern in module.PATTERNS:
        view = module.prefix_for_pattern(background, 0, 3, (4, 5, 6), pattern)
        result = module.target_gradient(model, view, named)
        assert result['calls'] == 4
        assert result['loss'] == pytest.approx(result['residual']**2)
        assert all(g is None or torch.isfinite(g).all() for g in result['gradients'])
        assert any(g is not None and g.abs().sum() > 0 for g in result['gradients'])
        results.append(result)
    for result in results[1:]:
        torch.testing.assert_close(result['memory'], results[0]['memory'], rtol=0, atol=0)
        torch.testing.assert_close(result['past_prediction'], results[0]['past_prediction'], rtol=0, atol=0)
    assert all(torch.equal(before[n], p) and p.grad is None for n, p in model.named_parameters())
    assert torch.equal(torch.get_rng_state(), rng)
    geometry = module.geometry(named, results[0]['gradients'], results[1]['gradients'])
    assert geometry['overall']['cosine'] is not None
    assert geometry['groups']['WriteKey']['visible_norm'] > 0
    assert geometry['groups']['WriteValue']['visible_norm'] > 0


def test_current_key_address_derivative_is_excluded_but_forward_unchanged(case):
    from experiments.osram_tasr_20261018.runner_helpers import forward
    from experiments.osram_text_condition_optimization_audit_20261022.run import isolated_current_encoding
    model, background = case
    module = core()
    view = module.prefix_for_pattern(background, 0, 3, (4, 5, 6), 'A')
    original = model.osram._project_sequence
    with isolated_current_encoding(model), module.historical_writer_only(model):
        nodes = []
        wrapped = model.osram._project_sequence
        def inspect(*args, **kwargs):
            result = wrapped(*args, **kwargs)
            nodes.append(result[0])
            return result
        model.osram._project_sequence = inspect
        try:
            actual = forward(model, view)
        finally:
            model.osram._project_sequence = wrapped
    assert model.osram._project_sequence == original
    named = module.write_parameters(model)
    current_grad = torch.autograd.grad(nodes[0]['audio'][-1].sum(),
        [p for _, p in named], allow_unused=True, retain_graph=True)
    assert all(g is None or not g.any() for g in current_grad)
    past_grad = torch.autograd.grad(nodes[0]['audio'][:-1].sum(),
        [p for _, p in named], allow_unused=True)
    assert any(g is not None and g.any() for g in past_grad)
    with isolated_current_encoding(model), torch.no_grad():
        expected = forward(model, view)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_first_utterance_has_zero_historical_writer_gradient(case):
    model, background = case
    module = core()
    view = module.prefix_for_pattern(background, 0, 0, (4, 5, 6), 'A')
    result = module.target_gradient(model, view)
    assert result['calls'] == 1
    assert not result['memory'].any()
    assert all(g is None or not g.any() for g in result['gradients'])


def test_zero_initialized_adapter_does_not_falsely_report_writer_signal(case):
    model, background = case
    module = core()
    torch.nn.init.zeros_(model.osram.emotion_adapter[-1].weight)
    view = module.prefix_for_pattern(background, 0, 3, (4, 5, 6), 'A')
    result = module.target_gradient(model, view)
    assert all(g is None or not g.any() for g in result['gradients'])
    geometry = module.geometry(module.write_parameters(model), result['gradients'], result['gradients'])
    assert geometry['overall']['cosine'] is None


def test_context_overrides_restore_on_exception_and_require_eval(case):
    model, _ = case
    module = core()
    project = model.osram._project_sequence
    scan = model.osram._scan
    with pytest.raises(RuntimeError, match='forced'):
        with module.historical_writer_only(model):
            raise RuntimeError('forced')
    assert model.osram._project_sequence == project
    states = torch.zeros(1, 1, 2, 4, 4)
    with pytest.raises(RuntimeError, match='forced'):
        with module.fixed_history_reads(model.osram, states):
            raise RuntimeError('forced')
    assert model.osram._scan == scan
    model.train()
    with pytest.raises(ValueError, match='eval'):
        with module.historical_writer_only(model):
            pass


def test_fixed_state_replay_reproduces_factual_and_never_writes(case):
    from experiments.osram_tasr_20261018.runner_helpers import forward
    from experiments.osram_text_condition_optimization_audit_20261022.run import isolated_current_encoding
    model, background = case
    module = core()
    view = module.prefix_for_pattern(background, 0, 3, (4, 5, 6), 'AT')
    states = []
    original = model.osram.block_write
    def capture(memory, *args, **kwargs):
        states.append(memory.detach().clone())
        return original(memory, *args, **kwargs)
    model.osram.block_write = capture
    with isolated_current_encoding(model), torch.no_grad():
        expected = forward(model, view)
    model.osram.block_write = original
    states = torch.stack(states)
    before = states.clone()
    def forbidden(*args, **kwargs):
        raise AssertionError('Fixed-state replay must not write')
    model.osram.block_write = forbidden
    try:
        with isolated_current_encoding(model), module.fixed_history_reads(model.osram, states), torch.no_grad():
            actual = forward(model, view)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        assert torch.equal(states, before)
        with isolated_current_encoding(model), module.fixed_history_reads(model.osram, torch.zeros_like(states)), torch.no_grad():
            empty = forward(model, view)
        # All zero supplied memories remove all history without changing Local.
        ablation = model.osram.osram_emotion_ablation
        model.osram.osram_emotion_ablation = 'local-only'
        try:
            with isolated_current_encoding(model), module.fixed_history_reads(model.osram, states), torch.no_grad():
                local = forward(model, view)
        finally:
            model.osram.osram_emotion_ablation = ablation
        torch.testing.assert_close(local, empty, rtol=0, atol=0)
    finally:
        model.osram.block_write = original


def test_fixed_state_shape_validation_and_padding(case):
    from experiments.osram_tasr_20261018.runner_helpers import forward
    model, background = case
    module = core()
    with pytest.raises(ValueError, match='shape'):
        with module.fixed_history_reads(model.osram, torch.zeros(4, 2, 1)):
            pass
    states = torch.randn(4, 2, 2, 4, 4)
    view = dict(background)
    with module.fixed_history_reads(model.osram, states), torch.no_grad():
        result = forward(model, view)
    # Classifier bias can make padded logits nonzero; OSRAM hidden must be zero.
    assert torch.isfinite(result).all()
    with module.fixed_history_reads(model.osram, states):
        keys = {name: torch.zeros(4, 2, 2, 4) for name in ('audio', 'text', 'visual')}
        queries = torch.ones(4, 2, 4, 2, 4)
        base, gap, _ = model.osram._scan(keys, keys, queries, background['availability'],
            background['umask'].T.bool(), False)
        assert not base[3, 1].any() and not gap[3, 1].any()
        assert not gap[background['availability'].bool()].any()
        with pytest.raises(ValueError, match='causal'):
            model.osram._scan(keys, keys, queries, background['availability'],
                background['umask'].T.bool(), True)
