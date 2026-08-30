import pytest
import torch

from gcnet_missing_m3_sdt_backbone.model import (
    PreNormTransformerLayer,
    SDTStyleConversationBackbone,
    SinusoidalPositionEncoding,
)


def _make_batch(
    lengths=(4, 2),
    input_dim=6,
    n_speakers=2,
    total_length=None,
):
    if total_length is None:
        total_length = max(lengths)

    batch_size = len(lengths)
    values = torch.randn(total_length, batch_size, input_dim, dtype=torch.float32)
    umask = torch.zeros(batch_size, total_length, dtype=torch.float32)
    qmask = torch.zeros(batch_size, total_length, dtype=torch.float32)
    for batch_index, length in enumerate(lengths):
        umask[batch_index, :length] = 1.0
        for time_index in range(length):
            qmask[batch_index, time_index] = time_index % n_speakers

    return values, qmask, umask, torch.tensor(lengths, dtype=torch.long)


def _small_model(**overrides):
    kwargs = {
        "input_dim": 6,
        "output_dim": 32,
        "n_speakers": 2,
        "d_model": 16,
        "num_heads": 4,
        "num_layers": 2,
        "ff_dim": 24,
        "dropout": 0.0,
        "max_len": 8,
    }
    kwargs.update(overrides)
    model = SDTStyleConversationBackbone(**kwargs)
    model.eval()
    return model


def test_position_encoding_is_persistent_and_has_expected_shape():
    encoding = SinusoidalPositionEncoding(dim=8, max_len=7)

    assert encoding.pe.shape == (7, 8)
    assert encoding.pe.dtype == torch.float32
    assert "pe" in encoding.state_dict()


@pytest.mark.parametrize(
    ("dim", "max_len"),
    [(0, 8), (-2, 8), (3, 8), (8, 0), (8, -1)],
)
def test_position_encoding_rejects_invalid_configuration(dim, max_len):
    with pytest.raises(ValueError):
        SinusoidalPositionEncoding(dim=dim, max_len=max_len)


def test_position_encoding_rejects_sequences_longer_than_capacity():
    encoding = SinusoidalPositionEncoding(dim=8, max_len=3)

    with pytest.raises(ValueError):
        encoding(torch.zeros(4, 2, 8, dtype=torch.float32))


def test_forward_has_requested_shape_dtype_and_strict_zero_padding():
    torch.manual_seed(1)
    model = _small_model(output_dim=11)
    values, qmask, umask, lengths = _make_batch()

    output = model(values, qmask, umask, lengths)

    assert output.shape == (4, 2, 11)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert torch.count_nonzero(output[2:, 1]).item() == 0
    assert torch.count_nonzero(output[:2, 1]).item() > 0


def test_float64_model_and_inputs_produce_float64_output():
    torch.manual_seed(10)
    model = _small_model().double()
    values, qmask, umask, lengths = _make_batch()

    output = model(values.double(), qmask.double(), umask.double(), lengths)

    assert output.dtype == torch.float64


def test_real_missing_m3_mosi_shapes_regression():
    torch.manual_seed(11)
    model = SDTStyleConversationBackbone(dropout=0.0)
    model.eval()
    values = torch.randn(7, 3, 256, dtype=torch.float32)
    qmask = torch.zeros(3, 7, dtype=torch.long)
    umask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([7, 5, 3], dtype=torch.long)

    output = model(values, qmask, umask, lengths)

    assert output.shape == (7, 3, 250)
    assert output.dtype == torch.float32
    assert torch.count_nonzero(output[5:, 1]).item() == 0
    assert torch.count_nonzero(output[3:, 2]).item() == 0


def test_padding_values_cannot_change_valid_outputs():
    torch.manual_seed(2)
    model = _small_model()
    values, qmask, umask, lengths = _make_batch()
    changed_values = values.clone()
    changed_values[2:, 1] = changed_values[2:, 1] + 10_000.0

    original = model(values, qmask, umask, lengths)
    changed = model(changed_values, qmask, umask, lengths)
    valid = umask.transpose(0, 1).bool()

    assert torch.allclose(original[valid], changed[valid], atol=1e-6, rtol=1e-6)


def test_nan_padding_features_cannot_change_valid_outputs_or_escape_zero_padding():
    torch.manual_seed(21)
    model = _small_model()
    values, qmask, umask, lengths = _make_batch()
    poisoned_values = values.clone()
    poisoned_values[2:, 1] = float("nan")

    original = model(values, qmask, umask, lengths)
    poisoned = model(poisoned_values, qmask, umask, lengths)
    valid = umask.transpose(0, 1).bool()

    assert torch.allclose(original[valid], poisoned[valid], atol=1e-6, rtol=1e-6)
    assert torch.isfinite(poisoned[~valid]).all().item()
    assert torch.count_nonzero(poisoned[~valid]).item() == 0


def test_future_valid_utterance_changes_earlier_output_with_full_context():
    torch.manual_seed(3)
    model = _small_model()
    values, qmask, umask, lengths = _make_batch(lengths=(4,))
    changed_values = values.clone()
    changed_values[3, 0] = changed_values[3, 0] + 25.0

    original = model(values, qmask, umask, lengths)
    changed = model(changed_values, qmask, umask, lengths)

    assert not torch.allclose(original[0, 0], changed[0, 0], atol=1e-6, rtol=1e-6)


def test_changing_explicit_speaker_id_changes_output():
    torch.manual_seed(4)
    model = _small_model()
    values, qmask, umask, lengths = _make_batch(lengths=(4,))
    changed_qmask = qmask.clone()
    changed_qmask[0, 1] = 0.0

    original = model(values, qmask, umask, lengths)
    changed = model(values, changed_qmask, umask, lengths)

    assert not torch.allclose(original, changed, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "bad_shape",
    ["values_feature", "qmask_time", "qmask_rank", "umask_time", "lengths_batch"],
)
def test_forward_rejects_inconsistent_shapes(bad_shape):
    model = _small_model()
    values, qmask, umask, lengths = _make_batch()

    if bad_shape == "values_feature":
        values = values[..., :-1]
    elif bad_shape == "qmask_time":
        qmask = qmask[:, :-1]
    elif bad_shape == "qmask_rank":
        qmask = qmask.unsqueeze(-1)
    elif bad_shape == "umask_time":
        umask = umask[:, :-1]
    elif bad_shape == "lengths_batch":
        lengths = lengths[:-1]

    with pytest.raises(ValueError):
        model(values, qmask, umask, lengths)


@pytest.mark.parametrize("failure", ["non_binary", "not_prefix", "length_mismatch"])
def test_forward_rejects_invalid_umask_or_lengths(failure):
    model = _small_model()
    values, qmask, umask, lengths = _make_batch()

    if failure == "non_binary":
        umask[0, 1] = 0.5
    elif failure == "not_prefix":
        umask[0, 1] = 0.0
    elif failure == "length_mismatch":
        lengths[0] -= 1

    with pytest.raises(ValueError):
        model(values, qmask, umask, lengths)


@pytest.mark.parametrize("invalid_id", [-1.0, 2.0, 0.5, float("nan")])
def test_forward_rejects_invalid_speaker_id_on_valid_utterances(invalid_id):
    model = _small_model()
    values, qmask, umask, lengths = _make_batch()
    qmask[0, 0] = invalid_id

    with pytest.raises(ValueError):
        model(values, qmask, umask, lengths)


@pytest.mark.parametrize("padding_id", [float("nan"), float("inf"), 0.5, 123.0])
def test_padding_speaker_ids_are_ignored_and_mapped_to_padding_index(padding_id):
    model = _small_model()
    values, qmask, umask, lengths = _make_batch()
    baseline = model(values, qmask, umask, lengths)
    changed_qmask = qmask.clone()
    changed_qmask[1, 2:] = padding_id
    embedded_ids = []
    handle = model.speaker_embedding.register_forward_pre_hook(
        lambda _module, inputs: embedded_ids.append(inputs[0].detach().clone())
    )

    try:
        changed = model(values, changed_qmask, umask, lengths)
    finally:
        handle.remove()

    valid = umask.transpose(0, 1).bool()
    assert torch.allclose(baseline[valid], changed[valid], atol=1e-6, rtol=1e-6)
    assert len(embedded_ids) == 1
    assert embedded_ids[0].shape == (2, 4)
    assert torch.equal(embedded_ids[0][0], qmask[0].long())
    assert torch.equal(embedded_ids[0][1, :2], qmask[1, :2].long())
    assert torch.all(embedded_ids[0][1, 2:] == model.n_speakers).item()
    assert torch.count_nonzero(changed[2:, 1]).item() == 0


def test_fast_validation_path_matches_strict_and_skips_semantic_validator(monkeypatch):
    torch.manual_seed(41)
    strict_model = _small_model(validate_inputs=True)
    fast_model = _small_model(validate_inputs=False)
    fast_model.load_state_dict(strict_model.state_dict())
    values, qmask, umask, lengths = _make_batch()
    strict_output = strict_model(values, qmask, umask, lengths)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fast path entered semantic input validation")

    monkeypatch.setattr(fast_model, "_validate_input_semantics", fail_if_called)
    fast_output = fast_model(values, qmask, umask, lengths)

    assert strict_model.validate_inputs is True
    assert fast_model.validate_inputs is False
    assert torch.equal(strict_output, fast_output)


def test_fast_validation_path_does_not_materialize_sequence_lengths(monkeypatch):
    model = _small_model(validate_inputs=False)
    values, qmask, umask, lengths = _make_batch()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fast path called torch.as_tensor for seq_lengths")

    monkeypatch.setattr(torch, "as_tensor", fail_if_called)
    output = model(values, qmask, umask, lengths)

    assert output.shape == (4, 2, 32)


def test_forward_rejects_sequences_longer_than_configured_capacity():
    model = _small_model(max_len=3)
    values, qmask, umask, lengths = _make_batch(lengths=(4,))

    with pytest.raises(ValueError):
        model(values, qmask, umask, lengths)


def test_encoder_layers_are_distinct_and_independently_initialized():
    torch.manual_seed(5)
    model = _small_model(num_layers=5)

    assert len(model.layers) == 5
    assert len({id(layer) for layer in model.layers}) == 5
    assert not torch.equal(model.layers[0].linear1.weight, model.layers[1].linear1.weight)
    for layer in model.layers:
        assert isinstance(layer, PreNormTransformerLayer)
        assert layer.norm_first is True


def test_custom_encoder_layer_executes_pre_norm_module_order():
    torch.manual_seed(51)
    layer = PreNormTransformerLayer(
        d_model=16,
        num_heads=4,
        ff_dim=24,
        dropout=0.0,
    )
    events = []
    modules = [
        ("norm1", layer.norm1),
        ("self_attn", layer.self_attn),
        ("norm2", layer.norm2),
        ("linear1", layer.linear1),
        ("activation", layer.activation),
        ("linear2", layer.linear2),
    ]
    handles = [
        module.register_forward_pre_hook(
            lambda _module, _inputs, name=name: events.append(name)
        )
        for name, module in modules
    ]

    try:
        output = layer(
            torch.randn(4, 2, 16, dtype=torch.float32),
            src_key_padding_mask=torch.tensor(
                [[False, False, False, False], [False, False, True, True]]
            ),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == (4, 2, 16)
    assert events == ["norm1", "self_attn", "norm2", "linear1", "activation", "linear2"]


@pytest.mark.parametrize(
    "dimension",
    [
        "input_dim",
        "output_dim",
        "n_speakers",
        "d_model",
        "num_heads",
        "num_layers",
        "ff_dim",
        "max_len",
    ],
)
def test_boolean_dimensions_are_rejected(dimension):
    with pytest.raises(ValueError):
        _small_model(**{dimension: True})


@pytest.mark.parametrize("dropout", [False, True])
def test_boolean_dropout_is_rejected(dropout):
    with pytest.raises(ValueError):
        _small_model(dropout=dropout)


def test_backward_reaches_all_trainable_backbone_components():
    torch.manual_seed(6)
    model = _small_model(num_layers=5)
    model.train()
    values, qmask, umask, lengths = _make_batch(lengths=(4, 3))

    output = model(values, qmask, umask, lengths)
    valid = umask.transpose(0, 1).bool()
    loss = output[valid].square().mean()
    loss.backward()

    modules = [
        model.input_projection,
        *model.layers,
        model.speaker_embedding,
        model.final_norm,
        model.output_projection,
    ]
    for module in modules:
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all().item() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0.0

    embedding_gradient = model.speaker_embedding.weight.grad
    active_gradient = embedding_gradient[: model.n_speakers]
    padding_gradient = embedding_gradient[model.speaker_embedding.padding_idx]
    assert torch.isfinite(active_gradient).all().item()
    assert torch.all(active_gradient.abs().sum(dim=1) > 0).item()
    assert torch.count_nonzero(padding_gradient).item() == 0


def test_default_registered_and_effective_parameter_counts_match_contract():
    model = SDTStyleConversationBackbone()
    registered_parameters = sum(parameter.numel() for parameter in model.parameters())
    effective_parameters = registered_parameters - model.speaker_embedding.embedding_dim
    control_parameters = 5_864_700

    assert registered_parameters == 5_869_754
    assert effective_parameters == 5_869_370
    assert abs(effective_parameters - control_parameters) / control_parameters < 0.002
    assert model.speaker_embedding.num_embeddings == 2
    assert model.speaker_embedding.padding_idx == 1
