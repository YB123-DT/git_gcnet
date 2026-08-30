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
    qmask = torch.zeros(
        batch_size,
        total_length,
        n_speakers,
        dtype=torch.float32,
    )
    for batch_index, length in enumerate(lengths):
        umask[batch_index, :length] = 1.0
        for time_index in range(length):
            qmask[batch_index, time_index, time_index % n_speakers] = 1.0

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
    changed_qmask[0, 1] = torch.tensor([1.0, 0.0])

    original = model(values, qmask, umask, lengths)
    changed = model(values, changed_qmask, umask, lengths)

    assert not torch.allclose(original, changed, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "bad_shape",
    ["values_feature", "qmask_time", "qmask_speakers", "umask_time", "lengths_batch"],
)
def test_forward_rejects_inconsistent_shapes(bad_shape):
    model = _small_model()
    values, qmask, umask, lengths = _make_batch()

    if bad_shape == "values_feature":
        values = values[..., :-1]
    elif bad_shape == "qmask_time":
        qmask = qmask[:, :-1]
    elif bad_shape == "qmask_speakers":
        qmask = qmask[..., :-1]
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


@pytest.mark.parametrize("failure", ["missing_speaker", "multiple_speakers", "non_binary"])
def test_forward_rejects_invalid_qmask_on_valid_utterances(failure):
    model = _small_model()
    values, qmask, umask, lengths = _make_batch()

    if failure == "missing_speaker":
        qmask[0, 0] = 0.0
    elif failure == "multiple_speakers":
        qmask[0, 0] = 1.0
    elif failure == "non_binary":
        qmask[0, 0, 0] = 0.5

    with pytest.raises(ValueError):
        model(values, qmask, umask, lengths)


def test_forward_accepts_all_zero_qmask_on_padding():
    model = _small_model()
    values, qmask, umask, lengths = _make_batch()

    assert torch.count_nonzero(qmask[1, 2:]).item() == 0
    output = model(values, qmask, umask, lengths)

    assert torch.count_nonzero(output[2:, 1]).item() == 0


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


def test_backward_reaches_every_layer_and_input_output_projection():
    torch.manual_seed(6)
    model = _small_model(num_layers=5)
    model.train()
    values, qmask, umask, lengths = _make_batch(lengths=(4, 3))

    output = model(values, qmask, umask, lengths)
    valid = umask.transpose(0, 1).bool()
    loss = output[valid].square().mean()
    loss.backward()

    modules = [model.input_projection, *model.layers, model.output_projection]
    for module in modules:
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all().item() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0.0


def test_default_parameter_count_matches_sdt_backbone_contract():
    model = SDTStyleConversationBackbone()

    assert sum(parameter.numel() for parameter in model.parameters()) == 5_869_754
    assert model.speaker_embedding.num_embeddings == 2
    assert model.speaker_embedding.padding_idx == 1
