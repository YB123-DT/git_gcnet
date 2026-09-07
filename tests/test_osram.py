import torch

from gcnet_missing_m3.osram import MODALITIES, OSRAMBackbone
from gcnet_missing_m3.model import ContextualM3Predictor, MissingM3GraphModel


def _inputs(latent_dim=8):
    torch.manual_seed(7)
    e = torch.randn(4, 2, latent_dim)
    latents = {
        name: torch.randn(4, 2, latent_dim)
        for name in ("audio", "text", "visual")
    }
    availability = torch.tensor(
        [
            [[1, 0, 1], [0, 1, 1]],
            [[1, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [1, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    qmask = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]])
    umask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.float32)
    return e, latents, availability, qmask, umask, [3, 3]


def test_osram_returns_fixed_slots_and_hard_masks_missing_context():
    model = OSRAMBackbone(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=4,
        n_speakers=2,
        dropout=0.0,
    ).eval()
    e, latents, availability, qmask, umask, lengths = _inputs()

    hidden, contexts = model(e, latents, availability, qmask, umask, lengths)

    assert hidden.shape == (4, 2, 10)
    assert contexts["base"].shape == (4, 2, 16)
    assert contexts["gap"].shape == (4, 2, 3, 16)
    assert torch.count_nonzero(contexts["gap"][2, 0]) == 0
    assert torch.count_nonzero(contexts["gap"][0, 0, 0]) == 0
    assert torch.count_nonzero(contexts["gap"][0, 0, 1]) > 0
    assert torch.count_nonzero(contexts["gap"][0, 0, 2]) == 0
    assert torch.count_nonzero(hidden[3]) == 0
    assert torch.isfinite(hidden).all()


def test_osram_query_availability_switch_only_changes_query_conditioning():
    explicit = OSRAMBackbone(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=4,
        n_speakers=2,
        dropout=0.0,
        query_use_availability=True,
    ).eval()
    without = OSRAMBackbone(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=4,
        n_speakers=2,
        dropout=0.0,
        query_use_availability=False,
    ).eval()
    without.load_state_dict(explicit.state_dict())
    e, latents, availability, qmask, _umask, _lengths = _inputs()

    explicit_keys, explicit_values, explicit_queries = explicit._project_sequence(
        e, latents, availability, qmask
    )
    without_keys, without_values, without_queries = without._project_sequence(
        e, latents, availability, qmask
    )
    for name in MODALITIES:
        torch.testing.assert_close(explicit_keys[name], without_keys[name])
        torch.testing.assert_close(explicit_values[name], without_values[name])
    assert not torch.equal(explicit_queries, without_queries)

    changed_availability = availability.clone()
    changed_availability[0, 0] = torch.tensor([0.0, 1.0, 1.0])
    _, _, explicit_changed_queries = explicit._project_sequence(
        e, latents, changed_availability, qmask
    )
    _, _, without_changed_queries = without._project_sequence(
        e, latents, changed_availability, qmask
    )
    assert not torch.equal(explicit_queries[0, 0], explicit_changed_queries[0, 0])
    torch.testing.assert_close(
        without_queries[0, 0], without_changed_queries[0, 0]
    )


def test_osram_ablation_modes_isolate_context_slots():
    full = OSRAMBackbone(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=4,
        n_speakers=2,
        dropout=0.0,
        osram_ablation="full",
    ).eval()
    local_only = OSRAMBackbone(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=4,
        n_speakers=2,
        dropout=0.0,
        osram_ablation="local-only",
    ).eval()
    local_base = OSRAMBackbone(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=4,
        n_speakers=2,
        dropout=0.0,
        osram_ablation="local-base",
    ).eval()
    torch.manual_seed(17)
    with torch.no_grad():
        full.emotion_adapter[-1].weight.normal_(0.0, 0.1)
        full.emotion_adapter[-1].bias.normal_(0.0, 0.1)
    local_only.load_state_dict(full.state_dict())
    local_base.load_state_dict(full.state_dict())
    inputs = _inputs()

    full_hidden, full_contexts = full(*inputs)
    local_hidden, local_contexts = local_only(*inputs)
    base_hidden, base_contexts = local_base(*inputs)

    assert torch.count_nonzero(full_contexts["base"][1:3]) > 0
    assert torch.count_nonzero(full_contexts["gap"][1:3]) > 0
    assert torch.count_nonzero(local_contexts["base"]) == 0
    assert torch.count_nonzero(local_contexts["gap"]) == 0
    assert torch.count_nonzero(base_contexts["gap"]) == 0
    torch.testing.assert_close(base_contexts["base"], full_contexts["base"])
    assert not torch.equal(local_hidden, full_hidden)
    assert not torch.equal(base_hidden, full_hidden)


def test_osram_block_write_is_invariant_to_modality_column_permutation():
    model = OSRAMBackbone(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=4,
        n_speakers=2,
        dropout=0.0,
    )
    torch.manual_seed(11)
    memory = torch.randn(2, 2, 4, 3)
    keys = torch.randn(2, 2, 3, 3)
    values = torch.randn(2, 2, 4, 3)
    availability = torch.tensor(
        [[1, 0, 1], [1, 1, 0]], dtype=torch.float32
    )
    beta = torch.sigmoid(model.beta_logits).detach()

    first = model.block_write(memory, keys, values, availability, beta)
    order = torch.tensor([2, 0, 1])
    second = model.block_write(
        memory,
        keys.index_select(-1, order),
        values.index_select(-1, order),
        availability.index_select(-1, order),
        beta.index_select(-1, order),
    )

    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


def test_osram_block_write_absent_slots_have_finite_gradients():
    model = OSRAMBackbone(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=4,
        dropout=0.0,
    )
    memory = torch.zeros(2, 2, 4, 3, requires_grad=True)
    keys = torch.randn(2, 2, 3, 3, requires_grad=True)
    values = torch.randn(2, 2, 4, 3, requires_grad=True)
    availability = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )

    output = model.block_write(memory, keys, values, availability)
    output.square().mean().backward()

    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    for value in (memory.grad, keys.grad, values.grad):
        assert value is not None
        assert torch.isfinite(value).all()


def test_osram_read_before_write_does_not_expose_current_value():
    model = OSRAMBackbone(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=4,
        n_speakers=2,
        dropout=0.0,
    ).eval()
    e, latents, availability, qmask, umask, lengths = _inputs()
    changed = {name: value.clone() for name, value in latents.items()}
    changed["audio"][0, 0].add_(10000.0)

    one_step_umask = torch.ones(2, 1)
    first, first_contexts = model(
        e[:1],
        {name: value[:1] for name, value in latents.items()},
        availability[:1],
        qmask[:, :1],
        one_step_umask,
        [1, 1],
    )
    second, second_contexts = model(
        e[:1],
        {name: value[:1] for name, value in changed.items()},
        availability[:1],
        qmask[:, :1],
        one_step_umask,
        [1, 1],
    )

    torch.testing.assert_close(first[:1], second, rtol=0, atol=0)
    torch.testing.assert_close(
        first_contexts["base"][:1], second_contexts["base"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        first_contexts["gap"][:1], second_contexts["gap"], rtol=0, atol=0
    )


def test_osram_padding_and_conversation_reset_are_independent():
    model = OSRAMBackbone(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=4,
        n_speakers=2,
        dropout=0.0,
    ).eval()
    e, latents, availability, qmask, umask, lengths = _inputs()
    padded_e = e.clone()
    padded_latents = {name: value.clone() for name, value in latents.items()}
    padded_e[3].fill_(12345.0)
    for value in padded_latents.values():
        value[3].fill_(-54321.0)

    base, _ = model(e, latents, availability, qmask, umask, lengths)
    changed, _ = model(
        padded_e, padded_latents, availability, qmask, umask, lengths
    )
    torch.testing.assert_close(base[:3], changed[:3], rtol=0, atol=0)
    assert torch.count_nonzero(changed[3]) == 0


def test_structured_predictor_keeps_target_gap_slots_separate():
    predictor = ContextualM3Predictor(
        latent_dim=4,
        context_dim=4,
        num_experts=2,
        top_k=1,
        dropout=0.0,
        structured=True,
    )
    base = torch.randn(2, 1, 4)
    gap = torch.randn(2, 1, 3, 4)
    contexts = predictor.structured_contexts(base, gap)
    changed_gap = gap.clone()
    changed_gap[:, :, 1] += 1000.0
    changed = predictor.structured_contexts(base, changed_gap)

    assert contexts.shape == (2, 1, 3, 4)
    torch.testing.assert_close(contexts[:, :, 0], changed[:, :, 0])
    torch.testing.assert_close(contexts[:, :, 2], changed[:, :, 2])
    assert not torch.equal(contexts[:, :, 1], changed[:, :, 1])


def test_osram_missing_m3_graph_model_uses_structured_backbone():
    model = MissingM3GraphModel(
        base_model="LSTM",
        adim=2,
        tdim=3,
        vdim=4,
        D_e=4,
        graph_hidden_size=2,
        n_speakers=2,
        window_past=1,
        window_future=1,
        n_classes=6,
        dropout=0.0,
        no_cuda=True,
        latent_dim=8,
        num_experts=2,
        top_k=1,
        projector_dropout=0.0,
        predictor_dropout=0.0,
        backbone_type="osram",
        osram_output_dim=10,
        osram_num_heads=2,
        osram_key_dim=3,
        osram_value_dim=4,
    ).eval()
    _node, _latents, availability, qmask, umask, lengths = _inputs()
    features = torch.randn(4, 2, 9)

    logits, hidden, _, predictions = model(
        [features], availability, qmask, umask, lengths, predict_missing=True
    )

    assert model.backbone_type == "osram"
    assert logits.shape == (4, 2, 6)
    assert hidden.shape == (4, 2, 10)
    assert predictions is not None
    assert model.missing_predictor.structured is True
    assert model.osram.last_diagnostics["base_context_norm"] >= 0.0
