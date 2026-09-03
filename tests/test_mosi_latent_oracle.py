from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import replace

import pytest
import torch

from gcnet_missing_m3.oracle_diagnostic import (
    OracleState,
    build_sample_keys,
    compute_path_output,
    concatenate_oracle_states,
    effective_rank,
    extract_oracle_batch,
    flatten_valid_lbd,
    restore_named_buffers,
    shuffle_targets_by_modality,
    snapshot_named_buffers,
    stable_seed,
    stack_teacher_targets,
    state_dict_sha256,
    tensor_sha256,
)
from gcnet_missing_m3.model import (
    MODALITIES,
    MissingLatentResidualFusion,
    MissingM3GraphModel,
)


ASSERT_CLOSE = getattr(torch.testing, "assert_close", torch.testing.assert_allclose)


def _row_multiset(value):
    return sorted(tuple(row.tolist()) for row in value)


def _historical_effective_rank(value):
    """Reference from the 2026-08-31 MOSI latent checkpoint audit."""
    centered = value.float() - value.float().mean(dim=0, keepdim=True)
    if hasattr(torch.linalg, "svdvals"):
        singular_values = torch.linalg.svdvals(centered)
    else:
        singular_values = torch.svd(centered, some=False).S
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum()
    return float(entropy.exp().item())


def _tiny_completion_model():
    torch.manual_seed(101)
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
        n_classes=2,
        dropout=0.0,
        time_attn=False,
        no_cuda=True,
        latent_dim=4,
        num_experts=2,
        top_k=1,
        projector_dropout=0.0,
        predictor_dropout=0.0,
        fusion_type="slot",
        classification_completion=True,
    ).eval()
    with torch.no_grad():
        for projection in model.missing_latent_fusion.target_projections:
            torch.nn.init.normal_(projection[-1].weight, std=0.2)
            torch.nn.init.normal_(projection[-1].bias, std=0.2)
    return model


def _oracle_view(all_observed=False):
    torch.manual_seed(103)
    complete = torch.randn(3, 2, 9)
    availability = torch.tensor(
        [
            [[1, 0, 1], [0, 1, 1]],
            [[1, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    umask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    if all_observed:
        availability[umask.T] = 1
    expanded = torch.repeat_interleave(
        availability, torch.tensor([2, 3, 4]), dim=-1
    )
    return {
        "complete": complete,
        "incomplete": complete * expanded,
        "availability": availability,
        "qmask": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "umask": umask,
        "labels": torch.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 99.0]]),
        "lengths": [3, 2],
        "conversation_ids": ["vid-a", "vid-b"],
    }


def _oracle_state(prefix, values):
    count = len(values)
    value = torch.tensor(values, dtype=torch.float32)
    return OracleState(
        sample_keys=tuple(f"{prefix}:{index}" for index in range(count)),
        labels=value,
        availability=value[:, None].repeat(1, 3),
        graph_hidden=value[:, None].repeat(1, 2),
        predicted_latents=value[:, None, None].repeat(1, 3, 2),
        teacher_latents=(value + 10)[:, None, None].repeat(1, 3, 2),
        target_mask=torch.zeros(count, 3, dtype=torch.bool),
        native_logits=value[:, None].repeat(1, 2),
    )


def test_flatten_and_sample_keys_use_conversation_major_metric_order():
    value = torch.arange(3 * 2 * 2).reshape(3, 2, 2)
    umask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)

    flattened = flatten_valid_lbd(value, umask)
    keys = build_sample_keys(["vid-a", "vid-b"], umask)

    expected = torch.stack([value[0, 0], value[1, 0], value[0, 1], value[2, 1]])
    assert torch.equal(flattened, expected)
    assert keys == ("vid-a:0", "vid-a:1", "vid-b:0", "vid-b:2")


def test_stack_teacher_targets_uses_canonical_modality_order():
    mapping = OrderedDict(
        [
            ("visual", torch.full((2, 2), 3.0)),
            ("audio", torch.full((2, 2), 1.0)),
            ("text", torch.full((2, 2), 2.0)),
        ]
    )

    stacked = stack_teacher_targets(mapping)

    assert stacked.shape == (2, 3, 2)
    for index, expected in enumerate((1.0, 2.0, 3.0)):
        assert torch.equal(stacked[:, index], torch.full((2, 2), expected))


def test_stable_seed_is_the_sha256_derived_value():
    payload = b"66:0.4:3:audio"
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    assert stable_seed(66, 0.4, 3, "audio") == expected
    assert stable_seed(66, 0.4, 3, "audio") != stable_seed(
        66, 0.4, 3, "text"
    )


def test_shuffle_is_deterministic_target_specific_derangement_and_preserves_pools():
    sample_count = 10
    teacher = torch.arange(sample_count * 3 * 2, dtype=torch.float32).reshape(
        sample_count, 3, 2
    )
    target_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=torch.bool,
    )

    shuffled_a, metadata_a = shuffle_targets_by_modality(
        teacher, target_mask, master_seed=66, rate=0.4, shuffle_index=2
    )
    shuffled_b, metadata_b = shuffle_targets_by_modality(
        teacher, target_mask, master_seed=66, rate=0.4, shuffle_index=2
    )
    shuffled_c, _ = shuffle_targets_by_modality(
        teacher, target_mask, master_seed=66, rate=0.4, shuffle_index=3
    )

    assert torch.equal(shuffled_a, shuffled_b)
    assert metadata_a == metadata_b
    assert not torch.equal(shuffled_a, shuffled_c)
    assert torch.equal(shuffled_a[~target_mask], teacher[~target_mask])
    permutation_hashes = []
    for target_index, name in enumerate(MODALITIES):
        selected = target_mask[:, target_index]
        assert _row_multiset(shuffled_a[selected, target_index]) == _row_multiset(
            teacher[selected, target_index]
        )
        assert not torch.any(
            torch.all(
                shuffled_a[selected, target_index]
                == teacher[selected, target_index],
                dim=-1,
            )
        )
        target_metadata = metadata_a["modalities"][name]
        assert target_metadata["count"] == int(selected.sum())
        assert target_metadata["fixed_points"] == 0
        assert target_metadata["unshufflable"] is False
        assert len(target_metadata["permutation_sha256"]) == 64
        permutation_hashes.append(target_metadata["permutation_sha256"])
    assert len(set(permutation_hashes)) == len(MODALITIES)


def test_shuffle_handles_empty_and_singleton_target_pools():
    teacher = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3, 1)
    target_mask = torch.tensor(
        [[0, 0, 1], [0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.bool
    )

    shuffled, metadata = shuffle_targets_by_modality(
        teacher, target_mask, master_seed=7, rate=0.7, shuffle_index=0
    )

    assert torch.equal(shuffled[:, 0], teacher[:, 0])
    assert torch.equal(shuffled[:, 1], teacher[:, 1])
    assert metadata["modalities"]["audio"]["count"] == 0
    assert metadata["modalities"]["audio"]["fixed_points"] == 0
    assert metadata["modalities"]["text"]["count"] == 1
    assert metadata["modalities"]["text"]["fixed_points"] == 1
    assert metadata["modalities"]["text"]["unshufflable"] is True
    assert metadata["modalities"]["visual"]["fixed_points"] == 0
    assert not torch.equal(
        shuffled[target_mask[:, 2], 2], teacher[target_mask[:, 2], 2]
    )


def test_one_target_pool_change_does_not_perturb_other_modality_permutations():
    teacher = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3, 1)
    first_mask = torch.ones(8, 3, dtype=torch.bool)
    second_mask = first_mask.clone()
    second_mask[-1, 2] = False

    first, first_metadata = shuffle_targets_by_modality(
        teacher, first_mask, master_seed=9, rate=0.5, shuffle_index=4
    )
    second, second_metadata = shuffle_targets_by_modality(
        teacher, second_mask, master_seed=9, rate=0.5, shuffle_index=4
    )

    assert torch.equal(first[:, :2], second[:, :2])
    for name in ("audio", "text"):
        assert (
            first_metadata["modalities"][name]["permutation_sha256"]
            == second_metadata["modalities"][name]["permutation_sha256"]
        )


def test_graph_only_path_bypasses_nonzero_fusion_bias():
    class BiasedFusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, target_latents, target_mask, umask):
            self.calls += 1
            return target_latents.new_full((target_latents.shape[0], 1, 3), 7.0)

    graph_hidden = torch.randn(5, 3)
    target_mask = torch.ones(5, 3, dtype=torch.bool)
    fusion = BiasedFusion()
    classifier = torch.nn.Linear(3, 2)

    logits, residual = compute_path_output(
        graph_hidden, None, target_mask, fusion, classifier
    )

    assert fusion.calls == 0
    assert torch.count_nonzero(residual) == 0
    ASSERT_CLOSE(logits, classifier(graph_hidden), rtol=0, atol=0)


def test_target_path_reuses_sequence_first_fusion_interface():
    class ShapeRecordingFusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = None

        def forward(self, target_latents, target_mask, umask):
            self.shapes = (
                tuple(target_latents.shape),
                tuple(target_mask.shape),
                tuple(umask.shape),
            )
            weights = target_mask.to(target_latents.dtype).unsqueeze(-1)
            return (target_latents * weights).sum(dim=2)

    graph_hidden = torch.randn(4, 2)
    target_latents = torch.randn(4, 3, 2)
    target_mask = torch.tensor(
        [[1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=torch.bool
    )
    fusion = ShapeRecordingFusion()

    logits, residual = compute_path_output(
        graph_hidden, target_latents, target_mask, fusion, torch.nn.Identity()
    )

    assert fusion.shapes == ((4, 1, 3, 2), (4, 1, 3), (1, 4))
    expected_residual = (
        target_latents * target_mask.to(target_latents.dtype).unsqueeze(-1)
    ).sum(dim=1)
    ASSERT_CLOSE(residual, expected_residual)
    ASSERT_CLOSE(logits, graph_hidden + expected_residual)


def test_values_outside_target_mask_are_inert_for_original_fusion():
    torch.manual_seed(17)
    fusion = MissingLatentResidualFusion(latent_dim=2, hidden_dim=4)
    for projection in fusion.target_projections:
        torch.nn.init.normal_(projection[-1].weight)
        torch.nn.init.normal_(projection[-1].bias)
    classifier = torch.nn.Linear(4, 2)
    graph_hidden = torch.randn(5, 4)
    target_latents = torch.randn(5, 3, 2)
    target_mask = torch.tensor(
        [[1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 1]],
        dtype=torch.bool,
    )
    changed = target_latents.clone()
    changed[~target_mask] += 10_000.0

    first = compute_path_output(
        graph_hidden, target_latents, target_mask, fusion, classifier
    )
    second = compute_path_output(graph_hidden, changed, target_mask, fusion, classifier)

    ASSERT_CLOSE(first[0], second[0], rtol=0, atol=0)
    ASSERT_CLOSE(first[1], second[1], rtol=0, atol=0)


def test_tensor_and_state_dict_hashes_are_canonical_and_sensitive_to_metadata():
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    padded = torch.empty(3, 8, dtype=torch.float32)
    padded[:, ::2] = value
    same_noncontiguous = padded[:, ::2]

    assert tensor_sha256(value) == tensor_sha256(same_noncontiguous)
    assert tensor_sha256(value) != tensor_sha256(value.to(torch.float64))

    left = OrderedDict(
        [("z", torch.tensor([3.0])), ("a", torch.tensor([1.0, 2.0]))]
    )
    right = OrderedDict(reversed(list(left.items())))
    assert state_dict_sha256(left) == state_dict_sha256(right)
    assert state_dict_sha256({"x": torch.tensor([1, 2], dtype=torch.int16)}) != (
        state_dict_sha256({"x": torch.tensor([[1, 2]], dtype=torch.int16)})
    )
    module = torch.nn.Linear(2, 1)
    assert state_dict_sha256(module) == state_dict_sha256(module.state_dict())


def test_tensor_and_state_dict_hashes_match_golden_values_on_available_devices():
    value = torch.tensor([[1.0, -2.5], [0.0, 3.25]], dtype=torch.float32)
    state = OrderedDict(
        [
            (
                "z.weight",
                torch.tensor([[1, -2], [3, 4]], dtype=torch.int16),
            ),
            ("a.bias", torch.tensor([0.5, -0.25], dtype=torch.float32)),
        ]
    )
    tensor_digest = "df4e390c73480b1638be2be7326538459b87e0371689e7f997abd16a529af8ca"
    state_digest = "03b3d46e1ae34015435847c82c7a2932f1ad9462a222403ad15fa0c807f4ec1c"

    assert tensor_sha256(value) == tensor_digest
    assert state_dict_sha256(state) == state_digest
    if torch.cuda.is_available():
        assert tensor_sha256(value.cuda()) == tensor_digest
        assert state_dict_sha256(
            OrderedDict((name, tensor.cuda()) for name, tensor in state.items())
        ) == state_digest


def test_named_buffer_snapshot_restores_persistent_and_nonpersistent_buffers():
    module = torch.nn.Module()
    module.register_buffer("persistent", torch.tensor([1.0, 2.0]))
    module.child = torch.nn.Module()
    module.child.register_buffer(
        "routing_count", torch.tensor([3.0, 4.0]), persistent=False
    )
    snapshot = snapshot_named_buffers(module)

    module.persistent.add_(100.0)
    module.child.routing_count.zero_()
    restore_named_buffers(module, snapshot)

    assert set(snapshot) == {"persistent", "child.routing_count"}
    assert torch.equal(module.persistent, torch.tensor([1.0, 2.0]))
    assert torch.equal(module.child.routing_count, torch.tensor([3.0, 4.0]))


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_buffer_restore_prevalidates_every_buffer_before_copying(mismatch):
    module = torch.nn.Module()
    module.register_buffer("first", torch.tensor([1.0, 2.0]))
    module.register_buffer("late", torch.tensor([3.0, 4.0]))
    snapshot = snapshot_named_buffers(module)
    if mismatch == "shape":
        snapshot["late"] = snapshot["late"].reshape(2, 1)
    else:
        snapshot["late"] = snapshot["late"].to(torch.float64)
    module.first.fill_(10.0)
    module.late.fill_(20.0)
    before_failure = snapshot_named_buffers(module)

    with pytest.raises(ValueError, match="metadata"):
        restore_named_buffers(module, snapshot)

    for name, buffer in module.named_buffers():
        assert torch.equal(buffer, before_failure[name])


def test_effective_rank_is_finite_for_regular_and_small_inputs():
    assert effective_rank(torch.eye(3)) == pytest.approx(2.0, abs=1e-5)
    assert effective_rank(torch.ones(5, 3)) == pytest.approx(1.0)
    assert effective_rank(torch.randn(1, 4)) == pytest.approx(1.0)
    assert effective_rank(torch.empty(0, 4)) == pytest.approx(0.0)
    assert effective_rank(torch.zeros(4, 3)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="finite"):
        effective_rank(torch.tensor([[float("nan"), 0.0]]))


def test_effective_rank_is_translation_invariant_and_matches_historical_reference():
    value = torch.tensor(
        [
            [0.0, 1.0, 4.0],
            [1.0, 3.0, 2.0],
            [2.0, -1.0, 0.0],
            [4.0, 2.0, 5.0],
        ]
    )
    translated = value + torch.tensor([100.0, -37.0, 12.5])
    historical = _historical_effective_rank(value)

    assert effective_rank(value) == pytest.approx(historical, abs=1e-6)
    assert effective_rank(translated) == pytest.approx(historical, abs=1e-6)
    if torch.cuda.is_available():
        assert effective_rank(value.cuda()) == pytest.approx(historical, abs=1e-6)


def test_oracle_state_concatenation_preserves_order_and_rejects_duplicate_keys():
    left = _oracle_state("left", [1.0, 2.0])
    right = _oracle_state("right", [3.0, 4.0])

    combined = concatenate_oracle_states([left, right])

    assert combined.sample_keys == (
        "left:0",
        "left:1",
        "right:0",
        "right:1",
    )
    for name in (
        "labels",
        "availability",
        "graph_hidden",
        "predicted_latents",
        "teacher_latents",
        "target_mask",
        "native_logits",
    ):
        assert torch.equal(
            getattr(combined, name),
            torch.cat([getattr(left, name), getattr(right, name)], dim=0),
        )
    with pytest.raises(ValueError, match="duplicate sample key"):
        replace(left, sample_keys=("left:0", "left:0"))
    with pytest.raises(ValueError, match="duplicate sample key"):
        concatenate_oracle_states([left, _oracle_state("left", [9.0])])


def test_extract_oracle_batch_matches_native_completion_and_preserves_buffers():
    model = _tiny_completion_model()
    view = _oracle_view()
    buffers_before = snapshot_named_buffers(model)

    state, audit = extract_oracle_batch(model, view)

    assert state.sample_keys == (
        "vid-a:0",
        "vid-a:1",
        "vid-a:2",
        "vid-b:0",
        "vid-b:1",
    )
    assert state.labels.shape == (5,)
    assert state.availability.shape == (5, 3)
    assert state.graph_hidden.shape == (5, 10)
    assert state.predicted_latents.shape == (5, 3, 4)
    assert state.teacher_latents.shape == (5, 3, 4)
    assert state.target_mask.shape == (5, 3)
    assert state.native_logits.shape == (5, 2)
    assert audit["predicted_hidden_max_abs_error"] < 1e-6
    assert audit["predicted_logits_max_abs_error"] < 1e-6
    manual_logits, _ = compute_path_output(
        state.graph_hidden,
        state.predicted_latents,
        state.target_mask,
        model.missing_latent_fusion,
        model.smax_fc,
    )
    ASSERT_CLOSE(manual_logits, state.native_logits, rtol=0, atol=1e-6)
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, buffers_before[name])


def test_extract_oracle_batch_rejects_local_context_before_computation(monkeypatch):
    model = _tiny_completion_model()
    model.local_context_residual = True

    def fail_if_computed(*_args, **_kwargs):
        raise AssertionError("local-context models must be rejected before computation")

    monkeypatch.setattr(model.observed_set, "forward", fail_if_computed)

    with pytest.raises(ValueError, match="local_context_residual"):
        extract_oracle_batch(model, _oracle_view())


def test_extract_oracle_batch_complete_targets_only_change_teacher_latents():
    model = _tiny_completion_model()
    view = _oracle_view()
    changed_view = dict(view)
    changed_complete = view["complete"].clone()
    valid = view["umask"].T.bool()
    start = 0
    for target_index, width in enumerate(model.dimensions):
        block = changed_complete[..., start : start + width]
        selected = valid & ~view["availability"][..., target_index].bool()
        delta = torch.arange(1, width + 1, dtype=block.dtype) * 100.0
        block[selected] = block[selected] + delta
        start += width
    changed_view["complete"] = changed_complete

    original, _ = extract_oracle_batch(model, view)
    changed, _ = extract_oracle_batch(model, changed_view)

    assert torch.equal(original.graph_hidden, changed.graph_hidden)
    assert torch.equal(original.predicted_latents, changed.predicted_latents)
    assert torch.equal(original.target_mask, changed.target_mask)
    assert torch.equal(original.native_logits, changed.native_logits)
    assert not torch.equal(
        original.teacher_latents[original.target_mask],
        changed.teacher_latents[changed.target_mask],
    )
    assert torch.equal(
        original.teacher_latents[~original.target_mask],
        changed.teacher_latents[~changed.target_mask],
    )


def test_extract_oracle_batch_rate_zero_has_no_targets_and_equal_paths():
    model = _tiny_completion_model()
    state, audit = extract_oracle_batch(model, _oracle_view(all_observed=True))

    assert int(state.target_mask.sum().item()) == 0
    graph_logits, _ = compute_path_output(
        state.graph_hidden,
        None,
        state.target_mask,
        model.missing_latent_fusion,
        model.smax_fc,
    )
    predicted_logits, _ = compute_path_output(
        state.graph_hidden,
        state.predicted_latents,
        state.target_mask,
        model.missing_latent_fusion,
        model.smax_fc,
    )
    teacher_logits, _ = compute_path_output(
        state.graph_hidden,
        state.teacher_latents,
        state.target_mask,
        model.missing_latent_fusion,
        model.smax_fc,
    )
    shuffled_teacher, _ = shuffle_targets_by_modality(
        state.teacher_latents,
        state.target_mask,
        master_seed=66,
        rate=0.0,
        shuffle_index=0,
    )
    shuffled_logits, _ = compute_path_output(
        state.graph_hidden,
        shuffled_teacher,
        state.target_mask,
        model.missing_latent_fusion,
        model.smax_fc,
    )
    assert torch.equal(shuffled_teacher, state.teacher_latents)
    for logits in (
        graph_logits,
        predicted_logits,
        teacher_logits,
        shuffled_logits,
    ):
        ASSERT_CLOSE(logits, state.native_logits, rtol=0, atol=1e-6)
    assert audit["predicted_hidden_max_abs_error"] < 1e-6
    assert audit["predicted_logits_max_abs_error"] < 1e-6
