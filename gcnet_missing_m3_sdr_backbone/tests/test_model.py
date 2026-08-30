import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from gcnet_missing_m3_sdr_backbone.layers import TEMPORAL_RELATION_TABLE
from gcnet_missing_m3_sdr_backbone.model import (
    SDRConversationBackbone,
    SDRRelationBranch,
)


UPSTREAM_SDR = Path("/data2/yb/paper/SDR-GNN_reproduction_mosi_20260830")


def _make_batch(
    lengths=(4, 2, 0),
    input_dim=6,
    n_speakers=2,
    total_length=None,
    dtype=torch.float32,
):
    if total_length is None:
        total_length = max(max(lengths, default=0), 1)
    values = torch.randn(total_length, len(lengths), input_dim, dtype=dtype)
    qmask = torch.zeros(len(lengths), total_length, dtype=torch.long)
    umask = torch.zeros(len(lengths), total_length, dtype=dtype)
    for batch_index, length in enumerate(lengths):
        umask[batch_index, :length] = 1
        if length:
            qmask[batch_index, :length] = (
                torch.arange(length, dtype=torch.long) % n_speakers
            )
    return values, qmask, umask, torch.tensor(lengths, dtype=torch.long)


def _small_backbone(variant="sdr-public", **overrides):
    arguments = {
        "variant": variant,
        "input_dim": 6,
        "recurrent_hidden": 3,
        "graph_hidden": 5,
        "n_speakers": 2,
        "window_past": 2,
        "window_future": 2,
        "dropout": 0.0,
    }
    arguments.update(overrides)
    model = SDRConversationBackbone(**arguments)
    model.eval()
    return model


def _module_gradient_total(module):
    gradients = [parameter.grad for parameter in module.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all().item() for gradient in gradients)
    return sum(gradient.abs().sum().item() for gradient in gradients)


def test_public_forward_has_requested_shape_and_registers_only_temporal_branch():
    torch.manual_seed(1)
    model = _small_backbone("sdr-public")
    values, qmask, umask, lengths = _make_batch()

    output = model(values, qmask, umask, lengths)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (4, 3, 11)
    assert output.dtype == values.dtype
    assert output.device == values.device
    assert torch.count_nonzero(output[2:, 1]).item() == 0
    assert torch.count_nonzero(output[:, 2]).item() == 0
    assert hasattr(model, "temporal_branch")
    assert not hasattr(model, "speaker_branch")
    assert not hasattr(model, "fusion")
    state_keys = tuple(model.state_dict())
    assert not any("speaker" in key or "fusion" in key for key in state_keys)


def test_paper_forward_is_tensor_and_registers_two_branches_and_fusion():
    torch.manual_seed(2)
    model = _small_backbone("sdr-paper")
    values, qmask, umask, lengths = _make_batch(lengths=(4, 3))

    output = model(values, qmask, umask, lengths)

    assert isinstance(output, torch.Tensor)
    assert not isinstance(output, tuple)
    assert output.shape == (4, 2, 11)
    assert isinstance(model.temporal_branch, SDRRelationBranch)
    assert isinstance(model.speaker_branch, SDRRelationBranch)
    assert isinstance(model.fusion, torch.nn.Linear)
    assert model.fusion.in_features == 22
    assert model.fusion.out_features == 11


def test_default_backbone_matches_formal_missing_m3_shape_and_widths():
    torch.manual_seed(3)
    model = SDRConversationBackbone(variant="sdr-public", dropout=0.0).eval()
    values, qmask, umask, lengths = _make_batch(
        lengths=(7, 5, 3),
        input_dim=256,
        n_speakers=1,
        total_length=7,
    )

    with torch.no_grad():
        output = model(values, qmask, umask, lengths)

    assert model.pre_graph_bigru.input_size == 256
    assert model.pre_graph_bigru.hidden_size == 200
    assert model.pre_graph_bigru.num_layers == 2
    assert model.pre_graph_bigru.bidirectional is True
    assert model.temporal_branch.rgcn.in_channels == 400
    assert model.temporal_branch.rgcn.out_channels == 100
    assert model.temporal_branch.post_graph_bigru.input_size == 500
    assert model.temporal_branch.post_graph_bigru.hidden_size == 500
    assert model.temporal_branch.post_graph_bigru.num_layers == 2
    assert model.temporal_branch.post_graph_bigru.bidirectional is True
    assert model.temporal_branch.output_linear.in_features == 1000
    assert model.temporal_branch.output_linear.out_features == 500
    assert output.shape == (7, 3, 500)
    assert torch.count_nonzero(output[5:, 1]).item() == 0
    assert torch.count_nonzero(output[3:, 2]).item() == 0


def test_padding_features_and_padding_speakers_cannot_change_valid_outputs():
    torch.manual_seed(4)
    model = _small_backbone("sdr-paper")
    values, qmask, umask, lengths = _make_batch(lengths=(4, 2, 0))
    poisoned_values = values.clone()
    poisoned_values[2:, 1] = float("nan")
    poisoned_values[:, 2] = float("nan")
    poisoned_qmask = qmask.to(dtype=torch.float32)
    poisoned_qmask[1, 2:] = float("nan")
    poisoned_qmask[2] = float("inf")

    baseline = model(values, qmask, umask, lengths)
    poisoned = model(poisoned_values, poisoned_qmask, umask, lengths)
    valid = umask.transpose(0, 1).bool()

    assert torch.allclose(baseline[valid], poisoned[valid], atol=1e-6, rtol=1e-6)
    assert torch.isfinite(poisoned).all().item()
    assert torch.count_nonzero(poisoned[~valid]).item() == 0


@pytest.mark.parametrize("variant", ["sdr-public", "sdr-paper"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_forward_backward_preserves_floating_dtype(variant, dtype):
    torch.manual_seed(5)
    model = _small_backbone(variant).to(dtype=dtype)
    model.train()
    values, qmask, umask, lengths = _make_batch(
        lengths=(4, 3, 0),
        dtype=dtype,
    )
    values.requires_grad_(True)

    output = model(values, qmask, umask, lengths)
    loss = output.square().sum()
    loss.backward()

    assert output.dtype == dtype
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all().item()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all().item()
    assert _module_gradient_total(model.pre_graph_bigru) > 0.0


def test_all_zero_length_batch_returns_differentiable_strict_zeros():
    model = _small_backbone("sdr-paper")
    model.train()
    values, qmask, umask, lengths = _make_batch(lengths=(0, 0), total_length=3)
    values.requires_grad_(True)

    output = model(values, qmask, umask, lengths)

    assert output.shape == (3, 2, 11)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert values.grad is not None
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_paper_backward_reaches_every_relation_stage_and_fusion():
    torch.manual_seed(6)
    model = _small_backbone("sdr-paper")
    model.train()
    with torch.no_grad():
        model.temporal_branch.output_linear.bias.fill_(1.0)
        model.speaker_branch.output_linear.bias.fill_(1.0)
        model.fusion.weight.fill_(0.1)
        model.fusion.bias.fill_(1.0)
    values, qmask, umask, lengths = _make_batch(lengths=(4, 4))

    output = model(values, qmask, umask, lengths)
    output.sum().backward()

    for branch in (model.temporal_branch, model.speaker_branch):
        assert _module_gradient_total(branch.rgcn) > 0.0
        assert _module_gradient_total(branch.hypergraph) > 0.0
        assert _module_gradient_total(branch.high_conv) > 0.0
        assert _module_gradient_total(branch.post_graph_bigru) > 0.0
        assert _module_gradient_total(branch.output_linear) > 0.0
    assert _module_gradient_total(model.fusion) > 0.0


@pytest.mark.parametrize("dropout", [False, True])
def test_boolean_dropout_is_rejected_before_numeric_conversion(dropout):
    with pytest.raises(TypeError, match="real probability, not bool"):
        _small_backbone(dropout=dropout)


@pytest.mark.parametrize("dropout", [-0.01, 1.01, float("nan")])
def test_dropout_outside_closed_unit_interval_is_rejected(dropout):
    with pytest.raises(ValueError, match="between 0 and 1"):
        _small_backbone(dropout=dropout)


def test_non_numeric_dropout_is_rejected():
    with pytest.raises(TypeError, match="real probability"):
        _small_backbone(dropout="0.5")


@pytest.mark.parametrize("n_speakers", [False, True, 1.0, 2.0])
def test_backbone_requires_an_integer_speaker_count(n_speakers):
    with pytest.raises(ValueError, match="n_speakers"):
        _small_backbone(n_speakers=n_speakers)


@pytest.mark.parametrize("n_speakers", [False, True, 1.0, 2.0])
def test_relation_branch_requires_an_integer_speaker_count(n_speakers):
    with pytest.raises(ValueError, match="n_speakers"):
        SDRRelationBranch(
            recurrent_dim=6,
            graph_hidden=5,
            num_relations=3,
            relation="temporal",
            n_speakers=n_speakers,
            dropout=0.0,
        )


@pytest.mark.parametrize("variant", ["public", "paper", "SDR-public", ""])
def test_only_two_exact_variant_names_are_accepted(variant):
    with pytest.raises(ValueError, match="sdr-public.*sdr-paper"):
        _small_backbone(variant)


@pytest.mark.parametrize(
    "failure",
    ["values_rank", "values_width", "qmask_shape", "umask_shape", "lengths_shape"],
)
def test_forward_rejects_invalid_input_shapes(failure):
    model = _small_backbone()
    values, qmask, umask, lengths = _make_batch()
    if failure == "values_rank":
        values = values.unsqueeze(0)
    elif failure == "values_width":
        values = values[..., :-1]
    elif failure == "qmask_shape":
        qmask = qmask[:, :-1]
    elif failure == "umask_shape":
        umask = umask.unsqueeze(-1)
    elif failure == "lengths_shape":
        lengths = lengths.unsqueeze(0)

    with pytest.raises(ValueError):
        model(values, qmask, umask, lengths)


@pytest.mark.parametrize("failure", ["non_binary", "not_prefix", "mismatch"])
def test_umask_must_be_a_binary_prefix_equal_to_lengths(failure):
    model = _small_backbone()
    values, qmask, umask, lengths = _make_batch()
    if failure == "non_binary":
        umask[0, 1] = 0.5
    elif failure == "not_prefix":
        umask[0, 1] = 0
    elif failure == "mismatch":
        lengths[0] -= 1

    with pytest.raises(ValueError, match="umask"):
        model(values, qmask, umask, lengths)


@pytest.mark.parametrize("bad_lengths", [[4, -1, 0], [5, 2, 0], [4.5, 2, 0]])
def test_lengths_must_be_finite_integers_within_padded_length(bad_lengths):
    model = _small_backbone()
    values, qmask, umask, _ = _make_batch()

    with pytest.raises(ValueError, match="length"):
        model(values, qmask, umask, bad_lengths)


def test_complex_lengths_are_rejected_before_integer_conversion():
    model = _small_backbone()
    values, qmask, umask, _ = _make_batch()
    complex_lengths = torch.tensor([4 + 0j, 2 + 0j, 0 + 0j])

    with pytest.raises(ValueError, match="complex"):
        model(values, qmask, umask, complex_lengths)


@pytest.mark.parametrize("failure", ["non_binary", "not_prefix", "mismatch"])
def test_direct_relation_branch_validates_umask_against_lengths(failure):
    branch = SDRRelationBranch(
        recurrent_dim=6,
        graph_hidden=5,
        num_relations=3,
        relation="temporal",
        n_speakers=2,
        dropout=0.0,
    )
    recurrent = torch.randn(4, 2, 6)
    _, qmask, umask, lengths = _make_batch(lengths=(4, 2))
    if failure == "non_binary":
        umask[1, 2] = 0.5
    elif failure == "not_prefix":
        umask[1] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    elif failure == "mismatch":
        umask[1, 2] = 1.0

    with pytest.raises(ValueError, match="umask"):
        branch(recurrent, qmask, umask, lengths)


@pytest.mark.parametrize("invalid_id", [-1.0, 2.0, 0.5, float("nan")])
def test_valid_speaker_ids_must_be_finite_integers_in_range(invalid_id):
    model = _small_backbone()
    values, qmask, umask, lengths = _make_batch()
    qmask = qmask.to(dtype=torch.float32)
    qmask[0, 0] = invalid_id

    with pytest.raises(ValueError, match="speaker"):
        model(values, qmask, umask, lengths)


def test_boolean_valid_speaker_ids_are_rejected():
    model = _small_backbone(n_speakers=1)
    values, _, umask, lengths = _make_batch(n_speakers=1)
    qmask = torch.zeros(umask.shape, dtype=torch.bool)

    with pytest.raises(ValueError, match="speaker"):
        model(values, qmask, umask, lengths)


def test_all_padding_boolean_qmask_is_ignored():
    model = _small_backbone("sdr-paper", n_speakers=1)
    values, _, umask, lengths = _make_batch(
        lengths=(0, 0),
        n_speakers=1,
        total_length=3,
    )
    padding_qmask = torch.ones(umask.shape, dtype=torch.bool)

    output = model(values, padding_qmask, umask, lengths)

    assert torch.equal(output, torch.zeros_like(output))


def test_default_parameter_counts_are_recorded_for_both_variants():
    public = SDRConversationBackbone(variant="sdr-public")
    paper = SDRConversationBackbone(variant="sdr-paper")

    public_parameters = sum(parameter.numel() for parameter in public.parameters())
    paper_parameters = sum(parameter.numel() for parameter in paper.parameters())

    assert public_parameters == 9_444_901
    assert paper_parameters == 18_038_302
    assert all(parameter.requires_grad for parameter in public.parameters())
    assert all(parameter.requires_grad for parameter in paper.parameters())


def test_backbone_does_not_register_excluded_sdr_components():
    for variant in ("sdr-public", "sdr-paper"):
        model = _small_backbone(variant)
        names = tuple(name.lower() for name, _ in model.named_modules())
        assert not any("match" in name for name in names)
        assert not any("attention" in name or "attn" in name for name in names)
        assert not any("reconstruct" in name or "classifier" in name for name in names)


@contextmanager
def _load_upstream_sdr_modules():
    if not UPSTREAM_SDR.is_dir():
        pytest.skip("the locked upstream SDR repository is unavailable")

    graph_spec = importlib.util.spec_from_file_location(
        "_locked_sdr_graph",
        str(UPSTREAM_SDR / "graph.py"),
    )
    model_spec = importlib.util.spec_from_file_location(
        "_locked_sdr_model",
        str(UPSTREAM_SDR / "model_SDRGNN.py"),
    )
    if graph_spec is None or graph_spec.loader is None:
        pytest.skip("the locked upstream SDR graph module cannot be loaded")
    if model_spec is None or model_spec.loader is None:
        pytest.skip("the locked upstream SDR model module cannot be loaded")

    graph_module = importlib.util.module_from_spec(graph_spec)
    model_module = importlib.util.module_from_spec(model_spec)
    temporary_names = (
        graph_spec.name,
        model_spec.name,
        "graph",
        "torch_scatter",
    )
    missing = object()
    previous_modules = {
        name: sys.modules.get(name, missing) for name in temporary_names
    }
    try:
        sys.modules[graph_spec.name] = graph_module
        sys.modules[model_spec.name] = model_module
        if previous_modules["torch_scatter"] is missing:
            torch_scatter = types.ModuleType("torch_scatter")

            def scatter_add(source, index, dim=0, dim_size=None):
                if dim != 0:
                    raise ValueError("the parity shim only supports dim=0")
                if dim_size is None:
                    dim_size = int(index.max().item()) + 1 if index.numel() else 0
                shape = list(source.shape)
                shape[dim] = dim_size
                output = source.new_zeros(shape)
                return output.index_add(dim, index.long(), source)

            torch_scatter.scatter_add = scatter_add
            sys.modules["torch_scatter"] = torch_scatter
        graph_spec.loader.exec_module(graph_module)
        sys.modules["graph"] = graph_module
        model_spec.loader.exec_module(model_module)
        yield graph_module, model_module
    finally:
        for name in reversed(temporary_names):
            previous = previous_modules[name]
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_upstream_loader_does_not_leak_private_module_names(monkeypatch):
    private_names = ("_locked_sdr_graph", "_locked_sdr_model")
    for name in private_names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    with _load_upstream_sdr_modules() as (graph_module, model_module):
        assert sys.modules[private_names[0]] is graph_module
        assert sys.modules[private_names[1]] is model_module

    assert all(name not in sys.modules for name in private_names)


def test_upstream_loader_restores_existing_private_modules(monkeypatch):
    previous_graph = types.ModuleType("_previous_locked_sdr_graph")
    previous_model = types.ModuleType("_previous_locked_sdr_model")
    monkeypatch.setitem(sys.modules, "_locked_sdr_graph", previous_graph)
    monkeypatch.setitem(sys.modules, "_locked_sdr_model", previous_model)

    with _load_upstream_sdr_modules() as (graph_module, model_module):
        assert graph_module is not previous_graph
        assert model_module is not previous_model

    assert sys.modules["_locked_sdr_graph"] is previous_graph
    assert sys.modules["_locked_sdr_model"] is previous_model


def test_upstream_loader_cleans_private_modules_when_import_fails(monkeypatch):
    private_names = ("_locked_sdr_graph", "_locked_sdr_model")
    for name in private_names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    real_spec_from_file_location = importlib.util.spec_from_file_location

    class FailingLoader:
        def create_module(self, _spec):
            return None

        def exec_module(self, _module):
            raise ImportError("deliberate parity import failure")

    def failing_model_spec(name, location):
        if name == "_locked_sdr_model":
            return importlib.util.spec_from_loader(name, FailingLoader())
        return real_spec_from_file_location(name, location)

    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        failing_model_spec,
    )

    with pytest.raises(ImportError, match="deliberate parity import failure"):
        with _load_upstream_sdr_modules():
            pass

    assert all(name not in sys.modules for name in private_names)


def test_temporal_branch_matches_public_code_after_semantic_weight_mapping():
    torch.manual_seed(17)
    recurrent = torch.randn(4, 2, 8)
    qmask = torch.zeros(2, 4, dtype=torch.long)
    umask = torch.ones(2, 4)
    lengths = [4, 4]
    branch = SDRRelationBranch(
        recurrent_dim=8,
        graph_hidden=3,
        num_relations=3,
        relation="temporal",
        n_speakers=1,
        window_past=2,
        window_future=2,
        dropout=0.0,
    ).eval()

    with _load_upstream_sdr_modules() as (upstream_graph, upstream_model):
        upstream_branch = upstream_model.GraphNetwork(
            num_features=8,
            num_relations=3,
            time_attn=False,
            hidden_size=3,
            dropout=0.0,
            no_cuda=True,
        ).eval()
        upstream_nodes, edge_index, edge_type, upstream_mapping = (
            upstream_graph.batch_graphify(
                recurrent.unsqueeze(2),
                qmask,
                lengths,
                1,
                2,
                2,
                "temporal",
                True,
            )
        )
        with torch.no_grad():
            for relation_name, explicit_id in TEMPORAL_RELATION_TABLE.items():
                upstream_id = upstream_mapping[relation_name]
                branch.rgcn.weight[explicit_id].copy_(
                    upstream_branch.conv1.weight[upstream_id]
                )
            branch.rgcn.root.copy_(upstream_branch.conv1.root)
            branch.rgcn.bias.copy_(upstream_branch.conv1.bias)
            branch.hypergraph.bias.copy_(upstream_branch.hypergraph.bias)
            branch.high_conv.gate.load_state_dict(
                upstream_branch.high_conv.gate.state_dict()
            )
            branch.post_graph_bigru.load_state_dict(
                upstream_branch.grufusion.state_dict()
            )
            branch.output_linear.load_state_dict(
                upstream_branch.linear.state_dict()
            )

            expected = upstream_branch(
                upstream_nodes,
                edge_index,
                edge_type,
                lengths,
                umask,
            )[0]
            actual = branch(recurrent, qmask, umask, lengths)

    assert actual.shape == expected.shape == (4, 2, 11)
    assert (actual - expected).abs().max().item() < 1e-6
