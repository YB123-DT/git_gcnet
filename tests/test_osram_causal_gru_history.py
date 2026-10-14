"""P2 controls: a strictly past ordinary GRU, with unchanged default OSRAM."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone


def inputs():
    torch.manual_seed(17)
    node = torch.randn(4, 2, 8)
    latents = {name: torch.randn_like(node) for name in ("audio", "text", "visual")}
    availability = torch.ones(4, 2, 3)
    availability[2:, 1] = 0
    umask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
    qmask = torch.zeros(2, 4, dtype=torch.long)
    return node, latents, availability, qmask, umask, [4, 2]


def backbone(enabled=True):
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
                         value_dim=4, n_speakers=1, dropout=0., bidirectional=False,
                         osram_causal_gru_history=enabled).eval()


def test_gru_reads_strictly_before_current_update_and_never_calls_chm():
    model = backbone()
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary GRU control must not execute CHM")
    model._scan = forbidden
    model._project_sequence = forbidden
    args = inputs()
    _, context = model(*args)
    changed = list(args)
    changed[0] = args[0].clone()
    changed[0][2:] += 100
    _, other = model(*changed)
    torch.testing.assert_close(context["base"][:3], other["base"][:3], rtol=0, atol=0)
    assert not torch.equal(context["base"][3, 0], other["base"][3, 0])
    assert torch.count_nonzero(context["base"][0]) == 0
    assert torch.count_nonzero(context["base"][..., 8:]) == 0
    assert torch.count_nonzero(context["gap"]) == 0
    assert model.causal_gru_history.gru.num_layers == 1
    assert model.causal_gru_history.gru.hidden_size == 256
    assert not model.causal_gru_history.gru.bidirectional
    assert not any(name.startswith(("key_projectors.", "value_projectors.", "query_projector."))
                   for name in model.state_dict())


def test_gru_conversations_reset_and_padding_is_zero_even_with_nan():
    model = backbone()
    args = inputs()
    _, batch = model(*args)
    solo = (args[0][:, :1], {k: v[:, :1] for k, v in args[1].items()},
            args[2][:, :1], args[3][:1], args[4][:1], [4])
    _, separate = model(*solo)
    torch.testing.assert_close(batch["base"][:, :1], separate["base"], rtol=1e-6, atol=1e-7)
    _, repeated = model(*args)
    torch.testing.assert_close(batch["base"], repeated["base"], rtol=0, atol=0)
    dirty = list(args)
    dirty[0] = args[0].clone()
    dirty[0][2:, 1] = float("nan")
    hidden, padded = model(*dirty)
    assert torch.isfinite(hidden).all()
    assert torch.count_nonzero(hidden[2:, 1]) == 0
    assert torch.count_nonzero(padded["base"][2:, 1]) == 0
    torch.testing.assert_close(batch["base"][:2, 1], padded["base"][:2, 1], rtol=0, atol=0)


def test_gru_gradient_reaches_past_nodes_but_not_current_or_future():
    model = backbone()
    args = list(inputs())
    args[0].requires_grad_()
    _, contexts = model(*args)
    contexts["base"][2, 0].square().sum().backward()
    assert args[0].grad[:2, 0].abs().sum() > 0
    assert torch.count_nonzero(args[0].grad[2:]) == 0
    assert model.causal_gru_history.gru.weight_ih_l0.grad.abs().sum() > 0
    assert model.causal_gru_history.project.weight.grad.abs().sum() > 0


def test_disabled_flag_preserves_legacy_parameters_rng_and_outputs():
    options = dict(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
                   value_dim=4, n_speakers=1, dropout=0., bidirectional=False)
    torch.manual_seed(42)
    legacy = OSRAMBackbone(**options).eval()
    legacy_rng = torch.get_rng_state()
    torch.manual_seed(42)
    disabled = OSRAMBackbone(**options, osram_causal_gru_history=False).eval()
    assert torch.equal(torch.get_rng_state(), legacy_rng)
    assert disabled.state_dict().keys() == legacy.state_dict().keys()
    disabled.load_state_dict(legacy.state_dict(), strict=True)
    first, contexts = legacy(*inputs())
    second, other = disabled(*inputs())
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for name in contexts:
        torch.testing.assert_close(contexts[name], other[name], rtol=0, atol=0)


def test_p2_configs_pin_dataset_protocols_and_variants():
    from experiments.osram_history_training_controls_20261014.run import build_config, tasks
    from gcnet_missing_m3.train_gcnet import _build_schedule
    from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import LegacyFixedMaskSchedule
    for dataset in ("IEMOCAPFour", "IEMOCAPSix", "MELD"):
        local = build_config(dataset, "local-only", 66, 1)
        gru = build_config(dataset, "causal-gru", 66, 1)
        assert local.osram_ablation == "local-only"
        assert not local.osram_causal_gru_history
        assert gru.osram_causal_gru_history and gru.osram_ablation == "full"
        assert gru.osram_emotion_ablation == "full"
        assert gru.osram_meaningful_block == "none"
        assert gru.training_objective == "emotion-only" and gru.disable_unused_aux_modules
        assert gru.osram_output_dim == 1600
        assert (gru.osram_num_heads, gru.osram_key_dim, gru.osram_value_dim) == (8, 64, 64)
        assert gru.checkpoint_selection == "test-oracle-per-rate"
        assert gru.checkpoint_selection_metric == "weighted_f1"
        assert gru.epochs == (50 if dataset == "MELD" else 100)
        assert gru.missing_mask_protocol == ("gcnet-legacy-fixed-v1" if dataset == "MELD" else "conversation-mask-v2")
        if dataset == "MELD":
            assert isinstance(_build_schedule(gru, "train", .3), LegacyFixedMaskSchedule)
    assert len(tasks("IEMOCAPSix", [66])) == 10
    assert len(tasks("IEMOCAPFour", [66])) == 10
    assert len(tasks("MELD", [66])) == 2
    counts = [len(tasks(dataset, [66, 67, 68])) for dataset in ("IEMOCAPFour", "IEMOCAPSix", "MELD")]
    assert counts == [30, 30, 6]
    assert sum(counts) == 66


def test_iemocap4_uses_existing_label_map_ecerc_roots_and_four_class_contract():
    import config
    from gcnet_missing_m3.train_gcnet import _dataset_shape
    from experiments.osram_ecerc_features_20261012.run import FEATURE_ROOT
    from experiments.osram_history_training_controls_20261014.run import dataset_inputs
    source, roots, manifest = dataset_inputs("IEMOCAPFour")
    assert str(source) == config.PATH_TO_LABEL["IEMOCAPFour"]
    assert source.name == "IEMOCAP_features_raw_4way.pkl"
    assert roots == [FEATURE_ROOT / modality for modality in ("audio", "text", "visual")]
    assert manifest == FEATURE_ROOT / "MANIFEST.json"
    assert _dataset_shape("IEMOCAPFour")["num_classes"] == 4
    assert _dataset_shape("IEMOCAPSix")["num_classes"] == 6


def test_dispatch_rejects_bad_physical_gpu4_on_real_user22_hostname(tmp_path, monkeypatch):
    from experiments.osram_history_training_controls_20261014 import run
    monkeypatch.setattr(run.socket, "gethostname", lambda: "user22")
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU4 must be refused before querying or spawning")
    monkeypatch.setattr(run.subprocess, "check_output", forbidden)
    with pytest.raises(ValueError, match="GPU4"):
        run.dispatch("MELD", [66], ("local-only",), tmp_path, 4, 1)


def test_direct_fold_refuses_numeric_gpu4_visibility_before_loading_data(tmp_path, monkeypatch):
    from experiments.osram_history_training_controls_20261014 import run
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    def forbidden(*args, **kwargs):
        raise AssertionError("physical GPU4 must be refused before constructing model/config")
    monkeypatch.setattr(run, "build_config", forbidden)
    with pytest.raises(ValueError, match="GPU4"):
        run.run_fold(dict(dataset="MELD", control="local-only", seed=66, fold=1), tmp_path)


@pytest.mark.parametrize("visible", [
    "GPU-46fb379f-cc90-dc82-9b5e-5d011f552264", "GPU-46fb379f",
])
def test_direct_fold_rejects_bad_uuid_and_unique_prefix(tmp_path, monkeypatch, visible):
    from experiments.osram_history_training_controls_20261014 import run
    monkeypatch.setattr(run.socket, "gethostname", lambda: "user22")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(run, "build_config", lambda **kwargs: pytest.fail("bad card reached config"))
    with pytest.raises(ValueError, match="GPU4"):
        run.run_fold(dict(dataset="MELD", control="local-only", seed=66, fold=1), tmp_path)


def test_user22_requires_explicit_device_visibility(tmp_path, monkeypatch):
    from experiments.osram_history_training_controls_20261014 import run
    monkeypatch.setattr(run.socket, "gethostname", lambda: "user22")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(run, "build_config", lambda **kwargs: pytest.fail("unrestricted devices reached config"))
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        run.run_fold(dict(dataset="MELD", control="local-only", seed=66, fold=1), tmp_path)


@pytest.mark.parametrize("visible", ["6", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "GPU-aaaaaaaa"])
def test_user22_resolves_only_explicit_healthy_physical_devices(monkeypatch, visible):
    from experiments.osram_history_training_controls_20261014 import run
    monkeypatch.setattr(run.socket, "gethostname", lambda: "user22")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    def inventory(command, **kwargs):
        assert "--query-gpu=index,uuid" in command
        return ("4, GPU-46fb379f-cc90-dc82-9b5e-5d011f552264\n"
                "6, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n")
    monkeypatch.setattr(run.subprocess, "check_output", inventory)
    resolved = run.validate_visible_devices()
    assert resolved == [{"physical_index": 6, "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}]


@pytest.mark.parametrize("visible", ["0", "GPU-", "GPU-unknown"])
def test_user22_unknown_or_ambiguous_selectors_cannot_enter_whitelist(monkeypatch, visible):
    from experiments.osram_history_training_controls_20261014 import run
    monkeypatch.setattr(run.socket, "gethostname", lambda: "user22")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(run.subprocess, "check_output", lambda *args, **kwargs:
        "4, GPU-46fb379f-cc90-dc82-9b5e-5d011f552264\n6, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n")
    with pytest.raises(ValueError):
        run.validate_visible_devices()


def test_explicit_weighted_f1_selection_does_not_change_legacy_default():
    from gcnet_missing_m3.train_gcnet import _checkpoint_selection_metric
    assert _checkpoint_selection_metric("MELD") == "accuracy"
    assert _checkpoint_selection_metric("IEMOCAPSix", "weighted_f1") == "weighted_f1"
    assert _checkpoint_selection_metric("MELD", "weighted_f1") == "weighted_f1"


def test_meld_masks_replay_frozen_schedule_for_training_and_evaluation():
    from experiments.osram_history_training_controls_20261014.run import build_config
    from gcnet_missing_m3.train_gcnet import _build_schedule, _prepare_view
    from gcnet_modality_jepa.protocol import SeedBundle
    from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import LegacyFixedMaskSchedule
    cfg = build_config("MELD", "causal-gru", 66, 1)
    for split in ("train", "test"):
        schedule = _build_schedule(cfg, split, .3)
        frozen = LegacyFixedMaskSchedule("MELD", split, 1, .3, SeedBundle(66).derive("missing_mask"))
        for epoch in (0, 7):
            actual = schedule.generate("dialog", 12, "host", epoch, 9)
            expected = frozen.generate("dialog", 12, "host", epoch, 9)
            np.testing.assert_array_equal(actual.availability, expected.availability)
            assert actual.schedule_hash == expected.schedule_hash
    # Train/eval take the same ordinary view preparation path, preserving
    # host/guest fusion and padded lengths, rather than constructor relabeling.
    tensors = [torch.randn(12, 1, width) for width in (2, 3, 1, 2, 3, 1)]
    qmask = torch.zeros(1, 12, dtype=torch.long)
    umask = torch.tensor([[1] * 9 + [0] * 3], dtype=torch.float32)
    data = [*tensors, qmask, umask, torch.zeros(1, 12, dtype=torch.long), ["dialog"]]
    train = _build_schedule(cfg, "train", .3)
    view = _prepare_view(data, train, 7, (2, 3, 1))
    expected = train.generate("dialog", 12, "host", 7, 9)
    np.testing.assert_array_equal(view["availability"][:, 0].numpy(), expected.availability)


def test_gru_config_rejects_nonmean_or_auxiliary_training():
    from experiments.osram_history_training_controls_20261014.run import build_config
    cfg = build_config("IEMOCAPSix", "causal-gru", 66, 1)
    for change in ({"fusion_type": "slot"}, {"training_objective": "joint"},
                   {"osram_bidirectional": True}, {"osram_meaningful_block": "nested_gnn_rooted_evidence"}):
        with pytest.raises(ValueError, match="causal GRU"):
            replace(cfg, **change)


@pytest.mark.parametrize("field,value", [
    ("initial_backbone_checkpoint", "backbone.pt"),
    ("joint_pretrain_checkpoint", "projectors.pt"),
    ("b2_base_checkpoint", "b2.pt"),
    ("b2_pretrain_checkpoint", "b2_pretrain.pt"),
    ("teacher_checkpoint", "teacher.pt"),
    ("text_subspace_checkpoint", "subspace.pt"),
    ("pretrained_learning_rate", 1e-4),
    ("teacher_mode", "pretrained-frozen"),
])
def test_gru_config_rejects_every_pretrained_source(field, value):
    from experiments.osram_history_training_controls_20261014.run import build_config
    cfg = build_config("IEMOCAPFour", "causal-gru", 66, 1)
    with pytest.raises(ValueError, match="causal GRU"):
        replace(cfg, **{field: value})


def test_gru_model_connects_observed_mean_encoder_flat_classifier_and_optimizer():
    from gcnet_missing_m3.model import MissingM3GraphModel
    model = MissingM3GraphModel(
        base_model="LSTM", adim=2, tdim=3, vdim=1, D_e=4, graph_hidden_size=4,
        n_speakers=1, window_past=2, window_future=2, n_classes=3, no_cuda=True,
        latent_dim=8, dropout=0., projector_dropout=0., predictor_dropout=0.,
        backbone_type="osram", osram_output_dim=10, osram_num_heads=2,
        osram_key_dim=3, osram_value_dim=4, osram_bidirectional=False,
        osram_causal_gru_history=True, training_objective="emotion-only",
        disable_unused_aux_modules=True,
    )
    assert model.observed_set.fusion_type == "mean"
    assert model.teacher is None
    args = inputs()
    features = torch.randn(4, 2, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Flat's zero final layer initially blocks recurrent gradients. Its first
    # update activates that path, so verify real task gradients on step two.
    before = model.osram.causal_gru_history.gru.weight_ih_l0.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        logits, _, _, _ = model(features, args[2], args[3], args[4], args[5])
        loss = torch.nn.functional.cross_entropy(logits[1:, 0], torch.tensor([0, 1, 2]))
        loss.backward()
        optimizer.step()
    assert model.observed_set.projectors["audio"].fc1.weight.grad.abs().sum() > 0
    assert model.smax_fc.weight.grad.abs().sum() > 0
    assert model.osram.causal_gru_history.gru.weight_ih_l0.grad.abs().sum() > 0
    assert not torch.equal(before, model.osram.causal_gru_history.gru.weight_ih_l0)


def test_local_only_control_has_no_history_effect_on_target_hidden():
    model = OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
                         value_dim=4, n_speakers=1, dropout=0., bidirectional=False,
                         osram_ablation="local-only").eval()
    torch.nn.init.normal_(model.emotion_adapter[-1].weight, std=.1)
    args = inputs()
    hidden, contexts = model(*args)
    changed = list(args)
    changed[0] = args[0].clone()
    changed[0][:2] += 100
    changed[1] = {k: v.clone() for k, v in args[1].items()}
    for latent in changed[1].values():
        latent[:2] += 100
    other, _ = model(*changed)
    torch.testing.assert_close(hidden[2:], other[2:], rtol=0, atol=0)
    assert torch.count_nonzero(contexts["base"]) == 0
    assert torch.count_nonzero(contexts["gap"]) == 0


def test_dispatch_commands_isolate_dataset_control_seed_and_fold(tmp_path):
    from experiments.osram_history_training_controls_20261014.run import tasks, child_command, output_path
    selected = tasks("IEMOCAPSix", [66, 67])
    assert len({output_path(tmp_path, task) for task in selected}) == 20
    command = child_command(selected[0], tmp_path)
    assert command[command.index("--dataset") + 1] == "IEMOCAPSix"
    assert command[command.index("--control") + 1] == "local-only"
    assert command[command.index("--fold") + 1] == "1"
    with pytest.raises(ValueError):
        tasks("MELD", [66, 66])
