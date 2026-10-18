"""Selected-mask, single-trajectory MELD readout intervention contracts."""
import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

MODULE = "experiments.osram_meld_small_lr_readout_20261018.evaluate"


def _module():
    assert importlib.util.find_spec(MODULE) is not None, "Frozen MELD readout evaluator is missing"
    return importlib.import_module(MODULE)


class ToyOSRAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_dim = 2
        self.local_skip = nn.Linear(2, 5)
        self.emotion_adapter = nn.Linear(10, 5)
        self.emotion_norm = nn.LayerNorm(5)
        self.scans = 0

    def forward(self, inputs, availability, umask):
        self.scans += 1
        local = inputs[..., :2]
        base = inputs[..., 2:4].cumsum(0) - inputs[..., 2:4]
        gap = base.unsqueeze(-2).expand(-1, -1, 3, -1)
        gap = torch.where((1 - availability).bool().unsqueeze(-1), gap, 0.)
        flat = torch.cat((local, base, gap.flatten(2)), -1)
        hidden = self.emotion_norm(self.local_skip(local) + self.emotion_adapter(flat))
        return torch.where(umask.T.bool().unsqueeze(-1), hidden, 0.)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.osram = ToyOSRAM()
        self.smax_fc = nn.Linear(5, 7)

    def forward(self, inputs, availability, qmask, umask, lengths, predict_missing=False):
        return self.smax_fc(self.osram(inputs[0], availability, umask)), None, None, None


def _view(data, availability, dimensions):
    # Only a synthetic data adapter is replaced; all readouts use real torch layers.
    full = torch.cat(data[:3], -1)
    return dict(incomplete=full * availability.repeat_interleave(2, -1),
                availability=availability, qmask=data[6], umask=data[7], labels=data[8],
                lengths=data[7].sum(1).long().tolist(), conversation_ids=data[-1])


def _setup(monkeypatch):
    evaluator = _module()
    monkeypatch.setattr(evaluator, "replay_view_from_availability", _view)
    torch.manual_seed(72)
    model = ToyModel().eval().requires_grad_(False)
    raw = [torch.randn(3, 2, 2) for _ in range(6)]
    raw += [torch.tensor([[0, 1, 0], [1, 0, 0]]),
            torch.tensor([[1., 1., 1.], [1., 1., 0.]]),
            torch.tensor([[0, 1, 2], [3, 4, 0]]), ["convA", "convB"]]
    availability = torch.tensor([[[1., 0, 0], [0, 1, 1]],
                                 [[1, 1, 0], [0, 0, 1]],
                                 [[1, 1, 1], [0, 0, 0]]])
    view = _view(raw, availability, (2, 2, 2))
    logits = model([view["incomplete"]], availability, raw[6], raw[7], view["lengths"])[0]
    valid = raw[7].bool().numpy()
    saved = dict(logits=logits.transpose(0, 1).numpy()[valid],
                 labels=raw[8].numpy()[valid],
                 availability=availability.transpose(0, 1).numpy()[valid],
                 conversation_ids=np.broadcast_to(np.array(raw[-1])[:, None], valid.shape)[valid],
                 utterance_indices=np.broadcast_to(np.arange(3), valid.shape)[valid],
                 speakers=raw[6].numpy()[valid])
    saved["predictions"] = saved["logits"].argmax(1)
    model.osram.scans = 0
    cfg = SimpleNamespace(dataset="MELD", latent_dim=2, mosi_task_mode="regression")
    trainer = SimpleNamespace(_move_batch=lambda raw, device: raw)
    return evaluator, model, [raw], cfg, (2, 2, 2), saved, trainer


def test_three_conditions_reuse_one_scan_and_exact_selected_prediction(monkeypatch):
    evaluator, model, loader, cfg, dims, saved, trainer = _setup(monkeypatch)
    rng = torch.get_rng_state().clone()
    result, artifacts = evaluator.evaluate_rate(model, loader, cfg, dims, saved, trainer)
    assert model.osram.scans == result["checks"]["memory_scan_count"] == 1
    assert set(result["metrics"]) == {"L", "BG", "LBG"}
    assert result["checks"]["saved_predictions_exact"]
    assert torch.equal(rng, torch.get_rng_state())
    np.testing.assert_array_equal(artifacts["pred_LBG"], saved["predictions"])
    np.testing.assert_array_equal(artifacts["conversation_ids"], saved["conversation_ids"])
    np.testing.assert_allclose(artifacts["logits_LBG"], saved["logits"], rtol=2e-5, atol=2e-5)
    for scores in result["metrics"].values():
        assert scores["N"] == 5
        assert len(scores["recall"]) == len(scores["support"]) == 7
        assert np.isfinite(scores["CE"])
    cross = result["local_memory_cross"]
    assert sum(cross.values()) == 5
    full_correct = artifacts["pred_LBG"] == saved["labels"]
    local_correct = artifacts["pred_L"] == saved["labels"]
    assert result["local_to_full"]["corrections"] == int((~local_correct & full_correct).sum())
    assert result["local_to_full"]["harms"] == int((local_correct & ~full_correct).sum())


def test_memory_only_removes_both_local_paths_including_skip_bias(monkeypatch):
    evaluator, model, loader, cfg, dims, saved, trainer = _setup(monkeypatch)
    _, original = evaluator.evaluate_rate(model, loader, cfg, dims, saved, trainer)
    with torch.no_grad():
        model.osram.local_skip.bias.add_(100.)
    # Update only the selected artifact to the same altered full model.
    def compare_with_updated_saved():
        from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import scatter_saved_availability
        availability, _ = scatter_saved_availability(saved["availability"], 0, loader[0][7])
        view = _view(loader[0], availability, dims)
        logits = model([view["incomplete"]], availability, view["qmask"], view["umask"], view["lengths"])[0]
        updated = dict(saved, logits=logits.transpose(0, 1).numpy()[loader[0][7].bool().numpy()])
        updated["predictions"] = updated["logits"].argmax(1)
        return evaluator.evaluate_rate(model, loader, cfg, dims, updated, trainer)[1]
    changed = compare_with_updated_saved()
    np.testing.assert_array_equal(original["logits_BG"], changed["logits_BG"])


@pytest.mark.parametrize("field", ["conversation_ids", "utterance_indices", "speakers", "labels"])
def test_saved_metadata_misalignment_rejected_before_inference(monkeypatch, field):
    evaluator, model, loader, cfg, dims, saved, trainer = _setup(monkeypatch)
    saved[field] = saved[field][::-1].copy()
    if field == "speakers":  # This toy speaker trajectory is palindromic.
        saved[field][0] = 1 - saved[field][0]
    with pytest.raises((ValueError, AssertionError)):
        evaluator.evaluate_rate(model, loader, cfg, dims, saved, trainer)
    assert model.osram.scans == 0


def test_saved_logits_mismatch_rejected(monkeypatch):
    evaluator, model, loader, cfg, dims, saved, trainer = _setup(monkeypatch)
    saved["logits"][0, 0] += 1
    with pytest.raises((ValueError, AssertionError)):
        evaluator.evaluate_rate(model, loader, cfg, dims, saved, trainer)


def test_no_random_mask_generation_and_invalid_empty_mask_rejected(monkeypatch):
    evaluator, model, loader, cfg, dims, saved, trainer = _setup(monkeypatch)
    saved["availability"][0] = 0
    with pytest.raises(ValueError):
        evaluator.evaluate_rate(model, loader, cfg, dims, saved, trainer)
    assert model.osram.scans == 0


def test_frozen_eval_model_is_required(monkeypatch):
    evaluator, model, loader, cfg, dims, saved, trainer = _setup(monkeypatch)
    model.train()
    with pytest.raises(ValueError):
        evaluator.evaluate_rate(model, loader, cfg, dims, saved, trainer)


def test_diagnostic_targets_only_fixed_o6_configs_and_exact_sources():
    from experiments.osram_meld_small_lr_readout_20261018 import run
    from experiments.osram_meld_small_lr_3seed_20261018.run import make_config
    assert run.CONDITIONS == ('L', 'BG', 'LBG')
    assert set(run.SOURCES) == {66, 67, 68}
    assert run.expected_config(67) == make_config(67)
    assert run.expected_config(68) == make_config(68)
    assert run.expected_config(66)['learning_rate'] == 1e-5
    for invalid in (65,69,True,'66'):
        with pytest.raises(ValueError):run.expected_config(invalid)
