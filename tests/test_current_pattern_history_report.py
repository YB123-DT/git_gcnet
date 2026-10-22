import gzip
import importlib
import json

import pytest


MODULE = "experiments.osram_current_pattern_history_audit_20261022.report"


def _module():
    return importlib.import_module(MODULE)


def _write(tmp_path, history=(), gradients=(), complete=True):
    (tmp_path / "STATUS.json").write_text(json.dumps({
        "collection_complete": complete, "status": "running",
        "protocol": {"seed": 66, "history_rate": "0.3"},
    }))
    for name, rows in (("HISTORY_ROWS", history), ("GRADIENT_PAIRS", gradients)):
        with gzip.open(tmp_path / (name + ".jsonl.gz"), "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


def _history(t, label, real, empty, replaced, donor="other", length=2):
    return dict(conversation_id="c", target_t=t, split="test", pattern="A",
                history_rate="0.3", label=label, pred_correct=real,
                pred_empty=empty, pred_replaced=replaced, donor_id=donor,
                history_length=length)


def _gradient(c, cosine, jacobian, opposed=False):
    return dict(conversation_id=c, target_t=3, split="train", history_rate="0.3",
                pattern_a="A", pattern_b="T", label=1, residual_a=-1,
                residual_b=1 if opposed else -1, residual_sign_opposed=opposed,
                group="Writer", loss_cosine=cosine, loss_dot=0,
                grad_norm_a=0 if cosine is None else 1, grad_norm_b=1,
                jacobian_cosine=jacobian)


def test_requires_closed_verified_collection(tmp_path):
    _write(tmp_path, complete=False)
    with pytest.raises(ValueError, match="collection_complete"):
        _module().generate(tmp_path)
    assert not (tmp_path / "REPORT.json").exists()


def test_metrics_align_neutral_and_paired_support(tmp_path):
    rows = [_history(0, 1, .1, .1, None, donor=None, length=0),
            _history(1, 1, .8, -.4, -.2),
            _history(2, -1, .2, -.8, .3),
            _history(3, 0, 2, 0, None, donor=None)]
    _write(tmp_path, rows)
    report = _module().generate(tmp_path)
    all_metrics = report["history"]["all"]["A"]["correct"]
    assert all_metrics["N"] == 4
    assert all_metrics["polarity_N"] == 3
    assert all_metrics["ACC"] == pytest.approx(200 / 3)
    assert all_metrics["W_F1"] == pytest.approx(53.3333333333)
    assert all_metrics["MSE"] == pytest.approx((.81 + .04 + 1.44 + 4) / 4)
    paired = report["history"]["history_present"]["A"]["paired"]["empty"]
    assert paired["N"] == 3
    assert paired["corrections"] == 1 and paired["harms"] == 1
    assert paired["delta_MSE"] == pytest.approx((1.96 + .04 - .04 - 1.44 - 4) / 3)
    replacement = report["history"]["all"]["A"]["replaced"]
    assert replacement["N"] == 2 and replacement["unavailable"] == 2
    assert report["history"]["history_present"]["A"]["correct"]["N"] == 3
    assert len(list(tmp_path.glob("*.csv"))) == 2
    assert "INTERNAL DIAGNOSTIC ONLY" in (tmp_path / "RESULT.md").read_text()
    assert "ordinary BPTT" in (tmp_path / "RESULT.md").read_text()
    json.dumps(report, allow_nan=False)


def test_zero_norm_is_undefined_not_zero_and_matrices_are_symmetric(tmp_path):
    _write(tmp_path, gradients=[_gradient("a", -.5, .5, True),
                                _gradient("b", .25, .25),
                                _gradient("c", None, None)])
    report = _module().generate(tmp_path)
    writer = report["gradient"]["groups"]["Writer"]
    assert writer["pair_count"] == 3
    assert writer["loss_cosine"]["defined_count"] == 2
    assert writer["loss_cosine"]["mean"] == pytest.approx(-.125)
    assert writer["loss_cosine"]["negative_fraction"] == .5
    assert writer["jacobian_cosine"]["negative_fraction"] == 0
    assert writer["residual_sign_opposed_fraction"] == pytest.approx(1 / 3)
    matrix = report["gradient"]["writer_matrices"]["loss_cosine"]
    assert matrix["A"]["T"] == matrix["T"]["A"] == -.125
    assert matrix["A"]["V"] is None and matrix["A"]["A"] is None


def test_no_history_mismatch_and_duplicate_observations_rejected(tmp_path):
    row = _history(0, 1, .1, .2, None, donor=None, length=0)
    _write(tmp_path, [row])
    with pytest.raises(ValueError, match="no.history"):
        _module().generate(tmp_path)
    row = _history(1, 1, .1, .2, None, donor=None)
    _write(tmp_path, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        _module().generate(tmp_path)


def test_replacement_without_donor_and_nonfinite_predictions_rejected(tmp_path):
    _write(tmp_path, [_history(2, 1, .1, .1, .1, donor=None)])
    with pytest.raises(ValueError, match="donor"):
        _module().generate(tmp_path)
    _write(tmp_path, [_history(2, 1, float("nan"), .1, None, donor=None)])
    with pytest.raises(ValueError, match="finite"):
        _module().generate(tmp_path)


def test_multiple_gradient_targets_from_one_conversation_rejected(tmp_path):
    a, b = _gradient("c", .2, .2), _gradient("c", .1, .1)
    b["target_t"] = 4
    _write(tmp_path, gradients=[a, b])
    with pytest.raises(ValueError, match="one target"):
        _module().generate(tmp_path)
