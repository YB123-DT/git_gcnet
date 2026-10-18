"""Pure aggregation checks for the frozen five-fold IEMOCAP probes."""

import copy
import importlib
import math

import pytest


MODULE = "experiments.osram_iemocap_information_availability_20261018.analysis"
PAIRS = {
    "Base": "BaseDonor", "Memory": "MemoryDonor",
    "LocalBase": "LocalBaseDonor", "LocalMemory": "LocalMemoryDonor",
    "RawHistory": "RawHistoryDonor", "LocalRawHistory": "LocalRawHistoryDonor",
}
VARIANTS = ["Local", *[name for pair in PAIRS.items() for name in pair]]


def functions():
    try:
        spec = importlib.util.find_spec(MODULE)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "IEMOCAP analysis implementation is missing"
    module = importlib.import_module(MODULE)
    return module.aggregate, module.render_markdown


@pytest.fixture
def folds():
    summaries = []
    for dataset in ("IEMOCAPFour", "IEMOCAPSix"):
        for fold in range(1, 6):
            rates = {}
            for rate_id in range(8):
                fits = []
                for seed in (66, 67, 68):
                    for variant in VARIANTS:
                        delta = 2 if variant in PAIRS else 0
                        value = 20 + fold + rate_id + (seed - 66) + delta
                        n = 10 * fold
                        fits.append({
                            "seed": seed, "variant": variant,
                            "metrics": {"ACC": value, "UA": value + 1, "WF1": value - 1,
                                        "CE": value / 100, "N": n,
                                        "recall": [value] * (4 if dataset == "IEMOCAPFour" else 6),
                                        "support": [n, *([0] * (3 if dataset == "IEMOCAPFour" else 5))]},
                            "cross_vs_local": {"N": n, "corrections": fold,
                                               "harms": 2 * fold, "both_correct": 3 * fold,
                                               "both_wrong": 4 * fold},
                            **({"cross_vs_donor": {"N": n, "corrections": fold,
                                                    "harms": 2 * fold, "both_correct": 3 * fold,
                                                    "both_wrong": 4 * fold}} if variant in PAIRS else {}),
                        })
                rates[f"{rate_id / 10:.1f}"] = {"status": "complete", "fits": fits}
            summaries.append({"status": "complete", "dataset": dataset, "fold": fold,
                              "source_summary": f"/{dataset}/fold{fold}/SUMMARY.json", "rates": rates})
    return summaries


def test_macro_order_probe_seeds_and_fold_references(folds):
    aggregate, _ = functions()
    original = copy.deepcopy(folds)
    report = aggregate(folds)
    local = report["datasets"]["IEMOCAPFour"]["variants"]["Local"]
    assert local["eight_rate"]["ACC"]["values"] == pytest.approx([26.5, 27.5, 28.5])
    assert local["eight_rate"]["ACC"]["mean"] == pytest.approx(27.5)
    assert local["eight_rate"]["ACC"]["sample_std"] == pytest.approx(1)
    assert local["high_missing"]["ACC"]["mean"] == pytest.approx(30)
    assert local["per_rate"]["0.0"]["ACC"]["mean"] == pytest.approx(24)
    assert report["backbone_seed"] == 66
    assert report["probe_seeds"] == [66, 67, 68]
    assert report["fold_summaries"] == original
    assert folds == original


def test_real_donor_deltas_and_cross_counts(folds):
    aggregate, _ = functions()
    data = aggregate(folds)["datasets"]["IEMOCAPSix"]
    delta = data["paired_deltas"]["LocalMemory"]
    assert delta["control"] == "LocalMemoryDonor"
    assert delta["eight_rate"]["ACC"]["mean"] == pytest.approx(2)
    assert delta["per_rate"]["0.7"]["UA"]["sample_std"] == pytest.approx(0)
    cross = data["cross_vs_local"]["LocalMemory"]
    assert cross["per_rate"]["0.0"]["N"]["mean"] == 150
    assert cross["eight_rate"]["corrections"]["mean"] == 15
    assert cross["eight_rate"]["harms"]["mean"] == 30
    assert cross["event_totals_by_probe_seed"]["66"]["N"] == 1200
    assert data["cross_vs_donor"]["Memory"]["eight_rate"]["corrections"]["mean"] == 15


@pytest.mark.parametrize("mutation", ["missing_fold", "duplicate_fold", "missing_rate", "duplicate_rate",
                                      "missing_fit", "duplicate_fit", "unknown_seed", "unknown_variant",
                                      "not_complete", "bad_metric", "bad_cross", "donor_n"])
def test_reject_incomplete_or_invalid_grid(folds, mutation):
    aggregate, _ = functions()
    if mutation == "missing_fold":
        folds.pop()
    elif mutation == "duplicate_fold":
        folds.append(copy.deepcopy(folds[0]))
    elif mutation == "missing_rate":
        folds[0]["rates"].pop("0.1")
    elif mutation == "duplicate_rate":
        folds[0]["rates"]["0.00"] = copy.deepcopy(folds[0]["rates"]["0.0"])
    elif mutation == "missing_fit":
        folds[0]["rates"]["0.0"]["fits"].pop()
    elif mutation == "duplicate_fit":
        folds[0]["rates"]["0.0"]["fits"].append(copy.deepcopy(folds[0]["rates"]["0.0"]["fits"][0]))
    elif mutation == "unknown_seed":
        folds[0]["rates"]["0.0"]["fits"][0]["seed"] = 69
    elif mutation == "unknown_variant":
        folds[0]["rates"]["0.0"]["fits"][0]["variant"] = "Unknown"
    elif mutation == "not_complete":
        folds[0]["status"] = "running"
    elif mutation == "bad_metric":
        folds[0]["rates"]["0.0"]["fits"][0]["metrics"]["CE"] = math.nan
    elif mutation == "bad_cross":
        folds[0]["rates"]["0.0"]["fits"][0]["cross_vs_local"]["N"] += 1
    elif mutation == "donor_n":
        folds[0]["rates"]["0.0"]["fits"][2]["metrics"]["N"] += 1
    with pytest.raises(ValueError):
        aggregate(folds)


def test_markdown_includes_protocol_limits_and_required_tables(folds):
    aggregate, render = functions()
    text = render(aggregate(folds))
    for required in ("INTERNAL DIAGNOSTIC ONLY", "gcnet-random-mask-v1", "IEMOCAPFour", "IEMOCAPSix",
                     "LocalMemoryDonor", "0.7", "corrections", "harms", "Test-oracle",
                     "主干 seed66", "probe initialization", "容量", "不重新训练主干"):
        assert required in text
