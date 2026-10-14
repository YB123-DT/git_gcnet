import numpy as np

from experiments.osram_meld_information_availability_20261014.analyze import (
    aggregate_records,
    classify_case,
    render_markdown,
)


def _records(delta=0.02):
    records = []
    for seed in (66, 67, 68):
        for index in range(8):
            rate = index / 10
            for variant, score in (
                ("Local", 0.50),
                ("Memory", 0.54), ("MemoryDonor", 0.54 - delta),
                ("LocalMemory", 0.58), ("LocalMemoryDonor", 0.58 - delta),
                ("RawHistory", 0.53), ("RawHistoryDonor", 0.53 - delta),
                ("LocalRawHistory", 0.57), ("LocalRawHistoryDonor", 0.57 - delta),
            ):
                records.append({
                    "seed": seed, "rate": rate, "variant": variant,
                    "accuracy": score + rate / 100, "weighted_f1": score,
                    "n": 100,
                })
    return records


def test_aggregate_reports_per_rate_eight_rate_high_and_sample_std():
    result = aggregate_records(_records())
    local = result["variants"]["Local"]
    assert local["per_rate"]["0.0"]["weighted_f1"]["mean"] == 0.5
    assert local["eight_rate"]["weighted_f1"]["mean"] == 0.5
    assert local["high_missing"]["weighted_f1"]["mean"] == 0.5
    assert local["eight_rate"]["weighted_f1"]["sample_std"] == 0.0


def test_paired_deltas_are_real_minus_donor_on_same_records():
    result = aggregate_records(_records(delta=0.03))
    delta = result["paired_deltas"]["LocalMemory"]
    assert np.isclose(delta["eight_rate"]["weighted_f1"]["mean"], 0.03)
    assert delta["coverage"]["minimum_n"] == 100


def test_decision_rules_do_not_call_redundant_memory_case_c():
    assert classify_case(raw_history_signal=False, memory_signal=False, conditional_memory_signal=False) == "A"
    assert classify_case(raw_history_signal=True, memory_signal=False, conditional_memory_signal=False) == "B"
    assert classify_case(raw_history_signal=True, memory_signal=True, conditional_memory_signal=True) == "C"
    assert classify_case(raw_history_signal=False, memory_signal=True, conditional_memory_signal=False) == "mixed"


def test_markdown_contains_per_rate_acc_wf1_and_paired_deltas():
    result = aggregate_records(_records())
    text = render_markdown(result, {
        "case": "C", "raw_history_signal": True,
        "memory_signal": True, "conditional_memory_signal": True,
    })
    assert "## Per-rate ACC" in text
    assert "## Per-rate W-F1" in text
    assert "## Per-rate paired ΔW-F1" in text
    assert "| LocalMemory |" in text
