"""Finite Block2 screening must rank common-mask paired results, never oracle UA."""
import copy
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "experiments/osram_meld_block2_auto_20261014/summarize.py"
RATES = tuple(f"{i / 10:.1f}" for i in range(8))


def api():
    assert SOURCE.exists(), "The paired-screen summarizer is not implemented"
    spec = importlib.util.spec_from_file_location("block2_summary", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(method, acc=60.0, ua=36.0, status="complete"):
    rates = {
        rate: {"ACC": acc, "UA": ua, "WF1": 57.0, "CE": 1.2,
               "recall": [ua] * 7, "support": [10] * 7, "N": 70,
               "epoch": i + 1, "mask_sha256": "common-" + rate}
        for i, rate in enumerate(RATES)
    }
    block = {"rates": rates, "mean": {"ACC": acc, "UA": ua, "WF1": 57.0, "CE": 1.2},
             "high": {"ACC": acc, "UA": ua, "WF1": 57.0, "CE": 1.2}}
    return {"method": method, "status": status, "name": method + " name", "config": {"seed": 66},
            "provenance": {"seed": 66, "code_sha256": "code", "data_sha256": "data",
                           "mask_protocol": "gcnet-random-mask-v1"},
            "parameters": 123, "totaltrainsec": 12.5, "peakmemorybytes": 1024,
            "testbest": copy.deepcopy(block),
            "paired": {"on": copy.deepcopy(block), "off": copy.deepcopy(block),
                       "changes": {rate: {"corrections": 0, "harms": 0,
                                         "both_correct": 42, "both_wrong": 28, "N": 70}
                                   for rate in RATES}}}


def reports():
    return [report("Flat")] + [report(f"B{i}", 59.0, 35.0) for i in range(1, 7)]


def test_rank_uses_paired_eight_rate_mean_and_fixed_gate():
    rows = reports()
    rows[1] = report("B1", 60.4, 36.1)
    rows[2] = report("B2", 60.1, 36.4)
    rows[3] = report("B3", 61.0, 35.49)  # UA safety exclusion despite positive score.
    rows[4] = report("B4", 59.69, 40.0)  # ACC safety exclusion.
    rows[5] = report("B5", 60.0, 36.0)  # Zero is not signal.
    rows[6] = report("B6", 60.2, 36.3)
    rows[3]["testbest"]["mean"] = {"ACC": 100, "UA": 100}
    result = api().pick_candidates(rows)
    assert result["selected"] == ["B1", "B6"]
    assert result["status"] == "ready_for_hpo"
    ranking = {row["method"]: row for row in result["rankings"]}
    assert ranking["B1"]["score"] == pytest.approx(.28)
    assert not ranking["B3"]["eligible"]
    assert not ranking["B4"]["eligible"]
    assert not ranking["B5"]["signal"]
    assert "B3" in result["pareto"]  # Pareto includes excluded but complete methods.
    assert "B4" in result["pareto"]


def test_equal_scores_tie_break_by_method_and_boundary_is_inclusive():
    rows = reports()
    rows[1] = report("B1", 59.7, 37.0)
    rows[2] = report("B2", 59.7, 37.0)
    result = api().pick_candidates(list(reversed(rows)))
    assert result["selected"] == ["B1", "B2"]


def test_no_signal_stops_hpo():
    result = api().pick_candidates(reports())
    assert result["selected"] == []
    assert result["status"] == "no_positive_signal"
    assert result["pareto"] == ["Flat"]


def test_unfinished_or_missing_methods_cannot_start_hpo():
    rows = reports()
    rows[1] = report("B1", 65, 50)
    rows[2]["status"] = "running"
    rows.pop()
    result = api().pick_candidates(rows)
    assert result["selected"] == []
    assert result["status"] == "incomplete"
    assert result["methods"]["B2"]["status"] == "running"
    assert result["methods"]["B6"]["status"] == "missing"
    assert "B2" not in result["complete_methods"]


def test_failed_method_is_reported_and_complete_candidates_can_rank():
    rows = reports()
    rows[1] = report("B1", 61, 37)
    rows[6]["status"] = "failed"
    result = api().pick_candidates(rows)
    assert result["selected"] == ["B1"]
    assert result["methods"]["B6"]["status"] == "failed"


@pytest.mark.parametrize("damage", ["mask", "seed", "code", "data", "protocol", "rates", "nan", "N", "support"])
def test_invalid_comparison_cannot_rank(damage):
    rows = reports()
    rows[1] = report("B1", 70, 50)
    target = rows[1]
    if damage == "mask":
        target["paired"]["on"]["rates"]["0.3"]["mask_sha256"] = "different"
    elif damage == "seed":
        target["provenance"]["seed"] = 67
    elif damage in ("code", "data", "protocol"):
        key = {"code": "code_sha256", "data": "data_sha256", "protocol": "mask_protocol"}[damage]
        target["provenance"][key] = "different"
    elif damage == "rates":
        del target["paired"]["on"]["rates"]["0.7"]
    elif damage == "nan":
        target["paired"]["on"]["rates"]["0.3"]["ACC"] = float("nan")
    elif damage == "N":
        target["paired"]["on"]["rates"]["0.3"]["N"] = 71
    else:
        target["paired"]["on"]["rates"]["0.3"]["support"] = [11] * 7
    result = api().pick_candidates(rows)
    assert result["selected"] == []
    assert result["status"] == "invalid_comparison"
    assert result["methods"]["B1"]["reasons"]


def test_flat_on_off_must_be_identical():
    rows = reports()
    rows[0]["paired"]["off"]["rates"]["0.0"]["UA"] = 90
    assert api().pick_candidates(rows)["status"] == "invalid_comparison"


def test_recompute_means_from_per_rate_values():
    rows = reports()
    rows[1]["paired"]["on"]["mean"] = {"ACC": 100, "UA": 100}
    assert api().pick_candidates(rows)["selected"] == []


def test_output_keeps_oracle_and_paired_results_separate_and_full_trials(tmp_path):
    rows = reports()
    rows[1] = report("B1", 61, 37)
    rows[1]["testbest"]["rates"]["0.0"]["ACC"] = 99
    for row in rows:
        path = tmp_path / "screen" / row["method"] / "SUMMARY.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row))
    trials = {"studies": [{"method": "B1", "trials": [{"number": 0, "state": "COMPLETE",
              "params": {"lr": .001}, "value": .8}, {"number": 1, "state": "FAIL", "value": None}],
              "best_config": {"lr": .001}}]}
    (tmp_path / "optuna_search.json").write_text(json.dumps(trials))
    result = api().summarize(tmp_path)
    assert result["reports"]["B1"]["testbest"]["rates"]["0.0"]["ACC"] == 99
    assert result["reports"]["B1"]["paired"]["on"]["rates"]["0.0"]["ACC"] == 61
    assert result["optuna"]["studies"][0]["trials"][1]["state"] == "FAIL"
    assert result["hpo"]["B1"]["best_config"] == {"lr": .001}
    text = (tmp_path / "RESULT.md").read_text()
    for phrase in ("Test-best ACC", "common-mask", "single-seed", "validation", "Block2 off", "Pareto", "Optuna", "99.000", "parameters", "0.7"):
        assert phrase in text
    assert "testbest" in (tmp_path / "SUMMARY.csv").read_text()
    assert json.loads((tmp_path / "SUMMARY.json").read_text())["selected"] == ["B1"]


def test_no_trials_cannot_fabricate_hpo_best_configs(tmp_path):
    result = api().summarize(tmp_path, reports=reports())
    assert result["hpo"] == {}
    assert "not started" in (tmp_path / "RESULT.md").read_text()


def test_selected_method_without_hpo_has_explicit_pending_status(tmp_path):
    rows = reports()
    rows[1] = report("B1", 61, 37)
    result = api().summarize(tmp_path, reports=rows)
    assert result["hpo"]["B1"]["status"] == "not started"
    assert result["hpo"]["B1"]["best_config"] is None


def test_hpo_cannot_change_original_screen_selection(tmp_path):
    rows = reports()
    rows[1] = report("B1", 61, 37)
    trial_export = {"studies": [{"method": "B6", "status": "complete",
                                "best_config": {"fake_high_score": 999}, "trials": []}]}
    result = api().summarize(tmp_path, reports=rows, optuna_results=trial_export)
    assert result["selected"] == ["B1"]
    assert result["hpo"]["B6"]["best_config"] == {"fake_high_score": 999}


def test_sqlite_export_keeps_every_trial_and_distribution_metadata(tmp_path):
    path = tmp_path / "study.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
          CREATE TABLE studies(study_id INTEGER, study_name TEXT);
          CREATE TABLE trials(trial_id INTEGER, study_id INTEGER, number INTEGER, state TEXT);
          CREATE TABLE trial_params(trial_id INTEGER, param_name TEXT, param_value REAL, distribution_json TEXT);
          CREATE TABLE trial_values(trial_id INTEGER, objective INTEGER, value REAL);
          INSERT INTO studies VALUES(1, 'B1');
          INSERT INTO trials VALUES(1, 1, 0, 'COMPLETE'), (2, 1, 1, 'PRUNED'), (3, 1, 2, 'FAIL');
          INSERT INTO trial_params VALUES(1, 'width', 0, '{"choices": [64, 128]}');
          INSERT INTO trial_values VALUES(1, 0, 0.8);
        """)
    result = api().summarize(tmp_path, reports=reports())
    trials = result["optuna"]["sqlite_exports"][0]["studies"][0]["trials"]
    assert [row["state"] for row in trials] == ["COMPLETE", "PRUNED", "FAIL"]
    assert trials[0]["trial_params"][0]["distribution_json"] == '{"choices": [64, 128]}'
    assert trials[0]["trial_values"][0]["value"] == .8


def test_worker_provenance_schema_and_off_epoch_are_checked():
    rows = reports()
    for row in rows:
        row["provenance"] = {"seed": 66, "git_sha": "commit",
                             "code_hashes": {"worker.py": "sha"}, "dataset_sha256": "data",
                             "mask_protocol": "gcnet-random-mask-v1"}
    assert api().pick_candidates(rows)["status"] == "no_positive_signal"
    rows[1]["paired"]["off"]["rates"]["0.0"]["epoch"] += 1
    assert api().pick_candidates(rows)["status"] == "invalid_comparison"


def test_duplicate_ids_fail_instead_of_silently_overwriting():
    with pytest.raises(ValueError, match="Duplicate"):
        api().pick_candidates([report("Flat"), report("Flat")])


def test_best_config_without_run_status_does_not_claim_completion(tmp_path):
    result = api().summarize(tmp_path, reports=reports(), optuna_results={"studies": [
        {"method": "B1", "best_config": {"lr": .001}, "trials": []}]})
    assert result["hpo"]["B1"]["status"] == "reported (status not supplied)"
