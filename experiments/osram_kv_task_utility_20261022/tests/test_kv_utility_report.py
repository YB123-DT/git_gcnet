"""Regression tests for the frozen KV/task-utility association report."""
import importlib
import gzip
import json

import pytest


MODULE = "experiments.osram_kv_task_utility_20261022.report"


def api():
    return importlib.import_module(MODULE)


def record(q, utility, *, conversation="c1", rate="0.3", distance=1, pattern="AV"):
    return dict(conversation_id=conversation, rate=rate, distance=distance,
                source_t=0, target_t=distance, source_pattern="ATV",
                target_pattern=pattern, q_gain=q, e_before=1., e_after=1.-q,
                value_energy=2., write_update_norm=.1, label=1.,
                prediction_keep=.2, prediction_omit=.1,
                loss_keep=1., loss_omit=1.+utility, utility=utility,
                correction=False, harm=False, polarity_eligible=True)


def test_spearman_uses_average_ranks_for_ties():
    assert api().spearman([1, 1, 3], [1, 2, 3]) == pytest.approx(0.866025403784)


def test_spearman_constant_or_too_few_is_unidentified():
    assert api().spearman([1, 1], [1, 2]) is None
    assert api().spearman([1], [2]) is None
    assert api().spearman([], []) is None


def test_report_distinguishes_positive_neutral_and_negative_task_utility():
    result = api().build_report([
        record(.9, .2), record(.7, -.1), record(.2, 0), record(0, .1),
    ])
    overall = result["overall"]
    assert overall["N"] == 4
    assert overall["improve_count"] == 2
    assert overall["harm_count"] == 1
    assert overall["neutral_count"] == 1
    assert sum(overall["quadrants"].values()) == 4


def test_report_preserves_rate_distance_and_current_availability_groups():
    result = api().build_report([
        record(.2, .1), record(.3, -.2, rate="0.7", distance=4, pattern="A"),
    ])
    assert result["by_rate"]["0.3"]["N"] == 1
    assert result["by_distance"]["4"]["N"] == 1
    assert result["by_target_pattern"]["A"]["N"] == 1


def test_conversation_mean_does_not_overweight_longer_dialogues():
    result = api().build_report([
        record(.1, 1.), record(.2, 1.), record(.3, -1., conversation="c2"),
    ])
    assert result["overall"]["mean_utility"] == pytest.approx(1/3)
    assert result["overall"]["conversation_macro_mean_utility"] == pytest.approx(0.)


def test_nonfinite_records_are_rejected_instead_of_silently_dropped():
    with pytest.raises(ValueError, match="finite"):
        api().build_report([record(float("nan"), .1)])


def test_report_writes_machine_readable_and_caveated_text_without_training(tmp_path):
    result = api().write_report(tmp_path, [record(.2, .1)])
    assert result["overall"]["N"] == 1
    assert json.loads((tmp_path/"REPORT.json").read_text())["overall"]["N"] == 1
    assert (tmp_path/"AGGREGATE.csv").exists()
    text = (tmp_path/"RESULT.md").read_text()
    assert "INTERNAL DIAGNOSTIC ONLY" in text
    assert "not independent" in text
    assert "No new training" in text


def test_empty_report_is_valid_and_has_no_fabricated_statistics():
    row = api().build_report([])["overall"]
    assert row["N"] == 0
    assert row["spearman_q_utility"] is None
    assert row["mean_utility"] is None


def test_cluster_bootstrap_is_deterministic_and_uses_conversations():
    rows = [record(.1, .5), record(.2, .6),
            record(.3, -.1, conversation="c2"), record(.4, -.2, conversation="c2")]
    first = api().build_report(rows)
    assert first == api().build_report(rows)
    assert first["cluster_bootstrap"]["clusters"] == 2
    assert first["cluster_bootstrap"]["replicates"] == 200
    assert len(first["cluster_bootstrap"]["mean_utility_ci95"]) == 2


def test_gain_median_and_quartiles_are_descriptive_and_neutral_tolerance_explicit():
    rows = [record(.1, 1e-9), record(.2, -.1), record(.3, .2), record(.4, .3)]
    result = api().build_report(rows)
    assert result["overall"]["neutral_count"] == 1
    assert result["utility_tolerance"] == 1e-7
    assert sum(x["N"] for x in result["q_median_split"].values()) == 4
    assert sum(x["N"] for x in result["q_quartiles"].values()) == 4


def test_repeated_rate_pairs_do_not_claim_unique_targets():
    result = api().build_report([record(.1, .2), record(.1, .2)])
    assert result["overall"]["unique_writes"] == 1
    assert result["overall"]["unique_targets"] == 1
    assert result["by_distance_target_pattern"]["1|AV"]["N"] == 2


def test_cli_reads_gzip_rows_without_modifying_input(tmp_path):
    source = tmp_path/"rows.jsonl.gz"
    with gzip.open(source, "wt") as handle:
        handle.write(json.dumps(record(.1, .2))+"\n")
    before = source.read_bytes()
    api().main(["--rows", str(source), "--output", str(tmp_path/"report")])
    assert source.read_bytes() == before
    result = json.loads((tmp_path/"report/REPORT.json").read_text())
    assert result["overall"]["N"] == 1


def test_run_status_metadata_is_preserved_without_reinterpreting_it(tmp_path):
    status = {"checkpoint_sha256": "abc123", "status": "completed"}
    (tmp_path/"STATUS.json").write_text(json.dumps(status))
    result = api().write_report(tmp_path, [record(.1, .2)])
    assert result["run_status"] == status
    assert json.loads((tmp_path/"STATUS.json").read_text()) == status


def test_neutral_labels_still_contribute_mse_but_not_polarity_counts():
    row = record(.2, .1)
    row.update(label=0., polarity_eligible=False, correction=True, harm=True)
    result = api().build_report([row])["overall"]
    assert result["mean_utility"] == pytest.approx(.1)
    assert result["polarity_eligible_count"] == 0
    assert result["corrections"] == result["harms"] == 0
