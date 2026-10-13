from pathlib import Path


def test_nested_changes_only_the_meaningful_block(monkeypatch, tmp_path):
    monkeypatch.setenv("OSRAM_GCNET_MASK_OUTPUT", str(tmp_path / "results"))
    from experiments.osram_gcnet_random_mask_20261013.run import (
        apply_iemocap_overrides,
    )

    source = {
        "epochs": 17,
        "seed": 1,
        "fold": 5,
        "dataset": "old",
        "missing_mask_protocol": "cyclic",
        "osram_meaningful_block": "none",
        "learning_rate": 0.001,
    }
    flat = apply_iemocap_overrides(source, 4, 66, 2, "flat")
    nested = apply_iemocap_overrides(source, 4, 66, 2, "nested")

    assert flat["missing_mask_protocol"] == nested["missing_mask_protocol"] == "gcnet-random-mask-v1"
    assert flat["epochs"] == nested["epochs"] == 100
    assert flat["dataset"] == nested["dataset"] == "IEMOCAPFour"
    assert flat["osram_meaningful_block"] == "none"
    assert nested["osram_meaningful_block"] == "nested_gnn_rooted_evidence"
    assert {k: v for k, v in flat.items() if k != "osram_meaningful_block"} == {
        k: v for k, v in nested.items() if k != "osram_meaningful_block"
    }


def test_nested_dispatch_is_exactly_three_seeds_five_folds_two_tasks(tmp_path: Path):
    from experiments.osram_gcnet_random_mask_20261013.dispatch_nested import jobs

    rows = jobs(tmp_path)
    assert len(rows) == 30
    assert len({row["name"] for row in rows}) == 30
    assert {(row["classes"], row["gpu"]) for row in rows} == {(4, 0), (6, 1)}
    assert {row["seed"] for row in rows} == {66, 67, 68}
    assert {row["fold"] for row in rows} == {1, 2, 3, 4, 5}
    assert all("--model-variant" in row["command"] for row in rows)
    assert all("nested" in row["command"] for row in rows)

