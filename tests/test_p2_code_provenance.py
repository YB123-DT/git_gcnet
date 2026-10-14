"""P2 provenance must survive git-archive deployments without hiding I/O errors."""

import hashlib
import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from experiments.osram_history_training_controls_20261014 import run


SOURCE_FILES = (
    "gcnet_missing_m3/causal_gru_history.py",
    "gcnet_missing_m3/legacy_fixed_mask.py",
    "gcnet_missing_m3/osram.py",
    "gcnet_missing_m3/model.py",
    "gcnet_missing_m3/train_gcnet.py",
    "experiments/osram_history_training_controls_20261014/run.py",
    "experiments/osram_meld_flat_source_combinations_20261013/legacy_schedule.py",
)


def snapshot(tmp_path):
    root = tmp_path / "p2_snapshot_cfa8a6e"
    root.mkdir()
    for name in SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# source fixture: {name}\n")
    return root


def test_non_git_snapshot_records_identity_and_hashes_without_invoking_git(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    monkeypatch.setenv("OSRAM_P2_SOURCE_COMMIT", "cfa8a6e")
    monkeypatch.setattr(run.subprocess, "check_output", lambda *args, **kwargs:
                        pytest.fail("git archive snapshot must not invoke git"))
    provenance, diff = run.collect_code_provenance(root)
    assert provenance["code_source_kind"] == "snapshot"
    assert provenance["commit"] == "cfa8a6e"
    assert provenance["snapshot_identity"]["name"] == "p2_snapshot_cfa8a6e"
    assert provenance["snapshot_identity"]["path"] == str(root.resolve())
    assert not provenance["code_diff_available"]
    assert provenance["code_diff_sha256"] is None
    assert diff == ""
    for name in SOURCE_FILES:
        assert provenance["source_sha256"][name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    monkeypatch.delenv("OSRAM_P2_SOURCE_COMMIT")
    unknown, _ = run.collect_code_provenance(root)
    assert unknown["commit"] == "unknown"


def test_git_worktree_preserves_commit_and_tracked_diff_with_bound_cwd(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    # Linked worktrees use a .git file instead of a directory.
    (root / ".git").write_text("gitdir: fixture\n")
    monkeypatch.setenv("OSRAM_P2_SOURCE_COMMIT", "must-not-override-git")
    calls = []
    def git_output(command, **kwargs):
        assert kwargs["cwd"] == root.resolve()
        calls.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        assert command[:4] == ["git", "diff", "HEAD", "--"]
        return "diff --git a/file b/file\n"
    monkeypatch.setattr(run.subprocess, "check_output", git_output)
    provenance, diff = run.collect_code_provenance(root)
    assert provenance["code_source_kind"] == "git-worktree"
    assert provenance["commit"] == "a" * 40
    assert provenance["code_diff_available"]
    assert provenance["code_diff_sha256"] == hashlib.sha256(diff.encode()).hexdigest()
    assert len(calls) == 2


def test_snapshot_hash_io_errors_propagate(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    def failed_hash(path):
        raise PermissionError("source read denied")
    monkeypatch.setattr(run, "_sha256", failed_hash)
    with pytest.raises(PermissionError, match="source read denied"):
        run.collect_code_provenance(root)


def test_run_fold_non_git_archive_reaches_trainer_and_writes_provenance(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    data = root / "dataset.pkl"
    data.write_bytes(b"fixture dataset")
    monkeypatch.chdir(root)
    monkeypatch.setattr(run, "__file__", str(root / SOURCE_FILES[5]))
    monkeypatch.setenv("OSRAM_P2_SOURCE_COMMIT", "cfa8a6e")
    monkeypatch.setattr(run, "validate_visible_devices", lambda: [])
    monkeypatch.setattr(run, "dataset_inputs", lambda dataset: (data, ["MELD-pickle"] * 3, None))
    @dataclass
    class Config:
        dataset: str = "MELD"
    monkeypatch.setattr(run, "build_config", lambda **kwargs: Config())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda count: None, set_num_interop_threads=lambda count: None,
        __version__="fixture"))
    calls = []
    def train(config, *args, **kwargs):
        calls.append(config)
        return {"selection_metric": "weighted_f1"}
    monkeypatch.setitem(sys.modules, "gcnet_missing_m3.train_gcnet", SimpleNamespace(run_experiment=train))
    task = dict(dataset="MELD", control="local-only", seed=66, fold=1)
    destination = tmp_path / "outputs"
    run.run_fold(task, destination)
    provenance = json.loads((run.output_path(destination, task) / "PROVENANCE.json").read_text())
    assert len(calls) == 1
    assert provenance["status"] == "complete"
    assert provenance["commit"] == "cfa8a6e"
    assert provenance["code_source_kind"] == "snapshot"
    assert all(name in provenance["source_sha256"] for name in SOURCE_FILES)
