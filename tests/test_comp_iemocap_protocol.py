import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "comp_causal_iemocap_3seed_20261013"
    / "run_one.sh"
)


def test_iemocap_budget_matches_paper_training_curve():
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--print-budget", "IEMOCAPFour"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "epochs=200 stage_epoch=100"


def test_iemocap_runner_rejects_non_iemocap_dataset():
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--print-budget", "CMUMOSEI"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0

