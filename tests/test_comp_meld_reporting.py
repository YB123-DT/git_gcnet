from pathlib import Path


def test_meld_best_accuracy_summary_also_reports_companion_wf1():
    source = (
        Path(__file__).resolve().parents[1]
        / "external_repos"
        / "ComP_causal"
        / "ComP"
        / "train_mr.py"
    ).read_text()

    assert "--test_ua {bestcorr} --test_wf1 {bestf1}" in source
    assert "--test_ua {np.mean(folder_corr)} --test_wf1 {np.mean(folder_f1)}" in source
    assert "predictions=best_artifacts['predictions'][active]" in source
    assert "availability=best_artifacts['availability'].reshape(-1, 3)[active]" in source
    assert "--test_correct {correct_count} --test_incorrect {incorrect_count}" in source


def test_meld_runner_exposes_the_shared_fixed_mask_protocol():
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "comp_meld_20261012"
        / "run_one.sh"
    ).read_text()

    assert "COMP_MASK_PROTOCOL" in source
    assert '--mask-protocol="${MASK_PROTOCOL}"' in source
