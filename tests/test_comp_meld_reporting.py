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
    assert "--test_correct {correct_count} --test_incorrect {incorrect_count}" in source
