from gcnet_missing_m3_text_anchor.train_mosi import select_best_epoch


def test_best_epoch_uses_validation_weighted_f1():
    history = [
        {"epoch": 1, "validation": {"loss": 1.0, "weighted_f1": 0.82}},
        {"epoch": 2, "validation": {"loss": 0.8, "weighted_f1": 0.80}},
        {"epoch": 3, "validation": {"loss": 0.9, "weighted_f1": 0.85}},
    ]
    assert select_best_epoch(history) == 3
