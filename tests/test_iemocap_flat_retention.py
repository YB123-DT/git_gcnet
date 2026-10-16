"""Flat restoration must share exact-artifact retention, not the old discard wrapper."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def worker():
    spec = importlib.util.spec_from_file_location('flat_retention_worker',
        ROOT / 'experiments/osram_iemocap_b6_transfer_20261015/worker.py')
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    return implementation


def configuration():
    cfg = json.loads((ROOT / 'experiments/osram_iemocap_b6_transfer_20261015/GPU_SMOKE.json').read_text())['config']
    cfg['osram_meaningful_block'] = 'none'
    for name in ('osram_rpca_rank', 'osram_rpca_steps', 'osram_rpca_threshold_init'):
        cfg.pop(name, None)
    cfg['osram_rpca_lr_multiplier'] = 1.0
    return cfg


def test_explicit_flat_validation_accepts_original_flat_not_b6():
    module = worker()
    cfg = configuration()
    module.validate(cfg, flat_only=True)
    with pytest.raises(ValueError):
        module.validate(cfg)
    for key, value in (('osram_meaningful_block', 'learned_robust_pca_evidence_decomposition'),
                       ('osram_rpca_lr_multiplier', 2.8), ('preserve_selection_artifacts', False),
                       ('missing_mask_protocol', 'conversation-mask-v2'), ('seed', 67)):
        with pytest.raises(ValueError):
            module.validate(dict(cfg, **{key: value}), flat_only=True)


def best_files(tmp_path, classes=4):
    from gcnet_missing_m3.train_gcnet import _save_selection_artifacts
    module = worker()
    for i in range(8):
        label = np.arange(classes)
        logits = np.eye(classes, dtype=np.float32)
        artifact = dict(labels=label, logits=logits, predictions=label,
                        availability=np.ones((classes, 3), dtype=np.float32),
                        conversation_ids=np.asarray(['one']*classes),
                        utterance_indices=np.arange(classes), speakers=np.zeros(classes))
        reference = _save_selection_artifacts(tmp_path, str(i / 10), 1, artifact)
        torch.save(dict(epoch=1, selection_accuracy=1.0, selection_artifact=reference,
                        model={'test_parameter': torch.ones(1)}), tmp_path / f'best_miss_0p{i}.pt')
    return module


def test_first_epoch_checks_real_saved_weights_masks_predictions_and_metrics(tmp_path):
    module = best_files(tmp_path)
    result = module.verify_first_epoch_retention(tmp_path, torch, 4, cuda_rng_states=1)
    assert result['status'] == 'complete'
    assert result['best_checkpoints'] == result['selected_artifacts'] == 8
    assert result['testbest']['mean']['ACC'] == result['testbest']['mean']['UA'] == 100
    assert (tmp_path / 'FIRST_EPOCH_RETENTION_CHECK.json').is_file()


def test_first_epoch_rejects_missing_checkpoint_before_continuing(tmp_path):
    module = best_files(tmp_path)
    (tmp_path / 'best_miss_0p7.pt').unlink()
    with pytest.raises(FileNotFoundError):
        module.verify_first_epoch_retention(tmp_path, torch, 4, cuda_rng_states=1)
    assert not (tmp_path / 'FIRST_EPOCH_RETENTION_CHECK.json').exists()


def test_first_epoch_requires_gpu_rng_for_complete_recovery(tmp_path):
    module = best_files(tmp_path)
    with pytest.raises(ValueError):
        module.verify_first_epoch_retention(tmp_path, torch, 4, cuda_rng_states=0)
