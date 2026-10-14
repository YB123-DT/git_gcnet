"""Same-scan bypass and compact recovery contracts for the finite screen."""
import importlib
from pathlib import Path

import numpy as np
import torch
import subprocess
import sys
import os


def worker():
    return importlib.import_module('experiments.osram_meld_block2_auto_20261014.worker')


def test_input_bypass_preserves_original_local_skip():
    from gcnet_missing_m3.osram import OSRAMBackbone
    from types import SimpleNamespace
    module = worker()
    osram = OSRAMBackbone(latent_dim=8, output_dim=12, num_heads=8, key_dim=64,
                         value_dim=64, n_speakers=2, bidirectional=False, dropout=0.).eval()
    torch.nn.init.normal_(osram.emotion_adapter[-1].weight, std=.1)
    head = torch.nn.Linear(12, 7)
    model = SimpleNamespace(osram=osram, smax_fc=head)
    local = torch.randn(3, 2, 8)
    base = torch.randn(3, 2, 1024)
    gap = torch.randn(3, 2, 3, 1024)
    availability = torch.tensor([[[1., 0., 1.]] * 2] * 3)
    umask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    logits = module.original_flat_logits(model, local, base, gap, availability, umask)
    flat = torch.cat((local, base, (gap * (1 - availability)[..., None]).flatten(2)), -1)
    hidden = osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(flat))
    hidden = torch.where(umask.T.bool()[..., None], hidden, 0.)
    torch.testing.assert_close(logits, head(hidden))


def test_compact_state_prunes_only_unreferenced_versions(tmp_path):
    module = worker()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    state = module.CompactState(tmp_path, identity={'version': 'test'}).bind(model, optimizer)
    for epoch in (1, 2):
        checkpoint = dict(model=model.state_dict(), epoch=epoch)
        state.save_best(epoch=epoch, rate='0.0', checkpoint=checkpoint)
        state.commit_epoch(next_epoch=epoch, history=[{'epoch': i} for i in range(1, epoch + 1)],
                           selection_state={}, schedule_state={'next_epoch': epoch})
    assert len(list((tmp_path / 'training_versions').glob('*_model.pt'))) == 1
    restored = module.CompactState(tmp_path, identity={'version': 'test'}).bind(model, optimizer).restore()
    assert restored['next_epoch'] == 2
    assert (tmp_path / 'best_miss_0p0.pt').is_file()


def test_metric_and_class_probability_ce():
    module = worker()
    labels = np.array([0, 1, 2])
    logits = np.eye(7)[labels] * 2
    score = module.class_metrics(labels, logits)
    assert score['ACC'] == 100. and score['UA'] == 100. * 3 / 7
    assert len(score['recall']) == len(score['support']) == 7


def test_dataset_environment_precedes_trainer_import():
    result = subprocess.check_output([sys.executable, '-c',
        'import experiments.osram_meld_block2_auto_20261014.worker as w; '
        'import config; assert config.PATH_TO_LABEL["MELD"] == str(w.DATASET)'],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stderr=subprocess.STDOUT, text=True)


def test_smoke_constructor_is_trainable_not_frozen_inference_helper():
    import json
    from experiments.osram_meld_block2_auto_20261014.methods import build_config
    from gcnet_missing_m3.train_gcnet import TrainConfig
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / 'experiments/osram_meld_block2_auto_20261014/BASE_CONFIG.json').read_text())
    model = worker().build_smoke_model(TrainConfig(**build_config(base, 'B1', 'unused')), (4, 4, 4), torch.device('cpu'))
    assert model.training and any(p.requires_grad for p in model.parameters())
    assert all(p.requires_grad for p in model.osram.meaningful_block.parameters())
    assert model.teacher is None or all(not p.requires_grad for p in model.teacher.parameters())
