"""Small real-model backward and training-loop checks for four transfers."""
import torch

from test_core20_integration import config, batch
from gcnet_missing_m3.core20 import attach, TRANSFER_METHODS
from gcnet_missing_m3.train_gcnet import _task_loss, train_epoch, _schedules
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model


def test_transfers_real_backward_and_eval():
    torch.set_num_threads(1)
    for method in TRANSFER_METHODS:
        cfg = config(method)
        model = _build_model(cfg, (3, 4, 5))
        rng = torch.get_rng_state().clone()
        attach(model, cfg)
        assert torch.equal(rng, torch.get_rng_state())
        view = batch()
        args = ([view['incomplete']], view['availability'], view['qmask'], view['umask'], view['lengths'])
        criterion = lambda p: _task_loss(cfg.dataset, p, view['labels'], view['umask'], cfg.mosi_task_mode)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
        # Original Flat adapter is zero-init: storage gradients begin after its
        # first update, not on the initial Local-only numerical working point.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            if method == 'R18':
                model.core20.complete_features = view['complete']
            logits = model(*args)[0]
            loss = model.core20.loss(model, cfg, view, logits, criterion(logits), criterion)
            loss.backward()
            optimizer.step()
        new = [p for n, p in model.named_parameters() if 'core20' in n and p.requires_grad]
        if method != 'R12':
            assert new and any(p.grad is not None and p.grad.abs().sum() > 0 for p in new), method
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.no_grad():
            pred, hidden, _, _ = model(*args)
        assert torch.isfinite(pred).all()
        assert torch.count_nonzero(hidden[3, 1]) == 0


def test_r18_real_training_loop_passes_privileged_train_features():
    torch.set_num_threads(1)
    cfg = config('R18')
    model = _build_model(cfg, (3, 4, 5))
    attach(model, cfg)
    view = batch()
    blocks = list(view['complete'].split((3, 4, 5), -1))
    raw = blocks + blocks + [view['qmask'], view['umask'], view['labels'], ['train-a', 'train-b']]
    metrics = train_epoch(model, [raw], torch.optim.Adam(model.parameters()), cfg,
                          _schedules(cfg, 'train'), 0, (3, 4, 5), torch.device('cpu'))
    assert metrics['optimizer_steps'] == 1
    assert model.core20.complete_features is None
