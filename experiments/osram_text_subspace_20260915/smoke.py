"""One Stage1 epoch (two MOSI batches), then one Stage2 batch; never test/train a grid."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from gcnet_missing_m3.text_subspace import TextSubspacePretrainer, text_jepa_loss, state_sha256
from experiments.osram_supervised_teacher_20260914.smoke import build
from experiments.osram_text_subspace_20260915.run import (
    ROOT, TEACHER_ROOT, tr, configuration, loaders, stage1_data,
    fit_stage1, stage2_config, gradient_norm,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'smoke_seed66')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    tr.set_random_seed(66)
    data, teacher_cfg, dims, teacher_hash, provenance = stage1_data(66, args.device)
    pretrainer = TextSubspacePretrainer()
    stage1 = fit_stage1(pretrainer, data, args.output/'stage1', teacher_hash, epochs=1,
        learning_rate=teacher_cfg.learning_rate, weight_decay=teacher_cfg.weight_decay,
        beta=1., device=args.device, smoke=True, provenance=provenance)
    del pretrainer, data
    base, _, _ = configuration(66)
    base = replace(base, device=args.device)
    teacher_path = TEACHER_ROOT/'teacher'/'seed_66'/'teacher_projectors.pt'
    cfg = stage2_config(base, teacher_path, args.output/'stage1'/'subspace.pt', 'predictable-subspace')
    reference_cfg = replace(cfg, target_space='all-modalities', text_subspace_checkpoint=None)
    tr.set_random_seed(66)
    reference = build(reference_cfg, dims)
    reference_weights = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    del reference
    tr.set_random_seed(66)
    model = build(cfg, dims)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng:
        assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    for key, value in reference_weights.items():
        assert torch.equal(value, model.state_dict()[key].cpu()), key
    baseline_count = sum(v.numel() for v in reference_weights.values())
    del reference_weights
    train, _, _unused_test, *loaded_dims = loaders(cfg)
    assert dims == loaded_dims
    raw = next(iter(train[0]))
    view = tr._prepare_view(tr._move_batch(raw, torch.device(args.device)),
                            tr._build_schedule(cfg, 'train', .5), 0, tuple(dims))
    teacher_before = model.teacher_integrity()
    r_before = model.text_subspace_integrity()
    model.train()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    logits, _, _, prediction = model([view['incomplete']], view['availability'],
        view['qmask'], view['umask'], view['lengths'], predict_missing=True)
    assert prediction.reg_predictions.shape[-1] == prediction.cl_predictions.shape[-1] == 256
    targets = model.encode_teacher_targets([view['complete']])
    jepa = text_jepa_loss(prediction, targets, model.text_subspace, cfg.temperature)
    emotion = tr._task_loss(cfg.dataset, logits, view['labels'], view['umask'], cfg.mosi_task_mode,
                            cfg.task_regression_loss, cfg.task_smooth_l1_beta)
    reg_gradient, = torch.autograd.grad(jepa.regression, prediction.reg_predictions, retain_graph=True)
    cl_gradient, = torch.autograd.grad(jepa.contrastive, prediction.cl_predictions, retain_graph=True)
    for grad in (reg_gradient, cl_gradient):
        assert grad[..., 1, :].abs().sum() > 0
        assert torch.count_nonzero(grad[..., (0, 2), :]) == 0
    grads = dict(reg_output=gradient_norm([reg_gradient]), cl_output=gradient_norm([cl_gradient]))
    for name, module in (('mmoe', model.missing_predictor), ('student_encoder', model.observed_set),
                         ('osram', model.osram)):
        grads[name] = gradient_norm(torch.autograd.grad(jepa.total, tuple(module.parameters()),
                                                       retain_graph=True, allow_unused=True))
    assert all(v > 0 for v in grads.values())
    total = emotion + cfg.jepa_weight * jepa.total
    total.backward()
    assert torch.isfinite(total) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is None for p in model.text_subspace.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
    optimizer.step()
    model.update_teacher(cfg.ema_tau)
    assert model.teacher_integrity() == teacher_before == teacher_hash
    assert model.text_subspace_integrity() == r_before
    model.eval()
    with torch.no_grad(), patch.object(model.teacher, 'forward', side_effect=AssertionError('Teacher inference')), \
         patch.object(model, 'encode_teacher_targets', side_effect=AssertionError('target inference')), \
         patch.object(model.missing_predictor, 'forward', side_effect=AssertionError('MMoE inference')), \
         patch.object(model.text_subspace, 'forward', side_effect=AssertionError('R inference')):
        result = model([view['incomplete']], view['availability'], view['qmask'], view['umask'], view['lengths'])
        assert torch.isfinite(result[0]).all() and result[3] is None
    report = dict(smoke_only=True, full_training_started=False, seed=66,
        stage1=stage1, teacher_frozen_hash=teacher_hash, teacher_unchanged=True, R_unchanged=True,
        shared_student_initialization_exact=True, shared_rng_exact=True,
        stage2=dict(optimizer_updates=1, emotion_loss=float(emotion.detach()),
                    jepa_loss=float(jepa.total.detach()), regression=float(jepa.regression.detach()),
                    contrastive=float(jepa.contrastive.detach()), total_loss=float(total.detach()),
                    missing_text_target_count=jepa.target_count, gradient_l2=grads,
                    reg_cl_output_dimension=256, target_subspace_dimension=32,
                    added_frozen_parameters=sum(p.numel() for p in model.text_subspace.parameters()),
                    added_trainable_parameters=0, inference_auxiliary_calls=0,
                    reference_state_elements=baseline_count,
                    teacher_gradient_none=True, R_gradient_none=True,
                    checkpoint_selection=cfg.checkpoint_selection),
        test_batches_consumed=0)
    (args.output/'SMOKE.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'stage1'}, indent=2))


if __name__ == '__main__':
    main()
