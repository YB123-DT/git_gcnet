"""Explicit Stage1/Stage2 commands. No chained or multi-seed launch."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from torch.nn import functional as F

from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import EMATeacherProjectors, ModalityProjector, MODALITIES
from gcnet_missing_m3.pretrained_teacher import read_source, load_pretrained_teacher, state_sha256
from gcnet_missing_m3.text_subspace import (
    TextSubspacePretrainer, PATTERNS, representation_stats, save_subspace,
)
from experiments.osram_causal_nojepa_20260910.run import configuration, runner
from experiments.osram_supervised_teacher_20260914.run import ROOT as TEACHER_ROOT
from experiments.osram_supervised_teacher_20260914.information_audit import sentiment_metrics

ROOT = Path('/data2/yb/remote_experiments/osram_text_subspace_20260915')
FEATURE_NAMES = ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')


def gradient_norm(grads):
    return sum(float(g.detach().double().square().sum()) for g in grads if g is not None) ** .5


def stage2_config(base, teacher, subspace, target_space):
    if target_space not in {'full-text', 'predictable-subspace'}:
        raise ValueError('Stage2 requires full-text or predictable-subspace')
    return replace(base, training_objective='joint', train_rate_mode='cyclic',
                   fixed_missing_rate=None, initial_backbone_checkpoint=None,
                   teacher_mode='pretrained-frozen', teacher_checkpoint=str(teacher),
                   target_space=target_space,
                   text_subspace_checkpoint=str(subspace) if target_space == 'predictable-subspace' else None,
                   checkpoint_selection='test-oracle-per-rate', evaluate_test=True)


def loaders(cfg):
    roots = [str(runner.FEATURES / n) for n in FEATURE_NAMES]
    return tr.get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset=cfg.dataset, batch_size=cfg.batch_size, num_workers=0,
        seed=cfg.seed, evaluation_protocol=cfg.evaluation_protocol,
        validation_fraction=cfg.validation_fraction)


@torch.no_grad()
def extract_teacher(teacher, loader, cfg, dims, split):
    if split not in {'train', 'validation'}:
        raise ValueError('Stage1 supports train/validation only')
    batches, ids = [], []
    schedule = tr._build_schedule(cfg, split, 0.)
    for raw in loader:
        view = tr._prepare_view(tr._move_batch(raw, torch.device(cfg.device)), schedule, 0, tuple(dims))
        valid = view['umask'].T.bool()
        features = view['complete'][valid]
        blocks = features.split(tuple(dims), dim=-1)
        batch = {m: teacher[m](x).detach().cpu() for m, x in zip(MODALITIES, blocks)}
        batch['labels'] = view['labels'].T[valid].detach().cpu()
        batches.append(batch)
        ids.extend(str(x) for x in view['conversation_ids'])
    return batches, ids


def stage1_data(seed, device):
    path = TEACHER_ROOT / 'teacher' / f'seed_{seed}' / 'teacher_projectors.pt'
    cp, _, teacher_hash = read_source(path)
    cfg = replace(tr.TrainConfig(**cp['config']), device=device)
    if cfg.dataset != 'CMUMOSI' or cfg.seed != seed or cfg.latent_dim != 256:
        raise ValueError('Stage1 requires matching MOSI seed and 256d Teacher')
    train, validation, _unused_test_loader, *dims = loaders(cfg)
    # Construct only projector modules: neither Teacher memory nor classifier exists here.
    bank = nn.ModuleDict({m: ModalityProjector(d, cfg.latent_dim, cfg.projector_dropout)
                          for m, d in zip(MODALITIES, dims)})
    teacher = EMATeacherProjectors(bank).to(device)
    provenance = load_pretrained_teacher(teacher, path)
    before = state_sha256(teacher.state_dict())
    data, split_ids = {}, {}
    for split, loader in (('train', train[cfg.fold - 1]), ('validation', validation[cfg.fold - 1])):
        data[split], split_ids[split] = extract_teacher(teacher, loader, cfg, dims, split)
    assert not set(split_ids['train']).intersection(split_ids['validation'])
    assert before == teacher_hash == state_sha256(teacher.state_dict())
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    provenance.update(teacher_unchanged=True, split_conversation_ids=split_ids,
                      teacher_memory_calls=0, test_batches_consumed=0)
    return data, cfg, dims, teacher_hash, provenance


@torch.no_grad()
def evaluate_subspace(model, batches, beta, device):
    all_rows = {k: torch.cat([b[k] for b in batches]).to(device) for k in (*MODALITIES, 'labels')}
    model.eval()
    losses = model.loss(all_rows, all_rows['labels'], beta)
    target = model.projector(all_rows['text'])
    scores = model.sentiment(target).squeeze(-1)
    y = all_rows['labels'].cpu().numpy()
    centered_target = target - target.mean(0)
    results = dict(losses={k: float(v) for k, v in losses.items()}, target=representation_stats(target),
                   sentiment=sentiment_metrics(y, scores.cpu().numpy()), patterns={})
    generator = torch.Generator(device=device).manual_seed(7723)
    permutation = torch.randperm(target.shape[0], generator=generator, device=device)
    for p in PATTERNS:
        prediction = model.predict(all_rows['audio'], all_rows['visual'], p)
        cosine = F.cosine_similarity(prediction, target)
        centered_prediction = prediction - prediction.mean(0)
        similarities = F.normalize(centered_prediction, dim=-1) @ F.normalize(centered_target, dim=-1).T
        results['patterns'][p] = dict(
            smooth_l1=float(F.smooth_l1_loss(prediction, target)),
            raw_cosine=float(cosine.mean()),
            centered_cosine=float(F.cosine_similarity(centered_prediction, centered_target).mean()),
            real_vs_shuffle_cosine_gap=float((cosine - F.cosine_similarity(prediction, target[permutation])).mean()),
            centered_retrieval=float((similarities.argmax(-1) == torch.arange(len(target), device=device)).float().mean()),
            retrieval_chance=1. / len(target), prediction=representation_stats(prediction),
            prediction_teacher_std_ratio=float(prediction.std(0, unbiased=False).mean() /
                                                target.std(0, unbiased=False).mean().clamp_min(1e-12)),
            sentiment=sentiment_metrics(y, model.sentiment(prediction).squeeze(-1).cpu().numpy()))
    return results


def fit_stage1(model, data, output, teacher_hash, *, epochs, learning_rate,
               weight_decay, beta, device, smoke=False, provenance=None):
    if set(data) != {'train', 'validation'} or any(not v for v in data.values()):
        raise ValueError('Stage1 requires nonempty train/validation only')
    if epochs < 1:
        raise ValueError('epochs must be positive')
    if smoke and epochs != 1:
        raise ValueError('Smoke is limited to one epoch and at most two optimizer updates')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    config = dict(epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                  beta=beta, subspace_dim=32, hidden_dim=model.hidden_dim,
                  smoke_only=smoke, pattern_aggregation='mean(A,V,AV)',
                  selection_split='validation', test_used=False,
                  provenance=provenance or {})
    history, first_gradients, steps = [], None, 0
    best = float('inf')
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(data['train'])).tolist()
        if smoke:
            indices = indices[:2]
        for index in indices:
            batch = {k: v.to(device) for k, v in data['train'][index].items()}
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(batch, batch['labels'], beta)
            if first_gradients is None:
                first_gradients = {}
                for loss_name, module in (('sentiment', model.projector), ('predictability', model.projector)):
                    first_gradients[f'R_from_{loss_name}'] = gradient_norm(torch.autograd.grad(
                        losses[loss_name], tuple(module.parameters()), retain_graph=True))
                first_gradients['Q_from_predictability'] = gradient_norm(torch.autograd.grad(
                    losses['predictability'], tuple(model.predictor.parameters()), retain_graph=True))
                assert all(v > 0 for v in first_gradients.values())
            losses['total'].backward()
            if not torch.isfinite(losses['total']) or any(
                    not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('Nonfinite Stage1 loss/gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            steps += 1
        # Select using validation only. Test has no data/metric/code path in this function.
        metrics = {split: evaluate_subspace(model, rows, beta, device) for split, rows in data.items()}
        score = metrics['validation']['losses']['total']
        history.append(dict(epoch=epoch, metrics=metrics))
        if score < best:
            best = score
            save_subspace(output/'subspace.pt', model, teacher_hash, epoch, score, config)
        print(json.dumps(dict(stage1_epoch=epoch, validation_loss=score, smoke_only=smoke)), flush=True)
    cp = torch.load(output/'subspace.pt', map_location=device, weights_only=False)
    model.load_state_dict(cp['stage1_model'], strict=True)
    selected = {split: evaluate_subspace(model, rows, beta, device) for split, rows in data.items()}
    report = dict(selection_split='validation', selected_epoch=cp['epoch'],
                  selected_metrics=selected, first_step_gradient_l2=first_gradients,
                  parameters={name: sum(p.numel() for p in module.parameters()) for name, module in
                              (('R', model.projector), ('Q', model.predictor), ('C', model.sentiment))},
                  optimizer_steps=steps, elapsed_seconds=time.monotonic()-start,
                  config=config, teacher_projector_sha256=teacher_hash,
                  smoke_only=smoke, full_training_started=not smoke,
                  test_batches_consumed=0)
    (output/'STAGE1.json').write_text(json.dumps(report, indent=2))
    (output/'history.json').write_text(json.dumps(history, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('stage1', 'stage2'))
    parser.add_argument('--seed', type=int, required=True, choices=runner.SEEDS)
    parser.add_argument('--target-space', choices=('full-text', 'predictable-subspace'), default='predictable-subspace')
    parser.add_argument('--subspace-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    tr.set_random_seed(args.seed)
    if args.stage == 'stage1':
        data, cfg, _, fingerprint, provenance = stage1_data(args.seed, args.device)
        model = TextSubspacePretrainer()
        fit_stage1(model, data, args.output, fingerprint, epochs=100,
                   learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay, beta=1.,
                   device=args.device, provenance=provenance)
    else:
        if args.target_space == 'predictable-subspace':
            if args.subspace_checkpoint is None:
                parser.error('predictable-subspace requires --subspace-checkpoint')
            cp = torch.load(args.subspace_checkpoint, map_location='cpu', weights_only=False)
            if cp.get('config', {}).get('smoke_only', False):
                raise ValueError('Smoke subspace cannot be used for full Stage2 experiments')
        base, _, _ = configuration(args.seed)
        cfg = replace(stage2_config(base, TEACHER_ROOT/'teacher'/f'seed_{args.seed}'/'teacher_projectors.pt',
                                    args.subspace_checkpoint, args.target_space), device=args.device)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'PROTOCOL.json').write_text(json.dumps(dict(
            label='INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT',
            checkpoint_selection='per-seed per-rate Test weighted-F1 maximum', config=asdict(cfg)), indent=2))
        tr.run_experiment(cfg, *[str(runner.FEATURES/n) for n in FEATURE_NAMES], output_dir=str(args.output))


if __name__ == '__main__':
    main()
