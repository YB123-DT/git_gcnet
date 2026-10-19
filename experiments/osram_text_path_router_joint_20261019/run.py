"""Finite from-scratch Flat versus one-stage Text-source Router training."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAGS = ('full', 'router', 'uniform', 'candidate_B', 'candidate_G')


def wrap_model(backbone, mode, router=None):
    if mode == 'flat':
        if router is not None:
            raise ValueError('Flat control cannot contain a Router')
        backbone.osram.osram_source_tracking = False
        backbone.osram.osram_source_audit = False
        backbone.osram.osram_source_grad = False
        return backbone.requires_grad_(True).train()
    if mode != 'joint':
        raise ValueError('Expected flat or joint mode')
    from .joint import JointTextPathModel
    from experiments.osram_text_path_router_20261019.router import TextPathRouter
    if router is None:
        router = TextPathRouter(local_dim=backbone.osram.latent_dim,
            context_dim=backbone.osram.num_heads * backbone.osram.value_dim,
            detach_inputs=False).to(next(backbone.parameters()).device)
    return JointTextPathModel(backbone, router).requires_grad_(True).train()


def make_optimizer(model, cfg, mode):
    import torch
    if mode == 'joint':
        groups = [dict(params=model.backbone.parameters(), lr=cfg.learning_rate, name='backbone'),
                  dict(params=model.router.parameters(), lr=3e-4, name='router')]
    else:
        groups = [dict(params=model.parameters(), lr=cfg.learning_rate, name='backbone')]
    return torch.optim.Adam(groups, weight_decay=cfg.weight_decay)


def forward_result(model, view, mode):
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if mode == 'joint':
        return model(view)
    logits = h.forward(model, view)
    weights = logits.new_zeros(*logits.shape[:2], 3)
    weights[..., 0] = 1
    return dict(log_probs=logits.log_softmax(-1), candidates=logits.unsqueeze(-2).expand(*logits.shape[:2], 3, logits.shape[-1]),
                weights=weights, eligible=torch.zeros(logits.shape[:2], dtype=torch.bool, device=logits.device))


def optimize_view(model, optimizer, view, cfg, mode):
    import torch
    from torch.nn import functional as F
    optimizer.zero_grad(set_to_none=True)
    out = forward_result(model, view, mode)
    valid = view['umask'].T.bool()
    if not valid.any():
        raise ValueError('Training batch has no valid utterances')
    labels = view['labels'].T[valid].long()
    # Original classification CE for Flat; mixture probability NLL for Router.
    loss = (F.cross_entropy(out['candidates'][..., 0, :][valid], labels) if mode == 'flat'
            else F.nll_loss(out['log_probs'][valid], labels))
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite one-stage task loss')
    loss.backward()
    groups = {}
    for group in optimizer.param_groups:
        grads = [p.grad for p in group['params'] if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Nonfinite gradient in '+group['name'])
        groups[group['name']] = dict(connected=len(grads), nonzero=sum(bool(g.ne(0).any()) for g in grads))
    if not groups['backbone']['nonzero']:
        raise RuntimeError('Joint task loss did not reach the original backbone')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite global gradient norm')
    optimizer.step()
    return dict(task_NLL=float(loss.detach()), N=int(valid.sum()), eligible_N=int(out['eligible'].sum()),
                optimizer_steps=1, gradient_norm=float(norm), gradients=groups)


def train_epoch(bundle, model, optimizer, epoch, mode, max_batches=None):
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    from experiments.osram_tasr_20261018 import runner_helpers as h
    trainer, cfg, loader = bundle['trainer'], bundle['cfg'], bundle['train_loader']
    model.train()
    loader.sampler.set_epoch(epoch)
    schedules = trainer._schedules(cfg, 'train')
    cycle = BalancedBatchRateSchedule(rates=cfg.train_missing_rates)
    rows, digest, observed, slots = [], hashlib.sha256(), 0, 0
    for index, raw in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        data = trainer._move_batch(raw, next(model.parameters()).device)
        rate = cycle.rate_for(epoch, index)
        view = trainer._prepare_view(data, schedules[rate], epoch, bundle['dimensions'])
        rows.append(optimize_view(model, optimizer, view, cfg, mode))
        availability = view['availability'].detach().cpu().numpy()
        digest.update(json.dumps([str(v) for v in data[-1]], separators=(',', ':')).encode())
        digest.update(availability.tobytes())
        observed += int(availability.sum())
        slots += int(view['umask'].sum()) * 3
    if not rows:
        raise RuntimeError('No training batches')
    return dict(epoch=epoch+1, batches=len(rows), memory_scans=len(rows), optimizer_steps=len(rows),
        N=sum(row['N'] for row in rows), eligible_N=sum(row['eligible_N'] for row in rows),
        task_NLL=sum(row['task_NLL'] * row['N'] for row in rows) / sum(row['N'] for row in rows),
        gradient_norm_max=max(row['gradient_norm'] for row in rows),
        gradients={name:{key:max(row['gradients'][name][key] for row in rows) for key in ('connected', 'nonzero')}
                   for name in rows[0]['gradients']},
        mask_trace_sha256=digest.hexdigest(), actual_missing_fraction=1-observed/max(1, slots))


def summarize_arrays(arrays, classes):
    from experiments.osram_text_path_router_20261019.run import summarize_predictions
    logits = arrays['candidates']
    shifted = logits - logits.max(-1, keepdims=True)
    candidate_log = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    maximum = candidate_log.max(1)
    uniform = maximum + np.log(np.exp(candidate_log - maximum[:, None]).mean(1))
    arrays = dict(arrays, log_probs_uniform=np.where(arrays['eligible'][:, None], uniform, candidate_log[:, 0]))
    return summarize_predictions(arrays, classes)


def evaluate(bundle, model, mode, loader, *, saved=None, batches=None, max_batches=None):
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    model.eval()
    loader.sampler.set_epoch(0)
    pieces, cursor = {}, 0
    with h.isolated_rng(66+99193), torch.no_grad():
        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            data = bundle['trainer']._move_batch(raw, next(model.parameters()).device)
            meta = h.metadata(data)
            n = len(meta['labels'])
            if saved is not None:
                for key, value in meta.items():
                    np.testing.assert_array_equal(value, saved[key][cursor:cursor+n])
                availability = torch.zeros((*data[7].shape, 3), device=data[7].device)
                availability[data[7].bool()] = torch.as_tensor(saved['availability'][cursor:cursor+n],
                    device=availability.device, dtype=availability.dtype)
                availability = availability.transpose(0, 1)
            elif batches is not None:
                record = batches[index]
                for key, value in meta.items():
                    np.testing.assert_array_equal(value, record[key])
                availability = record['availability'].to(data[7].device)
            else:
                raise ValueError('An explicit fixed validation or original exact test mask is required')
            view = bundle['trainer']._prepare_view_from_primary_masks(data, availability, availability, bundle['dimensions'])
            result = forward_result(model, view, mode)
            valid = data[7].bool()
            values = dict(meta, availability=availability.transpose(0, 1)[valid].cpu().numpy())
            for name in ('candidates', 'weights', 'eligible'):
                values[name] = result[name].transpose(0, 1)[valid].cpu().numpy()
            values['log_probs_router'] = result['log_probs'].transpose(0, 1)[valid].cpu().numpy()
            if not np.isfinite(values['log_probs_router']).all():
                raise RuntimeError('Nonfinite prediction')
            for name, value in values.items():
                pieces.setdefault(name, []).append(value)
            cursor += n
    if not pieces:
        raise RuntimeError('No evaluation utterances')
    if saved is not None and max_batches is None and cursor != len(saved['labels']):
        raise RuntimeError('Saved exact-mask cohort was not fully replayed')
    arrays = {name: np.concatenate(parts) for name, parts in pieces.items()}
    metrics = summarize_arrays(arrays, bundle['case']['classes'])
    metrics['mask_sha256'] = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
    metrics['checks'] = dict(exact_identity_and_saved_availability=True, new_model_logits_not_compared_to_old=True)
    return metrics, arrays


def checkpoint_payload(model, optimizer, *, epoch, identity, history, best_key):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    return dict(version=1, model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer=copy.deepcopy(optimizer.state_dict()), rng=h.rng_state(), epoch=epoch,
        identity=copy.deepcopy(identity), history=copy.deepcopy(history), best_key=best_key,
        recovery='Complete from-scratch model, Adam, RNG, epoch, configuration, fixed-mask hashes')


def restore(model, optimizer, state, identity):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if state['identity'] != identity:
        raise ValueError('Resume identity differs from immutable run')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    h.restore_rng(state['rng'])


def seed_training_loader(bundle, seed=66):
    """Vary training order only; keep the accepted seed66 split/eval banks."""
    if seed not in (66, 67, 68):
        raise ValueError('Approved training seeds are 66, 67, 68')
    loader = bundle['train_loader']
    if seed != 66:
        from gcnet_modality_jepa.protocol import SeedBundle
        from gcnet_modality_jepa.train_gcnet import _build_protocol_loader
        metadata = loader.protocol_metadata
        loader = _build_protocol_loader(loader.dataset, loader.sampler.indices,
            bundle['cfg'].dataset, metadata['fold'], metadata['split'],
            loader.batch_size, loader.num_workers, SeedBundle(seed), metadata['split_hash'],
            metadata['evaluation_protocol'])
        bundle['train_loader'] = loader
    return dict(training_seed=seed, split_seed=66, evaluation_seed=66,
                train_loader=copy.deepcopy(loader.protocol_metadata))


def prepare_fresh(bundle, mode, seed=66):
    """Protected model is a data/protocol parity gate, never initialization."""
    import gc
    import torch
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    seed_protocol = seed_training_loader(bundle, seed)
    cfg = replace(bundle['cfg'], seed=seed, checkpoint_selection='validation', checkpoint_selection_metric='accuracy', evaluate_test=True)
    if (cfg.learning_rate != (1e-5 if cfg.dataset == 'MELD' else 1e-3)
            or cfg.epochs != (50 if cfg.dataset == 'MELD' else 100) or cfg.weight_decay != 1e-5):
        raise ValueError('Protected baseline LR/epoch/weight-decay differs from the approved one-stage protocol')
    device = next(bundle['model'].parameters()).device
    del bundle['model']
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Canonical factory resets missing_m3_model_init:fold seed before construction.
    backbone = _build_model(cfg, bundle['dimensions'], device).requires_grad_(True).train()
    model = wrap_model(backbone, mode)
    bundle.update(model=model, cfg=cfg, seed_protocol=seed_protocol)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--mode', choices=('flat', 'joint'), required=True)
    parser.add_argument('--dataset', choices=('MELD', 'IEMOCAPSix'), required=True)
    parser.add_argument('--seed', type=int, choices=(66, 67, 68), default=66)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.protocol import RATES
    from experiments.osram_tasr_20261018.run import build_validation_masks
    if args.resume and args.phase != 'train':
        raise ValueError('Resume is supported only for the same immutable training run')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Use an independent output directory')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick = args.output/'SUMMARY.json', time.monotonic()
    if args.resume and path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        raise ValueError('Completed run must not be silently repeated')
    summary = dict(status='preflight', dataset=args.dataset, mode=args.mode, phase=args.phase,
                   source_commit=args.source_commit, pid=os.getpid(), seed=args.seed)
    try:
        bundle = h.setup_case(args.dataset, args.gpu_uuid, args.output)
        # Dataset paths must be installed by setup_case before importing trainer.
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        protected_backbone_sha256 = h.backbone_digest(bundle['model'])
        model = prepare_fresh(bundle, args.mode, seed=args.seed)
        backbone = model.backbone if args.mode == 'joint' else model
        initial_backbone_sha256 = h.backbone_digest(backbone)
        if protected_backbone_sha256 == initial_backbone_sha256:
            raise RuntimeError('Fresh model unexpectedly equals the protected trained checkpoint')
        optimizer = make_optimizer(model, bundle['cfg'], args.mode)
        masks = build_validation_masks(bundle)
        torch.save(masks, args.output/'validation_masks.pt')
        mask_hashes = {rate:hashlib.sha256(b''.join(row['availability'].numpy().tobytes() for row in rows)).hexdigest()
                       for rate, rows in masks.items()}
        sources = ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/tasr.py', 'gcnet_missing_m3/model.py',
                   'gcnet_missing_m3/train_gcnet.py', 'gcnet_modality_jepa/gcnet_legacy_random_mask.py',
                   'gcnet_modality_jepa/protocol.py', 'gcnet_modality_jepa/train_gcnet.py',
                   'experiments/osram_text_path_router_20261019/router.py',
                   'experiments/osram_text_path_router_joint_20261019/joint.py',
                   'experiments/osram_text_path_router_joint_20261019/run.py')
        identity = dict(source_commit=args.source_commit, source_hashes={p:h.sha(ROOT/p) for p in sources},
            mode=args.mode, config=asdict(bundle['cfg']), split=bundle['split'], validation_mask_hashes=mask_hashes,
            test_masks={rate:ref['mask_sha256'] for rate, ref in bundle['references'].items()},
            seed_protocol=bundle['seed_protocol'],
            original_reference_case=bundle['case'], initialization='canonical-fresh-factory-no-checkpoint-weights',
            router_learning_rate=3e-4 if args.mode == 'joint' else None, selection='validation-eight-rate-ACC-then-lower-CE')
        summary.update(identity=identity, environment=bundle['environment'], gpu=bundle['gpu'],
            anchor_parity_gate=bundle['anchor_parity'], total_parameters=sum(p.numel() for p in model.parameters()),
            router_parameters=sum(p.numel() for p in model.router.parameters()) if args.mode == 'joint' else 0,
            status='running', validation_mask_hashes=mask_hashes, initial_backbone_sha256=initial_backbone_sha256,
            protected_backbone_sha256=protected_backbone_sha256, protected_weights_excluded_from_initialization=True)
        h.write_json(args.output/'config.json', identity['config'])
        h.write_json(path, summary)
        reset_training_stochasticity(args.seed, bundle['case']['fold'])
        torch.cuda.reset_peak_memory_stats()
        if args.phase == 'smoke':
            backbone = model.backbone if args.mode == 'joint' else model
            before = h.backbone_digest(backbone)
            router_before = h.backbone_digest(model.router) if args.mode == 'joint' else None
            row = train_epoch(bundle, model, optimizer, 0, args.mode, max_batches=3)
            if h.backbone_digest(backbone) == before:
                raise RuntimeError('Smoke did not update the original backbone')
            if args.mode == 'joint' and h.backbone_digest(model.router) == router_before:
                raise RuntimeError('Smoke did not update the Router')
            val, _ = evaluate(bundle, model, args.mode, bundle['validation_loader'], batches=masks['0.3'], max_batches=1)
            state = checkpoint_payload(model, optimizer, epoch=1, identity=identity, history=[row], best_key=None)
            h.atomic_torch_save(args.output/'smoke.pt', state)
            stored = torch.load(args.output/'smoke.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, stored, identity)
            if not all(torch.equal(value.cpu(), stored['model'][name]) for name, value in model.state_dict().items()):
                raise RuntimeError('Full model reload failed')
            summary.update(smoke=row, smoke_validation=val, smoke_only=True, backbone_changed=True,
                           router_changed=args.mode == 'joint', full_checkpoint_reloaded=True)
        else:
            start, history, best_key = 0, [], None
            if args.resume:
                state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
                restore(model, optimizer, state, identity)
                start, history, best_key = state['epoch'], state['history'], state['best_key']
            for epoch in range(start, bundle['cfg'].epochs):
                row = train_epoch(bundle, model, optimizer, epoch, args.mode)
                validation = {rate:evaluate(bundle, model, args.mode, bundle['validation_loader'], batches=masks[rate])[0]
                              for rate in RATES}
                key = (sum(v['router']['ACC'] for v in validation.values())/8,
                       -sum(v['router']['CE'] for v in validation.values())/8)
                improved = best_key is None or tuple(key) > tuple(best_key)
                if improved:
                    best_key = key
                row.update(validation=validation, validation_key=key)
                history.append(row)
                state = checkpoint_payload(model, optimizer, epoch=epoch+1, identity=identity, history=history, best_key=best_key)
                if improved:
                    h.atomic_torch_save(args.output/'best.pt', state)
                h.atomic_torch_save(args.output/'last.pt', state)
                h.write_json(args.output/'history.json', history)
                summary.update(epoch=epoch+1, best_validation_key=best_key, seconds=time.monotonic()-tick,
                               peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
                h.write_json(path, summary)
                print(json.dumps(dict(event='epoch', epoch=epoch+1, loss=row['task_NLL'], eligible=row['eligible_N'],
                                      validation=key, best=best_key)), flush=True)
            state = torch.load(args.output/'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            summary.update(selected_epoch=state['epoch'], evaluation={})
            output = args.output/'evaluation'
            output.mkdir(exist_ok=True)
            for rate in RATES:
                metrics, arrays = evaluate(bundle, model, args.mode, bundle['test_loader'], saved=bundle['selected'][rate])
                artifact = output/('random_'+rate.replace('.', 'p')+'.npz')
                np.savez_compressed(artifact, **arrays)
                metrics['output_sha256'] = h.sha(artifact)
                summary['evaluation'][rate] = metrics
                h.write_json(path, summary)
            summary['aggregate'] = {tag:h.aggregate_rates({r:v[tag] for r,v in summary['evaluation'].items()}) for tag in TAGS}
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT/name) != digest:
                raise RuntimeError('Immutable source changed during training')
        summary.update(status='complete', seconds=time.monotonic()-tick,
                       final_backbone_sha256=h.backbone_digest(backbone),
                       peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        if args.phase == 'train':
            from .report import write_report
            write_report(args.output, summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
