"""First-stage A0--A3 from scratch; stage-two GNN is never enabled here."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import protocol as p
from . import runtime as r


def atomic_arrays(path, arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def persist_selected(output, model, *, epoch, identity, best, evaluated):
    """Earliest strict W-F1 best, with its SAME-call predictions and mask.

    BEST is deliberately weights-only; only LAST contains an optimizer and RNG
    and is eligible for exact continuation. This avoids eight optimizer copies.
    """
    if set(evaluated) != set(p.RATES):
        raise ValueError('Every epoch must evaluate all eight rates')
    for metrics, arrays in evaluated.values():
        if not np.isfinite(arrays['prediction']).all():
            raise ValueError('Nonfinite selected predictions')
        if hashlib.sha256(arrays['availability'].tobytes()).hexdigest() != metrics['mask_sha256']:
            raise ValueError('Metrics are not bound to their exact same-call availability')
        p.is_improvement(metrics, None)
    improved = [rate for rate in p.RATES if p.is_improvement(evaluated[rate][0],
                 best.get(rate, {}).get('metrics'))]
    if not improved:
        return []
    model_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    for rate in improved:
        metrics, arrays = evaluated[rate]
        suffix = rate.replace('.', 'p')
        prediction_name = f'predictions_miss_{suffix}.npz'
        checkpoint_name = f'best_miss_{suffix}.pt'
        atomic_arrays(Path(output) / prediction_name, arrays)
        row = dict(epoch=epoch, checkpoint=checkpoint_name, predictions=prediction_name,
            prediction_sha256=r.sha(Path(output) / prediction_name), metrics=copy.deepcopy(metrics),
            same_evaluation_call=True, criterion='TEST W-F1; strict improvement; earliest ties')
        state = dict(version=1, kind='weights-only', model=model_state, epoch=epoch,
            identity=identity, config=identity['config'], dimensions=identity.get('dimensions'),
            selection=dict(row, rate=rate))
        r.atomic_save(Path(output) / checkpoint_name, state)
        row['checkpoint_sha256'] = r.sha(Path(output) / checkpoint_name)
        best[rate] = row
    r.save_json(Path(output) / 'best_by_rate.json', best)
    return improved


def validate_selected(output, best):
    if set(best) != set(p.RATES):
        raise ValueError('Checkpoint ledger does not contain all eight selected rates')
    for row in best.values():
        for name, digest in (('checkpoint', 'checkpoint_sha256'), ('predictions', 'prediction_sha256')):
            path = Path(output) / row[name]
            if r.sha(path) != row[digest]:
                raise ValueError('Selected checkpoint/prediction changed: ' + str(path))


def checkpoint_payload(model, optimizer, *, epoch, identity, history, best):
    return dict(version=1, kind='full-resume', epoch=epoch, identity=identity,
        config=identity['config'], dimensions=identity.get('dimensions'), model=model.state_dict(),
        optimizer=optimizer.state_dict(), rng=r.rng_state(), history=history, best_by_rate=best)


def restore(model, optimizer, state, identity):
    if state.get('kind') != 'full-resume':
        raise ValueError('Exact continuation requires a full-resume LAST checkpoint, not BEST weights')
    if state.get('identity') != identity:
        raise ValueError('Resume identity differs from immutable configuration/source/masks')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    r.restore_rng(state['rng'])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--variant', choices=tuple(p.VARIANTS), required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--gpu-uuid', required=True)
    result.add_argument('--source-commit', required=True)
    result.add_argument('--smoke', action='store_true')
    result.add_argument('--resume', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.smoke and args.resume:
        raise ValueError('Smoke is not a resumable training run')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if args.output.exists() and not args.resume:
        raise FileExistsError('A fresh independent output directory is required')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    summary_path, tick = args.output / 'SUMMARY.json', time.monotonic()
    if summary_path.exists() and json.loads(summary_path.read_text()).get('status') in ('complete', 'smoke-passed'):
        raise ValueError('A completed run may not be overwritten or resumed silently')
    summary = dict(status='preflight', variant=args.variant, dataset='CMUMOSEI', seed=66,
        phase='smoke' if args.smoke else 'train', label=p.LABEL, source_commit=args.source_commit,
        pid=os.getpid(), old_checkpoint_used_for_training=False, second_stage_gnn=False)
    r.save_json(summary_path, summary)
    try:
        torch.set_num_threads(1)
        bundle = r.setup(args.variant, args.output, args.gpu_uuid)
        model, cfg = bundle['model'], bundle['cfg']
        identity = dict(bundle['identity'], source_commit=args.source_commit)
        # Canonical JSON also normalizes tuple-valued dataclass fields for resume.
        identity = json.loads(json.dumps(identity))
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=cfg.learning_rate, name='backbone')],
                                     weight_decay=cfg.weight_decay)
        summary.update(identity=identity, config=identity['config'], environment=bundle['environment'],
            gpu=bundle['gpu'], rng_repair=bundle['rng_repair'], initial_backbone_sha256=bundle['initial_backbone_sha256'],
            total_parameters=sum(v.numel() for v in model.parameters()), status='running')
        r.save_json(args.output / 'config.json', identity['config'])
        r.save_json(args.output / 'IDENTITY.json', identity)
        # Tiny immutable reference, not another copy of the full feature bank.
        r.save_json(args.output / 'fixed_masks_reference.json', identity['mask_reference'])
        r.save_json(summary_path, summary)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity

        reset_training_stochasticity(cfg.seed, cfg.fold)
        torch.cuda.reset_peak_memory_stats()
        if args.smoke:
            before = r.state_digest(model)
            row = r.train_epoch(bundle, model, optimizer, 0, max_batches=3)
            if r.state_digest(model) == before:
                raise RuntimeError('Smoke made no original task parameter update')
            metrics, _ = r.evaluate(bundle, model, '0.3', max_batches=1)
            state = checkpoint_payload(model, optimizer, epoch=1, identity=identity, history=[row], best={})
            r.atomic_save(args.output / 'smoke.pt', state)
            restored = torch.load(args.output / 'smoke.pt', map_location='cpu', weights_only=False)
            after = r.state_digest(model)
            restore(model, optimizer, restored, identity)
            if r.state_digest(model) != after:
                raise RuntimeError('Full smoke model checkpoint changed on strict reload')
            summary.update(status='smoke-passed', smoke=row, smoke_test=metrics,
                           optimizer_rng_strict_restore=True, epoch=0)
        else:
            start, history, best = 0, [], {}
            if args.resume:
                state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
                restore(model, optimizer, state, identity)
                start, history, best = state['epoch'], state['history'], state['best_by_rate']
                validate_selected(args.output, best)
            for epoch in range(start, cfg.epochs):
                row = r.train_epoch(bundle, model, optimizer, epoch)
                evaluated = {rate: r.evaluate(bundle, model, rate) for rate in p.RATES}
                row['test'] = {rate: result[0] for rate, result in evaluated.items()}
                history.append(row)
                # One model trajectory; each rate selects its own strict TEST-best epoch.
                improved = persist_selected(args.output, model, epoch=epoch + 1, identity=identity,
                                            best=best, evaluated=evaluated)
                r.atomic_save(args.output / 'last.pt', checkpoint_payload(model, optimizer,
                    epoch=epoch + 1, identity=identity, history=history, best=best))
                r.save_json(args.output / 'history.json', history)
                summary.update(epoch=epoch + 1, best_by_rate=best,
                    selected_epoch_by_rate={rate: result['epoch'] for rate, result in best.items()},
                    aggregate=p.aggregate({rate: result['metrics'] for rate, result in best.items()}),
                    seconds=time.monotonic() - tick, peak_memory_mib=torch.cuda.max_memory_allocated() / 2 ** 20)
                r.save_json(summary_path, summary)
                print(json.dumps(dict(event='epoch', variant=args.variant, epoch=epoch + 1,
                    task_MSE=row['task_MSE'], mask_trace_sha256=row['mask_trace_sha256'],
                    aggregate=summary['aggregate'], improved_rates=improved)), flush=True)
            validate_selected(args.output, best)
            if r.source_hashes() != identity['source_hashes']:
                raise RuntimeError('Numerical or runner source changed during training')
            summary.update(status='complete', evaluation={rate: row['metrics'] for rate, row in best.items()})
        summary.update(seconds=time.monotonic() - tick,
                       peak_memory_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                       peak_reserved_memory_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        r.save_json(summary_path, summary)


if __name__ == '__main__':
    main()
