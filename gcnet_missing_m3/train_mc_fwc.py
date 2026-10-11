"""Validation-selected MC-FWC experiments; original trainer remains unchanged.

One shared checkpoint is selected by mean eight-rate validation W-F1. Test
predictions are produced only after training. MC-FWC changes no inference state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import train_gcnet as old
from .mixed_rate import BalancedBatchRateSchedule, MISSING_RATES
from .training_resume import _atomic, _cpu

PATTERNS = {'A': (1, 0, 0), 'T': (0, 1, 0), 'V': (0, 0, 1),
            'AT': (1, 1, 0), 'AV': (1, 0, 1), 'TV': (0, 1, 1)}


def baseline_config(seed=66, device='cpu', epochs=150):
    return old.TrainConfig(dataset='CMUMOSI', fold=1, seed=seed, device=device,
        epochs=epochs, backbone_type='osram', training_objective='emotion-only',
        checkpoint_selection='validation', train_rate_mode='cyclic',
        osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64,
        osram_value_dim=64, osram_write_step=.6, osram_bidirectional=False,
        osram_readout_fusion='flat', disable_unused_aux_modules=True)


def learning_rate(epoch):
    return .001 if epoch < 100 else .0001


def batch_seed(seed, epoch, batch, stream):
    payload = json.dumps([int(seed), int(epoch), int(batch), stream]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**63 - 1)


def selection_score(metrics):
    if set(metrics) != {str(r) for r in MISSING_RATES}:
        raise ValueError('Selection requires all eight validation rates')
    result = sum(float(metrics[str(r)]['weighted_f1']) for r in MISSING_RATES) / 8
    if not math.isfinite(result):
        raise ValueError('Nonfinite validation score')
    return result


def write_json(path, value):
    _atomic(Path(path), lambda stream: stream.write(
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()))


def save_training(path, model, optimizer, identity, next_epoch, history, best):
    state = dict(schema_version=1, identity=identity, next_epoch=next_epoch,
        model=_cpu(model.state_dict()), optimizer=_cpu(optimizer.state_dict()),
        history=history, best=_cpu(best), scheduler=None, scaler=None,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None))
    _atomic(Path(path), lambda stream: torch.save(state, stream))


def restore_training(path, model, optimizer, identity):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['identity'] != identity or state['schema_version'] != 1:
        raise ValueError('Recovery identity/config/source/data mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state


def replace_availability(view, availability, dimensions):
    valid = view['umask'].T.bool()
    if availability.shape != (*valid.shape, 3):
        raise ValueError('Availability shape mismatch')
    if not bool(((availability == 0) | (availability == 1)).all()):
        raise ValueError('Availability must be binary')
    availability = torch.where(valid[..., None], availability, torch.zeros_like(availability))
    if bool((valid & ~availability.bool().any(-1)).any()):
        raise ValueError('Empty observed set')
    expanded = torch.repeat_interleave(availability.bool(),
        torch.tensor(dimensions, device=availability.device), dim=-1)
    return dict(view, availability=availability, incomplete=torch.where(
        expanded, view['complete'], torch.zeros_like(view['complete'])))


def build_model(config, dimensions):
    shape = old._resolve_task_contract(config.dataset, config.mosi_task_mode)
    parameters = inspect.signature(old.MissingM3GraphModel.__init__).parameters
    kwargs = {k: v for k, v in asdict(config).items() if k in parameters}
    kwargs.update(adim=dimensions[0], tdim=dimensions[1], vdim=dimensions[2],
        D_e=config.hidden, graph_hidden_size=config.hidden // 2,
        n_speakers=int(shape['num_speakers']), n_classes=int(shape['num_classes']),
        time_attn=config.time_attention, no_cuda=torch.device(config.device).type != 'cuda')
    return old.MissingM3GraphModel(**kwargs).to(config.device)


def prediction_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.size == 0:
        return {'n': 0, 'n_nonzero': 0, 'weighted_f1': None, 'accuracy': None}
    selected = labels != 0
    result = old._metrics('CMUMOSI', labels, predictions) if selected.any() else {
        'weighted_f1': None, 'accuracy': None}
    negative = labels < 0
    result.update(n=int(labels.size), n_nonzero=int(selected.sum()),
        mse=float(np.mean((predictions-labels)**2)),
        positive_ratio=float(np.mean(predictions[selected] > 0)) if selected.any() else None,
        negative_recall=float(np.mean(predictions[negative] <= 0)) if negative.any() else None,
        prediction_mean=float(predictions.mean()), prediction_std=float(predictions.std()),
        prediction_quantiles=np.quantile(predictions, [0, .05, .25, .5, .75, .95, 1]).tolist())
    return result


@torch.no_grad()
def evaluate(model, loader, schedules, dimensions, device, *, family='random', collect=False,
             utility_options=None):
    model.eval()
    metrics, records = {}, {}
    conditions = list(MISSING_RATES) if family == 'random' else list(PATTERNS)
    for condition in conditions:
        rows = []
        utility_rows, omit_forwards = [], 0
        for batch_index, raw in enumerate(loader):
            data = old._move_batch(raw, device)
            rate = condition if family == 'random' else 0.
            view = old._prepare_view(data, schedules[rate], epoch=0, dimensions=dimensions)
            onset = {}
            if family != 'random':
                av = view['availability'].clone()
                pattern = av.new_tensor(PATTERNS[condition])
                for b, length in enumerate(view['lengths']):
                    start = 0 if family == 'persistent' else max(1, length // 2)
                    av[start:length, b] = pattern
                    onset[b] = start
                view = replace_availability(view, av, dimensions)
            logits = model([view['incomplete']], view['availability'], view['qmask'],
                view['umask'], view['lengths'], predict_missing=False)[0].squeeze(-1)
            if utility_options is not None:
                from .fwc_sampling import sample_pairs
                from .fwc_evaluation import evaluate_write_utility
                pairs = sample_pairs(view['availability'], view['umask'], view['conversation_ids'],
                    seed=batch_seed(utility_options['seed'], 0, batch_index, f'eval:{family}:{condition}'),
                    horizon=utility_options['horizon'],
                    pairs_per_conversation=utility_options['pairs_per_conversation'],
                    strategy=utility_options['pair_strategy'], regimes=[family]*len(view['lengths']))
                utility = evaluate_write_utility(model, view, pairs, 'CMUMOSI')
                utility_rows.extend(utility['rows'])
                omit_forwards += utility['summary']['omit_forwards']
            for b, cid in enumerate(view['conversation_ids']):
                for t in range(view['lengths'][b]):
                    rows.append(dict(conversation_id=str(cid), utterance=t,
                        label=float(view['labels'][b, t]), prediction=float(logits[t, b]),
                        availability=view['availability'][t, b].long().tolist(),
                        transition_lag=t-onset[b]+1 if family == 'transition' and t >= onset[b] else None))
        key = str(condition)
        metrics[key] = prediction_metrics([r['label'] for r in rows], [r['prediction'] for r in rows])
        if family == 'transition':
            metrics[key]['post_onset'] = {}
            for lag in range(1, 5):
                chosen = [r for r in rows if r['transition_lag'] == lag]
                metrics[key]['post_onset'][str(lag)] = prediction_metrics(
                    [r['label'] for r in chosen], [r['prediction'] for r in chosen])
        if collect:
            records[key] = rows
        if utility_options is not None:
            from .fwc_evaluation import summarize_write_utility
            metrics[key]['write_utility'] = summarize_write_utility(utility_rows, omit_forwards)
            if collect:
                records[key+'_write_utility'] = utility_rows
    return metrics, records


def source_identity():
    root = Path(__file__).resolve().parents[1]
    names = [Path('config.py')]
    for package in ('gcnet_missing_m3', 'gcnet_modality_jepa', 'gcnet',
                    'gcnet_jepa_replacement', 'gcnet_plci_jepa'):
        names.extend(p.relative_to(root) for p in (root/package).glob('*.py'))
    return {str(p): hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(names)}


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()+b'\0')
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_gradient_norms(model):
    sums = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith('observed_set.'):
            group = 'shared_observed_encoder'
        elif name.startswith('osram.key_projectors.'):
            group = 'write_key_projectors'
        elif name.startswith('osram.value_projectors.'):
            group = 'write_value_projectors'
        elif name.startswith('osram.query_projector.'):
            group = 'query_projectors'
        elif name.startswith(('osram.local_', 'osram.emotion_', 'smax_fc.')):
            group = 'local_flat_classifier'
        else:
            group = 'memory_and_other'
        sums[group] = sums.get(group, 0.)+float(parameter.grad.detach().double().square().sum())
    return {k: math.sqrt(v) for k, v in sums.items()}


def make_loaders(args, config):
    root = Path(args.dataset_root).resolve()
    label = root/'CMUMOSI/CMUMOSI_features_raw_2way.pkl'
    features = [root/'CMUMOSI/features'/name for name in
                ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    if not label.is_file() or not all(p.is_dir() for p in features):
        raise FileNotFoundError(f'MOSI label/features missing under {root}')
    # The legacy config module may already be imported: change the exact label
    # route explicitly, not only the environment variable after import.
    old.config.PATH_TO_LABEL['CMUMOSI'] = str(label)
    loaders = old.get_loaders(*map(str, features), num_folder=1, dataset='CMUMOSI',
        batch_size=config.batch_size, num_workers=0, seed=config.seed,
        validation_fraction=config.validation_fraction, evaluation_protocol='official')
    train, valid, test, *dimensions = loaders
    feature_versions = {}
    for directory in features:
        digest, count, size = hashlib.sha256(), 0, 0
        for path in sorted(directory.rglob('*.npy')):
            content = path.read_bytes()
            digest.update(str(path.relative_to(directory)).encode()+b'\0')
            digest.update(hashlib.sha256(content).digest())
            count += 1
            size += len(content)
        if count == 0:
            raise ValueError(f'No feature arrays in {directory}')
        feature_versions[str(directory)] = dict(files=count, bytes=size, sha256=digest.hexdigest())
    provenance = dict(label=str(label), label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
        feature_versions=feature_versions,
        feature_roots=list(map(str, features)), splits={name: loader.protocol_metadata
        for name, loader in zip(('train', 'validation', 'test'), (train[0], valid[0], test[0]))})
    return train[0], valid[0], test[0], tuple(dimensions), provenance


def train(args):
    from .fwc_sampling import make_trajectories, sample_pairs
    from .mc_fwc import credit_backward
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'last.pt').exists() and not args.resume:
        raise ValueError('Existing training state: use --resume; no silent overwrite')
    if (output/'config.json').exists() and not args.resume:
        raise ValueError('Output already contains a run; use a new directory')
    if args.smoke_steps and args.resume:
        raise ValueError('Smoke and resume are distinct workflows')
    if args.epochs < 1 or args.smoke_steps < 0 or args.horizon < 1 or args.pairs_per_conversation < 1:
        raise ValueError('Invalid budget/pair configuration')
    if args.tau <= 0 or args.lambda_fwc < 0:
        raise ValueError('Invalid FWC scale')
    config = baseline_config(args.seed, args.device, args.epochs)
    old.set_random_seed(config.seed)
    torch.set_num_threads(args.threads)
    train_loader, val_loader, test_loader, dimensions, data = make_loaders(args, config)
    old.set_random_seed(old.SeedBundle(config.seed).derive('missing_m3_model_init:fold:5'))
    model = build_model(config, dimensions)
    initialization_sha256 = model_digest(model)
    groups, group_info = old._optimizer_parameter_groups(model, config)
    optimizer = torch.optim.Adam(groups, weight_decay=config.weight_decay)
    options = {k: v for k, v in vars(args).items() if k not in ('resume', 'output', 'config')}
    identity = dict(options=options, config=asdict(config), data=data, source=source_identity())
    identity = json.loads(json.dumps(identity))
    history, best, start = [], None, 0
    if args.resume:
        state = restore_training(output/'last.pt', model, optimizer, identity)
        history, best, start = state['history'], state['best'], state['next_epoch']
    write_json(output/'config.json', identity)
    provenance = dict(status='running', pid=os.getpid(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__,
        initialization_sha256=initialization_sha256,
        parameters=sum(p.numel() for p in model.parameters()), optimizer_groups=group_info,
        protocol='one-checkpoint-eight-rate-validation; never per-rate Test-oracle')
    write_json(output/'provenance.json', provenance)
    schedules = old._schedules(config, 'train')
    validation_schedules = old._schedules(config, 'validation')
    train_start = time.monotonic()
    smoke_rows = []
    total_steps = 0
    if torch.device(args.device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    epoch_budget = max(args.epochs, args.smoke_steps) if args.smoke_steps else args.epochs
    for epoch in range(start, epoch_budget):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(epoch)
        batches = []
        for batch, raw in enumerate(train_loader):
            data_batch = old._move_batch(raw, torch.device(args.device))
            rate = BalancedBatchRateSchedule().rate_for(epoch, batch)
            view = old._prepare_view(data_batch, schedules[rate], epoch, dimensions)
            regimes, trajectory = ['random']*len(view['lengths']), {}
            if args.mode != 'E0':
                av, regimes, trajectory = make_trajectories(view['availability'], view['umask'],
                    view['conversation_ids'], seed=batch_seed(args.seed, epoch, batch, 'trajectory'),
                    ratios=args.trajectory_ratios)
                view = replace_availability(view, av, dimensions)
            pairs = [] if args.mode == 'E0' else sample_pairs(
                view['availability'], view['umask'], view['conversation_ids'],
                seed=batch_seed(args.seed, epoch, batch, 'pairs'), horizon=args.horizon,
                pairs_per_conversation=args.pairs_per_conversation, strategy=args.pair_strategy,
                regimes=regimes)
            optimizer.zero_grad(set_to_none=True)
            step_start = time.monotonic()
            logits, task_loss, diagnostic = credit_backward(model, view, pairs,
                mode=args.mode, tau=args.tau, lambda_fwc=args.lambda_fwc,
                shuffle_seed=batch_seed(args.seed, epoch, batch, 'shuffle'))
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(task_loss)):
                raise FloatingPointError('Nonfinite Keep prediction or task loss')
            group_gradients = parameter_gradient_norms(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm,
                                                 error_if_nonfinite=True)
            optimizer.step()
            if torch.device(args.device).type == 'cuda':
                torch.cuda.synchronize()
            total_steps += 1
            item = dict(epoch=epoch+1, batch=batch, rate=rate, task_loss=float(task_loss.detach()),
                gradient_norm_before_clip=float(norm), clip_factor=min(1., config.gradient_clip_norm/(float(norm)+1e-6)),
                seconds=time.monotonic()-step_start, trajectory=trajectory, diagnostics=diagnostic,
                total_gradient_norms_by_group_before_clip=group_gradients,
                pair_hash=hashlib.sha256(repr(pairs).encode()).hexdigest(),
                mask_hash=old._sha256_tensor(view['availability']),
                valid_utterances=int(view['umask'].sum()), learning_rate=learning_rate(epoch))
            valid_av = view['availability'][view['umask'].T.bool()]
            item['actual_missing_fraction'] = float((valid_av == 0).float().mean())
            item['availability_counts'] = {''.join(map(str, bits)): int((valid_av == valid_av.new_tensor(bits)).all(-1).sum())
                for bits in ((1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1))}
            batches.append(item)
            if args.smoke_steps:
                smoke_rows.append(item)
                if total_steps >= args.smoke_steps:
                    result = dict(label='ENGINEERING SMOKE ONLY; not performance evidence',
                        initialization_sha256=initialization_sha256,
                        steps=smoke_rows, elapsed_seconds=time.monotonic()-train_start,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.device(args.device).type == 'cuda' else 0)
                    write_json(output/'SMOKE.json', result)
                    provenance.update(status='smoke-complete', optimizer_steps=total_steps)
                    write_json(output/'provenance.json', provenance)
                    print(json.dumps({k:v for k,v in result.items() if k != 'steps'}, allow_nan=False), flush=True)
                    return result
        if args.smoke_steps:
            continue
        validation, _ = evaluate(model, val_loader, validation_schedules, dimensions, args.device)
        score = selection_score(validation)
        if best is None or score > best['score']:
            best = dict(epoch=epoch+1, score=score, model=_cpu(model.state_dict()),
                        config=asdict(config), selection_split='validation', validation=validation)
        record = dict(epoch=epoch+1, train=batches, validation=validation,
                      validation_mean_weighted_f1=score, best_epoch=best['epoch'])
        history.append(record)
        save_training(output/'last.pt', model, optimizer, identity, epoch+1, history, best)
        # last.pt is authoritative: canonical best/history can be reconstructed.
        _atomic(output/'best.pt', lambda stream: torch.save(best, stream))
        write_json(output/'history.json', history)
        print(f'epoch={epoch+1:03d} validation_wf1={score:.6f} best_epoch={best["epoch"]}', flush=True)
    if best is None:
        raise RuntimeError('No validation-selected checkpoint')
    model.load_state_dict(best['model'], strict=True)
    test_schedules = old._schedules(config, 'test')
    result = dict(selection_split='validation', selected_epoch=best['epoch'],
                  selected_validation_score=best['score'], test={}, elapsed_seconds=time.monotonic()-train_start)
    for family in ('random', 'persistent', 'transition'):
        metrics, predictions = evaluate(model, test_loader, test_schedules, dimensions,
                                       args.device, family=family, collect=True,
                                       utility_options=vars(args))
        result['test'][family] = metrics
        write_json(output/f'predictions_{family}.json', predictions)
    result['eight_rate_mean_wf1'] = selection_score(result['test']['random'])
    result['high_missing_mean_wf1'] = sum(result['test']['random'][str(r)]['weighted_f1'] for r in (.5, .6, .7))/3
    result['peak_cuda_bytes'] = torch.cuda.max_memory_allocated() if torch.device(args.device).type == 'cuda' else 0
    write_json(output/'metrics.json', result)
    write_json(output/'COMPLETED.json', dict(selected_epoch=best['epoch'], epochs=args.epochs,
                                           status='complete', inference='unchanged single-view CHM'))
    provenance.update(status='complete', selected_epoch=best['epoch'], epochs=args.epochs)
    write_json(output/'provenance.json', provenance)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--mode', choices=('E0', 'E1', 'E2', 'E3', 'E4', 'shuffled'), default='E4')
    p.add_argument('--seed', type=int, default=66)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--output', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--horizon', type=int, default=4)
    p.add_argument('--pairs-per-conversation', type=int, default=1)
    p.add_argument('--pair-strategy', default='uniform')
    p.add_argument('--trajectory-ratios', type=float, nargs=3, default=(.5, .25, .25))
    p.add_argument('--tau', type=float, default=1.)
    p.add_argument('--lambda-fwc', type=float, default=.1)
    return p


def main():
    p = parser()
    preliminary, _ = p.parse_known_args()
    if preliminary.config:
        options = json.loads(Path(preliminary.config).read_text())
        unknown = set(options)-{action.dest for action in p._actions}
        if unknown:
            p.error(f'Unknown configuration fields: {sorted(unknown)}')
        p.set_defaults(**options)
    train(p.parse_args())


if __name__ == '__main__':
    main()
