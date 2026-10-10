"""Existing MOSI Flat/Nested checkpoints: input task gradients, no training."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REMOTE = Path('/data2/yb/remote_experiments')
SLOTS = ('Local', 'Base', 'Gap-A', 'Gap-T', 'Gap-V', 'Memory-joint')
HEALTHY = {
    6: 'GPU-e4cafb17-818e-216a-b94a-7440063a9153',
    7: 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e',
}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def sources(seed):
    flat = REMOTE / 'osram_mosi_memory_gap_ablation_20260920/full' / f'seed_{seed}'
    nested = (REMOTE / 'osram_new40_gpu0123_20261004/attempt2/runs/nested_gnn_rooted_evidence/seed_66'
              if seed == 66 else REMOTE / 'osram_readout_top3_3seed_20261005/runs/nested_gnn_rooted_evidence' / f'seed_{seed}')
    return {'flat': flat, 'nested': nested}


def capture(model, view, kind):
    import torch
    snapshots = []
    def hook(module, args):
        if kind == 'nested':
            snapshots.append(tuple(x.detach().clone() for x in args))
        else:
            x = args[0].detach().clone()
            local = x[..., :256]
            memory = x[..., 256:].reshape(*x.shape[:2], 4, 1024)
            snapshots.append((local, memory[..., 0, :], memory[..., 1:, :],
                              view['availability'], view['umask']))
    module = model.osram.meaningful_block if kind == 'nested' else model.osram.emotion_adapter
    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            result = model([view['incomplete']], view['availability'], view['qmask'],
                           view['umask'], view['lengths'], predict_missing=False)
    finally:
        handle.remove()
    if len(snapshots) != 1:
        raise AssertionError('Expected one pre-readout input snapshot')
    local, base, gap, availability, umask = snapshots[0]
    if not (torch.equal(availability, view['availability']) and torch.equal(umask, view['umask'])):
        raise AssertionError('Captured masks differ from the actual forward')
    if base[..., 512:].count_nonzero() or gap[..., 512:].count_nonzero():
        raise AssertionError('Expected zero backward Memory half')
    return snapshots[0], result[0].detach().squeeze(-1)


def summarize(rows, output):
    grouped = {}
    for row in rows:
        if row['included']:
            key = row['seed'], row['rate'], row['model'], row['slot']
            grouped.setdefault(key, []).append(row)
    per_rate = []
    for (seed, rate, model, slot), values in sorted(grouped.items()):
        record = dict(seed=seed, rate=rate, model=model, slot=slot, n=len(values))
        for field in ('loss_gradient_l2', 'loss_gradient_rms', 'jacobian_l2',
                      'jacobian_rms', 'feature_l2'):
            x = [r[field] for r in values]
            record[field + '_mean'] = statistics.mean(x)
            record[field + '_median'] = statistics.median(x)
        per_rate.append(record)
    paired = []
    index = {(r['seed'], r['rate'], r['model'], r['slot']): r for r in per_rate}
    for seed in (66, 67, 68):
        for rate in (i / 10 for i in range(8)):
            for slot in SLOTS:
                keys = [(seed, rate, model, slot) for model in ('flat', 'nested')]
                if not all(key in index for key in keys):
                    # No active Gap exists at rate0.0; this is not a missing run.
                    continue
                f, n = (index[key] for key in keys)
                if f['n'] != n['n']:
                    raise AssertionError('Paired gradient cohort mismatch')
                record = dict(seed=seed, rate=rate, slot=slot, n=f['n'])
                for field in ('loss_gradient_l2', 'jacobian_l2', 'feature_l2'):
                    a, b = f[field + '_mean'], n[field + '_mean']
                    record[field + '_flat'] = a
                    record[field + '_nested'] = b
                    record[field + '_nested_over_flat'] = b / a if a else None
                paired.append(record)
    for name, table in [('per_rate', per_rate), ('paired', paired)]:
        with (output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    macro = []
    for slot in SLOTS:
        values = [r for r in paired if r['slot'] == slot]
        record = dict(slot=slot, seed_rate_count=len(values))
        for field in ('loss_gradient_l2', 'jacobian_l2', 'feature_l2'):
            for model in ('flat', 'nested'):
                record[field + '_' + model] = statistics.mean(r[field + '_' + model] for r in values)
            a, b = record[field + '_flat'], record[field + '_nested']
            record[field + '_nested_over_flat'] = b / a if a else None
        macro.append(record)
    return dict(per_rate=per_rate, paired=paired, macro=macro)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=tuple(HEALTHY), default=7)
    parser.add_argument('--seeds', nargs='+', type=int, default=[66, 67, 68])
    parser.add_argument('--rates', nargs='+', type=float, default=[i / 10 for i in range(8)])
    args = parser.parse_args()
    expected = HEALTHY[args.gpu]
    actual = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=uuid',
                                      '--format=csv,noheader'], text=True).strip()
    if actual != expected or os.environ.get('CUDA_VISIBLE_DEVICES') != expected:
        raise ValueError('Explicit healthy physical GPU UUID required')
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 pid=os.getpid(), gpu=args.gpu, gpu_uuid=expected, seeds=args.seeds,
                 rates=args.rates, records=[], label='INTERNAL DIAGNOSTIC ONLY')
    write(args.output / 'STATUS.json', state)
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, _move_batch, _prepare_view
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    from experiments.osram_nested_diagnostics_20261007.evaluate import state_hash
    from experiments.osram_frozen_memory_audit_20261009.probe import metrics
    from experiments.osram_nested_input_gradient_20261010.gradient import measure_input_gradients
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    dataset = Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset')
    os.environ['GCNET_DATASET_ROOT'] = str(dataset)
    features = [str(dataset / 'CMUMOSI/features' / name) for name in
                ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    rows, matched = [], {}
    try:
        for seed in args.seeds:
            for kind, source in sources(seed).items():
                raw_config = json.loads((source / 'config.json').read_text())
                config = TrainConfig(**raw_config)
                if config.seed != seed or config.training_objective != 'emotion-only':
                    raise ValueError('Wrong source seed or objective')
                if (config.osram_readout_fusion != 'flat' or config.osram_bidirectional
                        or config.mosi_task_mode != 'regression' or config.task_regression_loss != 'mse'):
                    raise ValueError('Expected original causal cfg84 regression/MSE')
                method = getattr(config, 'osram_meaningful_block', 'none')
                if method != ('none' if kind == 'flat' else 'nested_gnn_rooted_evidence'):
                    raise ValueError('Unexpected Nested or Flat variant')
                train, val, test, *dimensions = get_loaders(
                    audio_root=features[0], text_root=features[1], video_root=features[2],
                    num_folder=1, dataset='CMUMOSI', batch_size=config.batch_size,
                    num_workers=0, seed=seed, validation_fraction=config.validation_fraction,
                    evaluation_protocol=config.evaluation_protocol)
                del train, val
                loader = test[0]
                model = _build_model(config, tuple(dimensions)).to(device).eval().requires_grad_(False)
                source_metrics = json.loads((source / 'metrics.json').read_text())
                for rate in args.rates:
                    path = source / f'best_miss_{rate:.1f}.pt'.replace('.', 'p', 1)
                    before_file = sha(path)
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    model.load_state_dict(saved['model'], strict=True)
                    epoch = int(saved['epoch'])
                    del saved
                    before_state = state_hash(model)
                    prediction, target, identities = [], [], []
                    maximum_error = 0.
                    loader.sampler.set_epoch(0)
                    for raw in loader:
                        view = _prepare_view(_move_batch(raw, device), _schedules(config, 'test')[rate], 0, tuple(dimensions))
                        inputs, original = capture(model, view, kind)
                        measurement = measure_input_gradients(model, *inputs, view['labels'], kind=kind)
                        valid = view['umask'].T.bool()
                        error = float((measurement['prediction'][valid] - original[valid]).abs().max())
                        maximum_error = max(maximum_error, error)
                        if error > 1e-5:
                            raise AssertionError('Gradient replay differs from original model prediction')
                        prediction.extend(original[valid].cpu().tolist())
                        target.extend(view['labels'].T[valid].cpu().tolist())
                        for t, b in valid.nonzero().cpu().tolist():
                            av = view['availability'][t, b].cpu().tolist()
                            pattern = ''.join(name for name, present in zip(('A', 'T', 'V'), av) if present)
                            identity = [str(view['conversation_ids'][b]), t, av]
                            identities.append(identity)
                            base_row = dict(seed=seed, rate=rate, model=kind, conversation=identity[0],
                                            utterance=t, pattern=pattern, label=float(view['labels'][b, t]),
                                            prediction=float(original[t, b]))
                            for i, slot in enumerate(SLOTS):
                                if i == 0:
                                    included = True
                                    dim = 256
                                    feature = measurement['feature_norm_local'][t, b]
                                    loss_norm = measurement['loss_grad_norm_local'][t, b]
                                    jac_norm = measurement['jacobian_norm_local'][t, b]
                                elif i < 5:
                                    included = t > 0 and (i == 1 or not av[i - 2])
                                    dim = 512
                                    feature = measurement['feature_norm_memory'][t, b, i - 1]
                                    loss_norm = measurement['loss_grad_norm_memory'][t, b, i - 1]
                                    jac_norm = measurement['jacobian_norm_memory'][t, b, i - 1]
                                else:
                                    included = t > 0
                                    active = measurement['active'][t, b]
                                    dim = int(active.sum()) * 512
                                    feature = measurement['feature_norm_memory'][t, b].norm()
                                    loss_norm = measurement['grad_memory'][t, b].norm()
                                    jac_norm = measurement['jacobian_memory'][t, b].norm()
                                rows.append(dict(base_row, slot=slot, included=included,
                                                 loss_gradient_l2=float(loss_norm),
                                                 loss_gradient_rms=float(loss_norm) / dim ** .5,
                                                 jacobian_l2=float(jac_norm), jacobian_rms=float(jac_norm) / dim ** .5,
                                                 feature_l2=float(feature)))
                    key = seed, rate
                    if kind == 'flat':
                        matched[key] = (identities, target)
                    elif matched.get(key) != (identities, target):
                        raise AssertionError('Flat and Nested input masks/sample IDs/labels differ')
                    metric = metrics(np.asarray(target), np.asarray(prediction))
                    reference = (source_metrics['test'][str(rate)]['weighted_f1'] if kind == 'flat'
                                 else source_metrics['selected_weighted_f1_by_rate'][str(rate)])
                    if abs(metric['weighted_f1_nonzero'] - reference) > 1e-10:
                        raise AssertionError('Existing checkpoint W-F1 parity failed')
                    if (state_hash(model) != before_state or sha(path) != before_file
                            or any(p.grad is not None or p.requires_grad for p in model.parameters())):
                        raise AssertionError('Frozen model or checkpoint changed')
                    record = dict(seed=seed, rate=rate, model=kind, checkpoint=str(path),
                                  checkpoint_sha256=before_file, epoch=epoch, metric=metric,
                                  prediction_replay_max_abs=maximum_error, frozen_unchanged=True,
                                  mask_sha256=hashlib.sha256(json.dumps(identities).encode()).hexdigest())
                    state['records'].append(record)
                    write(args.output / 'STATUS.json', state)
                    print(json.dumps(record), flush=True)
                del model
                torch.cuda.empty_cache()
        with (args.output / 'utterances.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize(rows, args.output)
        summary.update(label='INTERNAL DIAGNOSTIC ONLY', records=state['records'],
                       complete=len(state['records']) == 2 * len(args.seeds) * len(args.rates),
                       loss='sum of valid per-utterance MSE; no batch divisor',
                       model_parameters_frozen=True, optimizer_steps=0)
        write(args.output / 'SUMMARY.json', summary)
        state.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
        write(args.output / 'STATUS.json', state)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(args.output / 'STATUS.json', state)
        raise


if __name__ == '__main__':
    main()
