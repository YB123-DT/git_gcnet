"""Frozen original Flat + trainable evidence Gate: 24 paired INTERNAL TEST-ORACLE jobs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import MethodType

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_local_evidence_gate_20260930 import run as common
read, write, sha = common.read, common.write, common.sha
SEEDS, RATES = (66, 67, 68), tuple(i / 10 for i in range(8))
LABEL = common.LABEL
GATE_PREFIX = 'osram.local_evidence_gate.'
GPU_UUID = 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45'


def frozen_state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith(GATE_PREFIX):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_flat_state(model, state):
    expected = {k for k in model.state_dict() if k.startswith(GATE_PREFIX)}
    if not expected or any(k.startswith(GATE_PREFIX) for k in state):
        raise ValueError('source must be original Flat without Gate parameters')
    missing = set(model.state_dict()) - set(state)
    unexpected = set(state) - set(model.state_dict())
    if missing != expected or unexpected:
        raise ValueError(f'incompatible Flat checkpoint: missing={missing}, unexpected={unexpected}')
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise ValueError('unexpected state loading result')


def freeze_except_gate(model):
    import torch
    model.requires_grad_(False)
    gate = model.osram.local_evidence_gate
    gate.requires_grad_(True)
    def train_gate_only(self, mode=True):
        torch.nn.Module.train(self, False)
        self.osram.local_evidence_gate.train(mode)
        return self
    model.train = MethodType(train_gate_only, model)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if {id(p) for p in parameters} != {id(p) for p in gate.parameters()}:
        raise ValueError('non-Gate trainable parameter')
    return parameters


def rate_key(rate):
    return f'{rate:.1f}'.replace('.', 'p')


def task_root(args, seed, rate):
    return args.output_root / f'seed_{seed}' / f'miss_{rate_key(rate)}'


def preflight(args):
    if args.epochs != 100 and not args.smoke:
        raise ValueError('full protocol requires 100 epochs')
    if args.smoke and 'smoke' not in args.output_root.name:
        raise ValueError('smoke needs separate smoke output root')
    if not args.smoke and 'smoke' in args.output_root.name:
        raise ValueError('full run cannot use smoke output')
    records = []
    for seed in SEEDS:
        config = common.configuration_dict(seed, args.reference_root)
        if config['optimizer'] != 'adam' or config['learning_rate'] != .001 or config['weight_decay'] != 1e-5:
            raise ValueError('reference optimizer differs from verified cfg84')
        source = args.reference_root / f'seed_{seed}'
        if read(source / 'PROVENANCE.json')['status'] != 'complete':
            raise ValueError('incomplete reference')
        if read(source / 'metrics.json')['selection_protocol'] != 'per-rate-test-oracle':
            raise ValueError('unexpected reference selection protocol')
        config['epochs'] = args.epochs
        for rate in RATES:
            checkpoint = source / f'best_miss_{rate_key(rate)}.pt'
            predictions = source / f'predictions_miss_{rate_key(rate)}.npz'
            if not checkpoint.is_file() or not predictions.is_file():
                raise FileNotFoundError(f'{checkpoint} or {predictions}')
            records.append(dict(seed=seed, rate=rate, config=config, checkpoint=str(checkpoint),
                                reference_config_sha256=sha(source / 'config.json')))
    for root in common.feature_roots(args):
        if not Path(root).is_dir():
            raise FileNotFoundError(root)
    return records


def canonical_rows(artifacts):
    import numpy as np
    rows = np.concatenate([artifacts['labels'].reshape(-1, 1), artifacts['availability']], axis=1).astype(np.float64)
    return rows[np.lexsort(tuple(rows[:, i] for i in reversed(range(rows.shape[1]))))]


def train(args):
    import numpy as np
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, train_epoch, evaluate_rate, _apply_epoch_learning_rate
    from gcnet_modality_jepa.train_gcnet import get_loaders, set_random_seed
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    if args.rate is None:
        raise ValueError('--seed requires --rate')
    record = next(r for r in preflight(args) if r['seed'] == args.seed and r['rate'] == args.rate)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('CUDA_VISIBLE_DEVICES=0 required')
    gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if gpu != GPU_UUID:
        raise ValueError('GPU0 UUID mismatch')
    output = task_root(args, args.seed, args.rate)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(record['checkpoint'])
    provenance = dict(record, status='training', label=LABEL, source_checkpoint_sha256=sha(checkpoint),
        code_sha256={p: sha(REPO / p) for p in ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py',
            'gcnet_missing_m3/train_gcnet.py', 'experiments/osram_frozen_evidence_gate_20260930/run.py')},
        server='biggpu', gpu_uuid=GPU_UUID, host_gpu=0, torch=torch.__version__,
        started_utc=datetime.now(timezone.utc).isoformat(), resume_supported=False,
        selection='target-rate test-oracle including epoch0; nonnegative selected gains are guaranteed by construction')
    write(output / 'PROVENANCE.json', provenance)
    write(output / 'config.json', record['config'])
    try:
        torch.set_num_threads(2)
        c = TrainConfig(**record['config'])
        set_random_seed(c.seed)
        roots = common.feature_roots(args)
        train_loaders, _, test_loaders, ad, td, vd = get_loaders(audio_root=roots[0], text_root=roots[1],
            video_root=roots[2], num_folder=1, dataset=c.dataset, batch_size=c.batch_size, num_workers=0,
            seed=c.seed, validation_fraction=c.validation_fraction, evaluation_protocol=c.evaluation_protocol)
        dims, device = (ad, td, vd), torch.device('cuda:0')
        model = _build_model(c, dims).to(device)
        state = torch.load(checkpoint, map_location='cpu')
        load_flat_state(model, state['model'])
        provenance['source_epoch'] = int(state['epoch'])
        del state
        parameters = freeze_except_gate(model)
        anchor_hash = frozen_state_hash(model)
        provenance['frozen_state_sha256'] = anchor_hash
        provenance['trainable_parameters'] = sum(p.numel() for p in parameters)
        write(output / 'PROVENANCE.json', provenance)
        optimizer = torch.optim.Adam(parameters, lr=c.learning_rate, weight_decay=c.weight_decay)
        train_schedules, test_schedule = _schedules(c, 'train'), _schedules(c, 'test')[args.rate]
        def evaluate():
            return evaluate_rate(model, test_loaders[0], test_schedule, c.dataset, dims, device, True,
                                 c.mosi_task_mode, c.task_regression_loss, c.task_smooth_l1_beta)
        initial, initial_predictions = evaluate()
        reference = read(args.reference_root / f'seed_{args.seed}/metrics.json')['test'][f'{args.rate:.1f}']
        if abs(initial['weighted_f1'] - reference['weighted_f1']) > 1e-10:
            raise AssertionError('epoch0 does not reproduce original Flat W-F1')
        with np.load(checkpoint.parent / f'predictions_miss_{rate_key(args.rate)}.npz') as archive:
            if not np.array_equal(canonical_rows(initial_predictions), canonical_rows(archive)):
                raise AssertionError('reference canonical labels/masks differ')
        provenance['source_mask_sha256'] = reference.get('mask_sha256')
        provenance['epoch0_mask_sha256'] = initial['mask_sha256']
        np.savez_compressed(output / 'epoch0_predictions.npz', **initial_predictions)
        best, best_epoch = initial, 0
        history = []
        def save_state(name, epoch):
            payload = dict(gate=model.osram.local_evidence_gate.state_dict(), optimizer=optimizer.state_dict(),
                epoch=epoch, source_checkpoint=str(checkpoint), source_checkpoint_sha256=provenance['source_checkpoint_sha256'],
                frozen_state_sha256=anchor_hash, config=record['config'], best_epoch=best_epoch,
                best_metrics=best, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate(), resume_supported=False)
            temporary = output / (name + '.tmp')
            torch.save(payload, temporary)
            temporary.replace(output / name)
        save_state('best.pt', 0)
        np.savez_compressed(output / 'predictions.npz', **initial_predictions)
        for epoch in range(c.epochs):
            _apply_epoch_learning_rate(optimizer, c, epoch)
            sampler = getattr(train_loaders[0], 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
            train_metrics = train_epoch(model, train_loaders[0], optimizer, c, train_schedules, epoch, dims, device)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise AssertionError('nonfinite Gate gradients')
            if frozen_state_hash(model) != anchor_hash:
                raise AssertionError('frozen parameter/buffer changed')
            metrics, predictions = evaluate()
            for key in ('labels', 'availability'):
                if not np.array_equal(initial_predictions[key], predictions[key]):
                    raise AssertionError(f'evaluation {key} changed')
            if metrics['mask_sha256'] != initial['mask_sha256']:
                raise AssertionError('evaluation mask changed')
            if metrics['weighted_f1'] > best['weighted_f1']:
                best, best_epoch = metrics, epoch + 1
                save_state('best.pt', epoch + 1)
                np.savez_compressed(output / 'predictions.npz', **predictions)
            history.append(dict(epoch=epoch + 1, train=train_metrics, test=metrics, frozen_state_unchanged=True))
            write(output / 'history.json', history)
            save_state('last.pt', epoch + 1)
            print(f'seed={args.seed} target_rate={args.rate:.1f} epoch={epoch+1:03d} wf1={metrics["weighted_f1"]:.6f} best={best["weighted_f1"]:.6f}', flush=True)
        if frozen_state_hash(model) != anchor_hash:
            raise AssertionError('final frozen state changed')
        np.savez_compressed(output / 'last_predictions.npz', **predictions)
        write(output / 'metrics.json', dict(label=LABEL, epoch0=initial, best=best, best_epoch=best_epoch,
            final=metrics, epochs=c.epochs, selection_includes_epoch0=True, source_epoch=provenance['source_epoch']))
    except BaseException as error:
        write(output / 'PROVENANCE.json', dict(provenance, status='failed', error=repr(error)))
        raise
    write(output / 'PROVENANCE.json', dict(provenance, status='complete', frozen_state_unchanged=True,
                                         completed_utc=datetime.now(timezone.utc).isoformat()))


def summarize(args):
    rows = []
    for seed in SEEDS:
        for rate in RATES:
            root = task_root(args, seed, rate)
            if read(root / 'PROVENANCE.json')['status'] != 'complete':
                raise ValueError('all 24 tasks must complete')
            rows.append(dict(seed=seed, rate=rate, **read(root / 'metrics.json')))
    seeds = []
    for seed in SEEDS:
        values = [r for r in rows if r['seed'] == seed]
        seeds.append(dict(seed=seed, **{variant: dict(
            mean_8rate=statistics.mean(r[variant]['weighted_f1'] * 100 for r in values),
            high_missing=statistics.mean(r[variant]['weighted_f1'] * 100 for r in values if r['rate'] >= .5))
            for variant in ('epoch0', 'best', 'final')}))
    write(args.output_root / 'SUMMARY.json', dict(label=LABEL, rows=rows, seeds=seeds,
        warning='best includes epoch0: selected nonnegative gains are guaranteed, not evidence of consistent Gate learning'))


def command(args, *mode):
    return [sys.executable, '-u', str(Path(__file__).resolve()), *mode,
        '--output-root', str(args.output_root.resolve()), '--reference-root', str(args.reference_root.resolve()),
        '--dataset-root', str(args.dataset_root.resolve()), '--epochs', str(args.epochs),
        '--max-tasks-per-gpu', str(args.max_tasks_per_gpu)] + (['--smoke'] if args.smoke else [])


def coordinate(args):
    write(args.output_root / 'PREFLIGHT.json', preflight(args))
    pending, active, records = [(s, r) for s in SEEDS for r in RATES], {}, []
    failure_seen = False
    while pending or active:
        for pid, (process, record) in list(active.items()):
            code = process.poll()
            if code is not None:
                record.update(returncode=code, status='complete' if code == 0 else 'failed')
                if code != 0:
                    failure_seen = True
                    write(args.output_root / 'unlaunched.json', [dict(seed=s, rate=r) for s, r in pending])
                    pending.clear()
                del active[pid]
                write(args.output_root / 'children.json', records)
        while pending and not failure_seen and len(active) < args.max_tasks_per_gpu:
            info = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.free',
                                           '--format=csv,noheader,nounits'], text=True).strip().split(',')
            if info[0].strip() != GPU_UUID:
                raise ValueError('GPU0 UUID mismatch')
            if int(info[1]) < 3000:
                break
            seed, rate = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       GCNET_DATASET_ROOT=str(args.dataset_root), PYTHONPATH=str(REPO))
            with (args.output_root / f'seed_{seed}_miss_{rate_key(rate)}.log').open('x') as log:
                process = subprocess.Popen(command(args, '--seed', str(seed), '--rate', str(rate)),
                    cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            record = dict(seed=seed, rate=rate, pid=process.pid, host_gpu=0, status='running')
            records.append(record)
            active[process.pid] = process, record
            write(args.output_root / 'children.json', records)
            time.sleep(20)
        if pending or active:
            time.sleep(5)
    if any(r['returncode'] != 0 for r in records):
        raise RuntimeError('one or more children failed; inspect children.json')
    summarize(args)


def launch(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(p.name != '.launch.lock' for p in args.output_root.iterdir()):
            raise FileExistsError('refusing overwrite/nonempty output directory')
        preflight(args)
        env = dict(os.environ, FROZEN_EVIDENCE_GATE_LOCK_FD=str(lock.fileno()))
        with (args.output_root / 'coordinator.log').open('x') as log:
            process = subprocess.Popen(command(args, '--coordinate'), cwd=REPO, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        write(args.output_root / 'launch.json', dict(pid=process.pid, label=LABEL, host_gpu=0,
            max_concurrency=args.max_tasks_per_gpu, tasks=24, epochs=args.epochs,
            started_utc=datetime.now(timezone.utc).isoformat()))
        print(f'coordinator PID={process.pid} output={args.output_root}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--coordinate', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    mode.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--rate', type=float, choices=RATES)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-tasks-per-gpu', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--reference-root', type=Path, default=Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full'))
    parser.add_argument('--dataset-root', type=Path, default=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'))
    parser.add_argument('--output-root', type=Path, default=Path('/data1/yb/remote_experiments/osram_frozen_evidence_gate_20260930/runs'))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.launch:
        launch(args)
    elif args.coordinate:
        if 'FROZEN_EVIDENCE_GATE_LOCK_FD' not in os.environ:
            raise RuntimeError('use --launch')
        write(args.output_root / 'status.json', dict(status='running', label=LABEL))
        try:
            coordinate(args)
        except BaseException as error:
            write(args.output_root / 'status.json', dict(status='failed', error=repr(error), label=LABEL))
            raise
        write(args.output_root / 'status.json', dict(status='complete', label=LABEL))
    elif args.preflight:
        import json
        print(json.dumps(preflight(args), indent=2))
    elif args.summarize:
        summarize(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
