"""MOSI66 recent NoArea: cyclic training of one block on an own-rate frozen Flat."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import resource
import time


NUMERICAL_SOURCE = 'e0ca94c4c33de0efda2c2f4786fd289067e1ab2c'
REFERENCE_ROOT = Path('/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/execution_v2/train')
RATES = tuple(str(i / 10) for i in range(8))
EPOCHS = 150


def validate_scope(mode, rate, seed=66, fold=1):
    if mode not in ('parallel', 'serial') or type(seed) is not int or seed != 66 or type(fold) is not int or fold != 1:
        raise ValueError('Only parallel/serial MOSI66 single-fold frozen runs are authorized')
    if isinstance(rate, bool):
        raise ValueError('Only the eight rates 0.0 through 0.7 are authorized')
    try:
        rate = str(float(rate))
    except (TypeError, ValueError):
        raise ValueError('Only the eight rates 0.0 through 0.7 are authorized') from None
    if rate not in RATES:
        raise ValueError('Only the eight rates 0.0 through 0.7 are authorized')
    return rate


def learning_rate(epoch):
    if type(epoch) is not int or not 0 <= epoch < EPOCHS:
        raise ValueError('Epoch outside the finite stage2 budget')
    return 3e-4 if epoch < 100 else 3e-4 / 10


def improves(current, previous):
    return previous is None or current['WF1'] > previous['WF1']


def validate_history(history, epochs):
    if [row['epoch'] for row in history] != list(range(1, epochs + 1)):
        raise ValueError('Require complete contiguous training history')


def validate_epoch_trace(row, original):
    # Old Signature records used the adjacent float literal 3e-5. Compare the
    # original data/mask/update trace, but require the exact original LR formula.
    for key in ('epoch', 'optimizer_steps', 'training_rates', 'mask_trace_sha256', 'N'):
        if row.get(key) != original.get(key):
            raise ValueError('Training sampling/update trace differs from original Signature: ' + key)
    if row.get('learning_rate') != learning_rate(row['epoch'] - 1):
        raise ValueError('Learning rate differs from the exact 3e-4/10 expression')


def _check_frozen(model):
    if model.parent.training or any(p.requires_grad or p.grad is not None for p in model.parent.parameters()):
        raise RuntimeError('Frozen parent changed mode, requires gradients or received a gradient')


def build_optimizer(model):
    import torch
    _check_frozen(model)
    parameters = list(model.block.parameters())
    if not parameters or not all(p.requires_grad for p in parameters):
        raise RuntimeError('Every new block parameter must be trainable')
    return torch.optim.Adam(parameters, lr=learning_rate(0), weight_decay=1e-5)


def _forward_with_scan_count(model, view):
    """Count the actual CHM trajectory call and restore the method on failure."""
    osram = getattr(model.parent, 'osram', None)
    scan = getattr(osram, '_scan', None)
    if scan is None:
        return model(view), None
    calls = 0
    def counted_scan(*args, **kwargs):
        nonlocal calls
        calls += 1
        return scan(*args, **kwargs)
    had_instance_method = '_scan' in osram.__dict__
    osram._scan = counted_scan
    try:
        result = model(view)
    finally:
        if had_instance_method:
            osram._scan = scan
        else:
            del osram._scan
    if calls != 1:
        raise RuntimeError('Require exactly one live CHM _scan per training forward')
    return result, calls


def train_epoch(bundle, model, optimizer, epoch, rate, smoke):
    """Original preparation/MSE/clip; one live wrapper forward, no auxiliary loss."""
    import torch
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    trainer, cfg, loader = bundle['trainer'], bundle['cfg'], bundle['train_loader']
    device = next(model.block.parameters()).device
    loader.sampler.set_epoch(epoch)
    schedules = trainer._schedules(cfg, 'train')
    cycle = BalancedBatchRateSchedule(rates=cfg.train_missing_rates)
    model.train()
    _check_frozen(model)
    before = [p.detach().clone() for p in model.block.parameters()]
    total, count, steps, coverage, seen, scans = 0., 0, 0, [], [], []
    trace = hashlib.sha256()
    for index, raw in enumerate(loader):
        if smoke and index >= 1:
            break
        actual_rate = cycle.rate_for(epoch, index)
        seen.append(float(actual_rate))
        data = trainer._move_batch(raw, device)
        view = trainer._prepare_view(data, schedules[actual_rate], epoch, bundle['dimensions'])
        optimizer.zero_grad(set_to_none=True)
        result, scan_calls = _forward_with_scan_count(model, view)
        scans.append(scan_calls)
        if result['aux_loss'].detach().item() != 0:
            raise RuntimeError('No auxiliary loss is authorized')
        loss = trainer._task_loss(cfg.dataset, result['prediction'], view['labels'], view['umask'],
            cfg.mosi_task_mode, cfg.task_regression_loss, cfg.task_smooth_l1_beta)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite cyclic stage2 task MSE')
        loss.backward()
        _check_frozen(model)
        grads = [p.grad for p in model.block.parameters() if p.grad is not None]
        if len(grads) != len(before) or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Disconnected/nonfinite new block gradients')
        coverage.append(dict(connected=len(grads), nonzero=sum(int(g.ne(0).any()) for g in grads)))
        norm = torch.nn.utils.clip_grad_norm_(model.block.parameters(), cfg.gradient_clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient clipping norm')
        optimizer.step()
        n = int(view['umask'].sum())
        total += float(loss.detach()) * n
        count += n
        steps += 1
        trace.update(json.dumps(list(map(str, view['conversation_ids']))).encode())
        trace.update(view['availability'].detach().cpu().numpy().tobytes())
    if not count:
        raise RuntimeError('Empty cyclic stage2 training epoch')
    updated = sum(int(not torch.equal(old, p.detach())) for old, p in zip(before, model.block.parameters()))
    return dict(epoch=epoch + 1, training_rates=seen, evaluation_rate=str(rate), task_MSE=total / count,
        N=count, optimizer_steps=steps, mask_trace_sha256=trace.hexdigest(), gradients=coverage[-1],
        nonzero_gradient_batches=sum(int(row['nonzero'] > 0) for row in coverage),
        updated_parameters=updated, live_scan_calls=scans, learning_rate=optimizer.param_groups[0]['lr'])


def evaluate(model, rows):
    """Evaluate only the caller's protected own-rate cached Flat frames."""
    import numpy as np
    import torch
    from experiments.osram_signature_cyclic_multi_20261025.run import measured_metrics
    model.eval()
    pieces = []
    device = next(model.block.parameters()).device
    with torch.no_grad():
        for row in rows:
            frame = row['frame'].to(device)
            prediction = model.predict_frame(frame)
            valid = frame.valid.T
            pieces.append(dict(row['metadata'], availability=row['availability'],
                prediction=prediction.transpose(0, 1)[valid].cpu().numpy().reshape(-1),
                base_prediction=frame.prediction.transpose(0, 1)[valid].cpu().numpy().reshape(-1)))
    if not pieces:
        raise ValueError('Empty own-rate test cache')
    arrays = {key: np.concatenate([row[key] for row in pieces]) for key in pieces[0]}
    return measured_metrics('CMUMOSI', arrays['labels'], arrays['prediction'], arrays['base_prediction']), arrays


def checkpoint_payload(model, optimizer, identity, epoch, history, best):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    return dict(version=1, block={k: v.detach().cpu().clone() for k, v in model.block.state_dict().items()},
        optimizer=optimizer.state_dict(), identity=identity, epoch=epoch, history=history,
        best=best, rng=h.rng_state())


def restore_checkpoint(model, optimizer, saved, identity, limit):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if saved.get('version') != 1 or saved['identity'] != identity:
        raise ValueError('Immutable resume identity/version differs')
    epoch = saved['epoch']
    if type(epoch) is not int or not 0 < epoch <= limit:
        raise ValueError('Saved epoch outside authorized budget')
    validate_history(saved['history'], epoch)
    model.block.load_state_dict(saved['block'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    h.restore_rng(saved['rng'])
    return epoch, saved['history'], saved['best']


def validate_reference(reference, parent, config, mask_sha, rate):
    expected = dict(mode='signature', dataset='CMUMOSI', seed=66, evaluation_rate=rate,
        source_commit=NUMERICAL_SOURCE, smoke=False, stage2_epochs=EPOCHS, cyclic_training=True,
        mask_protocol='gcnet-random-mask-v1', effective_stage2_config=config,
        test_mask_sha256=mask_sha, parent=parent)
    expected = json.loads(json.dumps(expected))
    actual = reference['identity']
    if (reference.get('status') != 'complete' or reference.get('epoch') != EPOCHS
            or reference.get('total_parameters') != 281856
            or reference.get('parent_id') != parent['parent_id']
            or any(actual.get(key) != value for key, value in expected.items())):
        raise ValueError('Completed original numerical reference differs in own parent/config/mask/protocol')


def _verify_artifacts(output, best):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    for key, digest in (('checkpoint', 'checkpoint_sha256'), ('predictions', 'prediction_sha256')):
        path = (output / best[key]).resolve()
        if not path.is_relative_to(output.resolve()) or h.sha(path) != best[digest]:
            raise ValueError('Selected block/prediction artifact integrity differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('parallel', 'serial'), required=True)
    parser.add_argument('--rate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    rate = validate_scope(args.mode, args.rate)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from experiments.osram_three_block_joint_mosi_20261025.scheduler import gpu_inventory, guard_gpu
    host = gpu_inventory().get(args.gpu_uuid, {}).get('host_index')
    if host not in (0, 1, 2):
        raise ValueError('Only healthy physical GPU0/1/2 authorized')
    guard_gpu(args.gpu_uuid, host)
    import fcntl
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import _atomic_arrays
    from experiments.osram_signature_cyclic_multi_20261025 import prepare
    from . import blocks, model as model_module
    from .model import RecentNoAreaModel
    torch.set_num_threads(1)
    repair = repair_uuid_rng(torch)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output / 'SUMMARY.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if not args.resume or previous['status'] in ('complete', 'smoke_passed'):
            raise FileExistsError('Explicit resume of unfinished run required')
    elif args.resume:
        raise FileNotFoundError('Resume requires an existing run summary')
    tick = time.monotonic()
    summary = dict(status='preflight', stage='frozen', mode=args.mode, dataset='CMUMOSI', seed=66,
        fold=1, rate=rate, pid=os.getpid(), source_commit=args.source_commit,
        label='INTERNAL TEST-ORACLE SCREEN')
    try:
        reference_dir = REFERENCE_ROOT / f'CMUMOSI_66_{rate}'
        original = json.loads((reference_dir / 'SUMMARY.json').read_text())
        original_history = json.loads((reference_dir / 'history.json').read_text())
        validate_history(original_history, EPOCHS)
        numerical_hashes = original['identity']['source_hashes']
        if any(h.sha(prepare.ROOT / name) != digest for name, digest in numerical_hashes.items()):
            raise ValueError('Original immutable numerical source hashes differ')
        _verify_artifacts(reference_dir, original['best'])
        bundle, reader, parent = prepare.setup('CMUMOSI', 66, args.gpu_uuid, args.output, rate)
        bundle['cfg'] = replace(bundle['cfg'], train_rate_mode='cyclic', fixed_missing_rate=None,
            train_missing_rates=tuple(float(r) for r in RATES), epochs=EPOCHS, learning_rate=3e-4)
        rows = prepare.test_cache(bundle, reader, rate)
        with h.isolated_rng(66):
            model = RecentNoAreaModel(reader.parent, mode=args.mode, stage='frozen').to('cuda:0')
        parent_before = h.backbone_digest(model.parent)
        baseline, initial = evaluate(model, rows)
        np.testing.assert_array_equal(initial['prediction'], initial['base_prediction'])
        mask_sha = hashlib.sha256(initial['availability'].tobytes()).hexdigest()
        initial_sha = h.backbone_digest(model.block)
        validate_reference(original, parent, asdict(bundle['cfg']), mask_sha, rate)
        if args.mode == 'parallel' and (initial_sha != original['identity']['initial_block_sha256']
                or sum(p.numel() for p in model.block.parameters()) != 281856):
            raise ValueError('Parallel NoArea must preserve original block initialization and count')
        audit_root = Path(__file__).resolve().parents[2]
        from experiments.osram_signature_noarea_iemocap68_20261025 import block as shared_noarea
        sources = [Path(__file__).resolve(), Path(blocks.__file__).resolve(),
            Path(model_module.__file__).resolve(), Path(shared_noarea.__file__).resolve()]
        identity = dict(stage='frozen', mode=args.mode, ablation='degree2_no_area', dataset='CMUMOSI',
            seed=66, fold=1, evaluation_rate=rate, training_rates=list(RATES), cyclic_training=True,
            source_commit=args.source_commit, numerical_source_commit=NUMERICAL_SOURCE,
            numerical_source_root=str(prepare.ROOT), smoke=args.smoke, stage2_epochs=EPOCHS,
            parent=parent, parent_config=parent['config'], parent_state_sha256=parent_before,
            initial_block_sha256=initial_sha, effective_stage2_config=asdict(bundle['cfg']),
            learning_rate_formula='3e-4 if epoch0 < 100 else 3e-4 / 10',
            source_hashes={str(p.relative_to(audit_root)): h.sha(p) for p in sources},
            numerical_source_hashes=numerical_hashes, environment=bundle['environment'],
            data_files=bundle['data_files'], split=bundle['split'], seed_protocol=bundle['seed_protocol'],
            mask_protocol='gcnet-random-mask-v1', test_mask_sha256=mask_sha, rng_repair=repair,
            reference_signature=dict(output=str(reference_dir),
                summary_sha256=h.sha(reference_dir / 'SUMMARY.json'),
                history_sha256=h.sha(reference_dir / 'history.json'),
                best=original['best'], historical_lr_last50_literal=3e-5),
            task='original MSE including gold0; no auxiliary objective',
            selection='one own-rate TEST W-F1; earliest strict tie; ACC from identical predictions',
            architecture='S1 plus symmetric S2, strict-past observed window8; original CHM/Flat/LocalSkip retained',
            recovery='Only block/Adam/RNG/history; reload immutable own-rate Flat parent separately')
        # Canonical identity must match JSON roundtrips and CPU checkpoint metadata.
        identity = json.loads(json.dumps(identity))
        optimizer = build_optimizer(model)
        start, history, best = 0, [], None
        limit = 2 if args.smoke else EPOCHS
        if args.resume:
            saved = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            _verify_artifacts(args.output, saved['best'])
            start, history, best = restore_checkpoint(model, optimizer, saved, identity, limit)
            if h.sha(args.output / 'baseline.npz') != previous['baseline_artifact']['sha256']:
                raise ValueError('Saved original Flat baseline integrity differs')
        else:
            _atomic_arrays(args.output / 'baseline.npz', initial)
            reset_training_stochasticity(66, bundle['cfg'].fold)
        expected_history = original_history
        if args.smoke:
            from experiments.osram_signature_cyclic68_20261025.train import train_epoch as original_train_epoch
            from osram_four_block_wiring import SignatureBlock
            state = h.rng_state()
            try:
                with h.isolated_rng(66):
                    control = SignatureBlock().to('cuda:0')
                control_optimizer = torch.optim.Adam(control.parameters(), lr=3e-4, weight_decay=1e-5)
                reset_training_stochasticity(66, bundle['cfg'].fold)
                expected_history = [original_train_epoch(bundle, reader, control, control_optimizer, e, rate, True)
                    for e in range(limit)]
                del control_optimizer, control
            finally:
                h.restore_rng(state)
        for row in history:
            validate_epoch_trace(row, expected_history[row['epoch'] - 1])
        summary.update(status='running', identity=identity, parent_id=parent['parent_id'], baseline=baseline,
            baseline_artifact=dict(path='baseline.npz', sha256=h.sha(args.output / 'baseline.npz')),
            initial_real_batch_exact=True, environment=bundle['environment'], gpu=bundle['gpu'],
            total_parameters=sum(p.numel() for p in model.block.parameters()),
            full_parameters=sum(p.numel() for p in model.parameters()), epoch=start, best=best)
        h.write_json(args.output / 'config.json', identity)
        h.write_json(path, summary)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, limit):
            optimizer.param_groups[0]['lr'] = learning_rate(epoch)
            row = train_epoch(bundle, model, optimizer, epoch, rate, args.smoke)
            if row['live_scan_calls'] != [1] * row['optimizer_steps']:
                raise RuntimeError('Real training must execute exactly one live CHM _scan per batch')
            validate_epoch_trace(row, expected_history[epoch])
            score, arrays = evaluate(model, rows)
            if improves(score, None if best is None else best['metrics']):
                ckpt, pred = f'best_epoch_{epoch + 1}.pt', f'predictions_epoch_{epoch + 1}.npz'
                _atomic_arrays(args.output / pred, arrays)
                h.atomic_torch_save(args.output / ckpt, dict(version=1, block=model.block.state_dict(),
                    identity=identity, epoch=epoch + 1, parent_id=parent['parent_id']))
                best = dict(epoch=epoch + 1, metrics=score, checkpoint=ckpt, predictions=pred,
                    checkpoint_sha256=h.sha(args.output / ckpt), prediction_sha256=h.sha(args.output / pred))
            row['test'] = score
            history.append(row)
            h.atomic_torch_save(args.output / 'last.pt', checkpoint_payload(model, optimizer, identity,
                epoch + 1, history, best))
            summary.update(epoch=epoch + 1, best=best, seconds=time.monotonic() - tick,
                peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
            h.write_json(args.output / 'history.json', history)
            h.write_json(path, summary)
            print(json.dumps(dict(stage='frozen', mode=args.mode, dataset='CMUMOSI', seed=66,
                rate=rate, epoch=epoch + 1, current=score, best=best['metrics'], seconds=summary['seconds'])), flush=True)
        _verify_artifacts(args.output, best)
        saved = torch.load(args.output / best['checkpoint'], map_location='cpu', weights_only=False)
        if saved['identity'] != identity or saved['parent_id'] != parent['parent_id']:
            raise ValueError('Selected block identity/own-rate parent differs')
        model.block.load_state_dict(saved['block'], strict=True)
        score, replay = evaluate(model, rows)
        with np.load(args.output / best['predictions'], allow_pickle=False) as stored:
            if set(replay) != set(stored.files):
                raise ValueError('Selected array fields differ after restoration')
            for key in replay:
                np.testing.assert_array_equal(replay[key], stored[key])
        if score != best['metrics']:
            raise ValueError('Selected checkpoint metrics differ after restoration')
        if not args.smoke and {r for row in history for r in row['training_rates']} != {float(r) for r in RATES}:
            raise ValueError('Cyclic training did not cover all eight rates')
        if not any(row['nonzero_gradient_batches'] and row['updated_parameters'] for row in history):
            raise RuntimeError('New branch never received a nonzero gradient and parameter update')
        _check_frozen(model)
        if parent_before != h.backbone_digest(model.parent):
            raise RuntimeError('Frozen original Flat parent changed')
        if any(h.sha(audit_root / name) != digest for name, digest in identity['source_hashes'].items()):
            raise RuntimeError('Overlay source changed during training')
        if any(h.sha(prepare.ROOT / name) != digest for name, digest in numerical_hashes.items()):
            raise RuntimeError('Immutable original numerical source changed')
        for field in ('summary', 'history'):
            if h.sha(reference_dir / f'{field.upper() if field == "summary" else field}.json') != identity['reference_signature'][field + '_sha256']:
                raise RuntimeError('Original completed reference changed during training')
        summary.update(status='smoke_passed' if args.smoke else 'complete', seconds=time.monotonic() - tick,
            checks=dict(parent_unchanged=True, parent_grad_none=True, parent_eval=True, only_block_optimizer=True,
                checkpoint_replay_exact=True, selected_all_arrays_exact=True, cyclic_train_rates_verified=True,
                only_requested_test_rate=True, original_parent_and_test_mask_exact=True,
                original_training_mask_and_optimizer_trace_exact=True, original_initial_prediction_exact=True,
                finite_connected_block_gradients=True, nonzero_branch_gradient_and_update=True,
                no_auxiliary_loss=True, exact_learning_rate_expression=True, single_live_scan_verified=True))
        h.write_json(args.output / 'RESULT.json', summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
