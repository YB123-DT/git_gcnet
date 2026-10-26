"""Own-seed frozen original Nested replication; no parent or model changes."""
import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import time

from .protocol import (DATASETS, RATES, PARAMETERS, METHOD, budget, improves,
    learning_rate, measured_metrics, source_root, source_commit, reference_dir,
    selection_metric, validate_epoch_trace, validate_reference, validate_scope)
from experiments.osram_frozen_nested_cross_20261025.run import starting_summary, costs


def load_reference(dataset, seed, fold, rate):
    path = reference_dir(dataset, seed, fold, rate)
    if path is None:
        return None
    summary = json.loads((path/'SUMMARY.json').read_text())
    history = json.loads((path/'history.json').read_text())
    if summary['status'] != 'complete' or summary['epoch'] != budget(dataset):
        raise ValueError('Existing same-seed reference is unfinished')
    if [r['epoch'] for r in history] != list(range(1, budget(dataset)+1)):
        raise ValueError('Existing same-seed reference history is incomplete')
    return path, summary, history


def evaluate(dataset, model, rows):
    import numpy as np
    import torch
    model.eval()
    pieces = []
    device = next(model.block.parameters()).device
    with torch.no_grad():
        for row in rows:
            frame = row['frame'].to(device)
            predicted = model.predict_frame(frame).transpose(0, 1)[frame.valid.T].cpu().numpy()
            original = frame.prediction.transpose(0, 1)[frame.valid.T].cpu().numpy()
            if dataset in ('CMUMOSI', 'CMUMOSEI'):
                predicted, original = predicted.reshape(-1), original.reshape(-1)
            pieces.append(dict(row['metadata'], availability=row['availability'],
                prediction=predicted, base_prediction=original))
    if not pieces:
        raise ValueError('Empty own-rate test cache')
    arrays = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    return measured_metrics(dataset, arrays['labels'], arrays['prediction'], arrays['base_prediction']), arrays


def numerical_hashes(root, sha):
    """Fingerprint actual numerical imports, independent of another seed's trace."""
    files = []
    for folder in ('gcnet_missing_m3', 'gcnet_modality_jepa', 'osram_four_block_wiring'):
        files.extend((root/folder).glob('*.py'))
    if not files:
        raise ValueError('Missing original numerical source')
    return {str(p.relative_to(root)): sha(p) for p in sorted(files)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--rate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    rate = validate_scope(args.dataset, args.seed, args.fold, args.rate)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from . import prepare
    prepare.configure_environment(args.dataset, args.seed, args.fold, args.output)
    numeric = source_root(args.dataset, args.seed)
    if Path.cwd().resolve() != numeric.resolve():
        raise ValueError('Require own-seed pinned numerical working directory')
    from experiments.osram_three_block_joint_mosi_20261025.scheduler import gpu_inventory, guard_gpu
    host = gpu_inventory().get(args.gpu_uuid, {}).get('host_index')
    if host not in (0, 1, 2):
        raise ValueError('Only physical healthy GPU0/1/2 authorized')
    guard_gpu(args.gpu_uuid, host)
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import _atomic_arrays
    from experiments.osram_recent_noarea_serial_20261025 import stage2 as shared
    from experiments.osram_frozen_nested_gcnet_20261025 import model as model_module
    torch.set_num_threads(1)
    repair = repair_uuid_rng(torch)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'SUMMARY.json'
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is not None and (not args.resume or previous['status'] in ('complete', 'smoke_passed')):
        raise FileExistsError('Explicit full-state resume of unfinished run required')
    if previous is None and args.resume:
        raise FileNotFoundError('No unfinished summary to resume')
    tick = time.monotonic()
    summary = starting_summary(previous, args.dataset, args.seed, args.fold, rate, args.source_commit)
    try:
        reference = load_reference(args.dataset, args.seed, args.fold, rate)
        bundle, reader, parent = prepare.setup(args.dataset, args.seed, args.fold,
            args.gpu_uuid, args.output, rate)
        epochs = budget(args.dataset)
        bundle['cfg'] = replace(bundle['cfg'], train_rate_mode='cyclic', fixed_missing_rate=None,
            train_missing_rates=tuple(float(r) for r in RATES), epochs=epochs, learning_rate=3e-4)
        rows = prepare.test_cache(bundle, reader, rate)
        with h.isolated_rng(args.seed):
            model = model_module.FrozenNestedModel(reader.parent).to('cuda:0')
        parent_before = h.backbone_digest(model.parent)
        baseline, initial = evaluate(args.dataset, model, rows)
        np.testing.assert_array_equal(initial['prediction'], initial['base_prediction'])
        mask_sha = hashlib.sha256(initial['availability'].tobytes()).hexdigest()
        reference_proof = None
        if reference is not None:
            reference_path, reference_summary, original_history = reference
            shared._verify_artifacts(reference_path, reference_summary['best'])
            validate_reference(reference_summary, parent, asdict(bundle['cfg']), mask_sha,
                args.dataset, args.seed, args.fold, rate)
            if baseline != reference_summary['baseline']:
                raise ValueError('Initial parent metrics differ from same-seed completed reference')
            reference_proof = dict(output=str(reference_path),
                summary_sha256=h.sha(reference_path/'SUMMARY.json'),
                history_sha256=h.sha(reference_path/'history.json'))
        count = sum(p.numel() for p in model.block.parameters())
        if count != PARAMETERS:
            raise ValueError('Require unchanged original Nested159235 parameters')
        audit_root = Path(__file__).resolve().parents[2]
        sources = list(Path(__file__).parent.glob('*.py')) + [
            Path(model_module.__file__).resolve(), Path(shared.__file__).resolve()]
        source_hashes = {str(p.relative_to(audit_root)): h.sha(p) for p in sources}
        numeric_hashes = numerical_hashes(numeric, h.sha)
        nested_names = ('meaningful_input', 'meaningful_input_new40', 'meaningful_blocks_common',
                        'meaningful_new40_structure', 'meaningful_new40_registry')
        nested_files = {str(Path(importlib.import_module('gcnet_missing_m3.'+name).__file__).resolve()):
            h.sha(Path(importlib.import_module('gcnet_missing_m3.'+name).__file__).resolve()) for name in nested_names}
        if any(not Path(name).is_relative_to(numeric.resolve()) for name in nested_files):
            raise ValueError('Nested imports must come from pinned numerical source')
        identity = dict(stage='frozen', mode='nested', method=METHOD, dataset=args.dataset,
            seed=args.seed, fold=args.fold, evaluation_rate=rate, training_rates=list(RATES),
            cyclic_training=True, source_commit=args.source_commit,
            numerical_source_commit=source_commit(args.dataset, args.seed), numerical_source_root=str(numeric),
            smoke=args.smoke, stage2_epochs=epochs, parent=parent, parent_state_sha256=parent_before,
            initial_block_sha256=h.backbone_digest(model.block), effective_stage2_config=asdict(bundle['cfg']),
            learning_rate_formula='3e-4 first100 then 3e-4/10' if args.dataset == 'CMUMOSI' else '3e-4 constant',
            source_hashes=source_hashes, numerical_source_hashes=numeric_hashes,
            nested_source_hashes=nested_files, environment=bundle['environment'], data_files=bundle['data_files'],
            split=bundle['split'], seed_protocol=bundle['seed_protocol'],
            mask_protocol='gcnet-random-mask-v1', test_mask_sha256=mask_sha, rng_repair=repair,
            reference_signature=reference_proof,
            task='original MSE including gold0; no auxiliary loss' if args.dataset in ('CMUMOSI','CMUMOSEI')
                else 'original CE; no auxiliary loss',
            selection=f'own-rate TEST {selection_metric(args.dataset)}; strict improvement/earliest tie',
            architecture='original Nested dim64/depth3/MLP64/heads8; raw LocalSkip unchanged',
            recovery='block/Adam/RNG/history plus immutable own-seed own-rate original Flat')
        identity = json.loads(json.dumps(identity))
        optimizer = shared.build_optimizer(model)
        start, history, best = 0, [], None
        limit = 2 if args.smoke else epochs
        if args.resume:
            saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            shared._verify_artifacts(args.output, saved['best'])
            start, history, best = shared.restore_checkpoint(model, optimizer, saved, identity, limit)
            if h.sha(args.output/'baseline.npz') != previous['baseline_artifact']['sha256']:
                raise ValueError('Baseline changed during resume')
        else:
            _atomic_arrays(args.output/'baseline.npz', initial)
            reset_training_stochasticity(args.seed, args.fold)
        expected_history = reference[2] if reference is not None else None
        if args.smoke:
            # A separate RNG-restored sampler control, never another seed's trace.
            from experiments.osram_signature_cyclic68_20261025.train import train_epoch as control_train
            from osram_four_block_wiring import SignatureBlock
            state = h.rng_state()
            try:
                with h.isolated_rng(args.seed):
                    control = SignatureBlock().to('cuda:0')
                control_optimizer = torch.optim.Adam(control.parameters(), lr=3e-4, weight_decay=1e-5)
                reset_training_stochasticity(args.seed, args.fold)
                expected_history = [control_train(bundle, reader, control, control_optimizer, e, rate, True)
                    for e in range(limit)]
                del control_optimizer, control
            finally:
                h.restore_rng(state)
        for row in history:
            validate_epoch_trace(args.dataset, row, None if expected_history is None else expected_history[row['epoch']-1])
        summary.update(status='running', identity=identity, parent_id=parent['parent_id'], baseline=baseline,
            baseline_artifact=dict(path='baseline.npz', sha256=h.sha(args.output/'baseline.npz')),
            environment=bundle['environment'], gpu=bundle['gpu'], total_parameters=count,
            full_parameters=sum(p.numel() for p in model.parameters()), epoch=start, best=best)
        h.write_json(args.output/'config.json', identity)
        h.write_json(path, summary)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, limit):
            optimizer.param_groups[0]['lr'] = learning_rate(args.dataset, epoch)
            row = shared.train_epoch(bundle, model, optimizer, epoch, rate, args.smoke)
            if args.dataset not in ('CMUMOSI', 'CMUMOSEI'):
                row['task_CE'] = row.pop('task_MSE')
            if row['live_scan_calls'] != [1]*row['optimizer_steps']:
                raise RuntimeError('Require exactly one original CHM scan')
            validate_epoch_trace(args.dataset, row, None if expected_history is None else expected_history[epoch])
            score, arrays = evaluate(args.dataset, model, rows)
            if improves(args.dataset, score, None if best is None else best['metrics']):
                checkpoint, prediction = f'best_epoch_{epoch+1}.pt', f'predictions_epoch_{epoch+1}.npz'
                _atomic_arrays(args.output/prediction, arrays)
                h.atomic_torch_save(args.output/checkpoint, dict(version=1, block=model.block.state_dict(),
                    identity=identity, epoch=epoch+1, parent_id=parent['parent_id']))
                best = dict(epoch=epoch+1, metrics=score, checkpoint=checkpoint, predictions=prediction,
                    checkpoint_sha256=h.sha(args.output/checkpoint), prediction_sha256=h.sha(args.output/prediction))
            row['test'] = score
            history.append(row)
            h.atomic_torch_save(args.output/'last.pt', shared.checkpoint_payload(model, optimizer, identity,
                epoch+1, history, best))
            summary.update(epoch=epoch+1, best=best, **costs(previous, time.monotonic()-tick,
                torch.cuda.max_memory_allocated()/2**20, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
            h.write_json(args.output/'history.json', history)
            h.write_json(path, summary)
            print(json.dumps(dict(dataset=args.dataset, seed=args.seed, fold=args.fold, rate=rate,
                epoch=epoch+1, current=score, best=best['metrics'])), flush=True)
        shared._verify_artifacts(args.output, best)
        selected = torch.load(args.output/best['checkpoint'], map_location='cpu', weights_only=False)
        if selected['identity'] != identity or selected['parent_id'] != parent['parent_id']:
            raise ValueError('Selected identity/parent mismatch')
        model.block.load_state_dict(selected['block'], strict=True)
        score, replay = evaluate(args.dataset, model, rows)
        with np.load(args.output/best['predictions'], allow_pickle=False) as stored:
            if set(replay) != set(stored.files):
                raise ValueError('Selected array keys differ')
            for key in replay:
                np.testing.assert_array_equal(replay[key], stored[key])
        if score != best['metrics']:
            raise ValueError('Selected metrics do not replay')
        if not any(r['nonzero_gradient_batches'] and r['updated_parameters'] for r in history):
            raise RuntimeError('No actual Nested gradient/update')
        if not args.smoke and {r for row in history for r in row['training_rates']} != {float(r) for r in RATES}:
            raise RuntimeError('Incomplete cyclic rate coverage')
        shared._check_frozen(model)
        if h.backbone_digest(model.parent) != parent_before:
            raise RuntimeError('Frozen parent changed')
        if any(h.sha(audit_root/p) != digest for p, digest in source_hashes.items()):
            raise RuntimeError('Immutable overlay changed')
        if any(h.sha(numeric/p) != digest for p, digest in numeric_hashes.items()):
            raise RuntimeError('Pinned numerical source changed')
        if reference_proof is not None and (
            h.sha(reference[0]/'SUMMARY.json') != reference_proof['summary_sha256'] or
            h.sha(reference[0]/'history.json') != reference_proof['history_sha256']):
            raise RuntimeError('Completed same-seed reference changed')
        summary.update(status='smoke_passed' if args.smoke else 'complete',
            **costs(previous, time.monotonic()-tick, torch.cuda.max_memory_allocated()/2**20,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20),
            checks=dict(parent_unchanged=True, parent_grad_none=True, parent_eval=True,
                only_block_optimizer=True, checkpoint_replay_exact=True, selected_all_arrays_exact=True,
                cyclic_train_rates_verified=True, only_requested_test_rate=True,
                original_parent_and_test_mask_exact=True, original_initial_prediction_exact=True,
                training_trace_validated=True, finite_connected_block_gradients=True,
                nonzero_branch_gradient_and_update=True, no_auxiliary_loss=True,
                exact_learning_rate_expression=True, single_live_scan_verified=True))
        h.write_json(args.output/'RESULT.json', summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error),
            seconds=(previous or {}).get('seconds', 0.)+time.monotonic()-tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
