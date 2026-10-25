"""Original159235-parameter Nested on own-rate frozen MOSEI/MELD/IEMOCAP parents."""
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
    learning_rate, measured_metrics, numerical_root, numerical_source, reference_dir,
    selection_metric, validate_epoch_trace, validate_reference, validate_scope)


def starting_summary(previous, dataset, seed, fold, rate, source):
    summary = dict(previous or {})
    summary.pop('error', None)
    summary.update(status='preflight', dataset=dataset, seed=seed, fold=fold, rate=rate,
        stage='frozen', mode='nested', pid=os.getpid(), source_commit=source,
        label='INTERNAL TEST-ORACLE SCREEN')
    return summary


def costs(previous, elapsed, peak, rss):
    previous = previous or {}
    return dict(seconds=previous.get('seconds', 0.) + elapsed,
        peak_memory_mib=max(previous.get('peak_memory_mib', 0.), peak),
        peak_rss_gib=max(previous.get('peak_rss_gib', 0.), rss))


def configure_environment(dataset, prepare, output):
    """The trainer snapshots config paths at import, including sentiment paths."""
    if dataset == 'IEMOCAPSix':
        prepare.configure_environment(output)
    else:
        prepare.configure_environment(dataset, output)
    if dataset == 'CMUMOSEI':
        manifest_module = importlib.import_module('experiments.osram_mosei_gcnet_router_20261020.worker')
        manifest, _ = manifest_module.load_manifest()
        os.environ.update(GCNET_DATASET_ROOT=manifest['dataset_root'],
            GCNET_CACHE_ROOT=str(Path(output).parent/'feature_cache'/'CMUMOSEI'))
        os.environ.pop('GCNET_MELD_PICKLE', None)


def evaluate(dataset, model, rows):
    """Replay saved own-rate frames; retain all class logits and metadata."""
    import numpy as np
    import torch
    model.eval()
    pieces = []
    device = next(model.block.parameters()).device
    with torch.no_grad():
        for row in rows:
            frame = row['frame'].to(device)
            prediction = model.predict_frame(frame)
            valid = frame.valid.T
            prediction = prediction.transpose(0, 1)[valid].cpu().numpy()
            base = frame.prediction.transpose(0, 1)[valid].cpu().numpy()
            if dataset == 'CMUMOSEI':
                prediction, base = prediction.reshape(-1), base.reshape(-1)
            pieces.append(dict(row['metadata'], availability=row['availability'],
                               prediction=prediction, base_prediction=base))
    if not pieces:
        raise ValueError('Empty own-rate test cache')
    arrays = {key: np.concatenate([row[key] for row in pieces]) for key in pieces[0]}
    return measured_metrics(dataset, arrays['labels'], arrays['prediction'], arrays['base_prediction']), arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--seed', type=int, default=66)
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
    # Configure dataset globals before importing any trainer/config or model.
    family = 'iemocap' if args.dataset == 'IEMOCAPSix' else 'multi'
    prepare_package = ('osram_signature_iemocap_cyclic_20261025' if family == 'iemocap'
                       else 'osram_signature_cyclic_multi_20261025')
    prepare = importlib.import_module(f'experiments.{prepare_package}.prepare')
    configure_environment(args.dataset, prepare, args.output)
    numeric = numerical_root(args.dataset)
    if prepare.ROOT.resolve() != numeric.resolve() or Path.cwd().resolve() != numeric.resolve():
        raise ValueError('Run from the dataset-specific pinned numerical source, using only an immutable overlay')
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
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output / 'SUMMARY.json'
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is not None and (not args.resume or previous['status'] in ('complete', 'smoke_passed')):
        raise FileExistsError('Explicit resume of unfinished own-rate run required')
    if previous is None and args.resume:
        raise FileNotFoundError('No unfinished summary to resume')
    tick = time.monotonic()
    summary = starting_summary(previous, args.dataset, args.seed, args.fold, rate, args.source_commit)
    try:
        reference_path = reference_dir(args.dataset, args.fold, rate)
        reference = json.loads((reference_path / 'SUMMARY.json').read_text())
        original_history = json.loads((reference_path / 'history.json').read_text())
        epochs = budget(args.dataset)
        shared.validate_history(original_history, epochs)
        numerical_hashes = reference['identity']['source_hashes']
        if any(h.sha(numeric / name) != digest for name, digest in numerical_hashes.items()):
            raise ValueError('Pinned numerical source differs from completed Signature')
        shared._verify_artifacts(reference_path, reference['best'])
        if family == 'iemocap':
            bundle, reader, parent = prepare.setup(args.seed, args.fold, args.gpu_uuid, args.output, rate)
        else:
            bundle, reader, parent = prepare.setup(args.dataset, args.seed, args.gpu_uuid, args.output, rate)
        bundle['cfg'] = replace(bundle['cfg'], train_rate_mode='cyclic', fixed_missing_rate=None,
            train_missing_rates=tuple(float(r) for r in RATES), epochs=epochs, learning_rate=3e-4)
        rows = prepare.test_cache(bundle, reader, rate)
        with h.isolated_rng(args.seed):
            model = model_module.FrozenNestedModel(reader.parent).to('cuda:0')
        parent_before = h.backbone_digest(model.parent)
        baseline, initial = evaluate(args.dataset, model, rows)
        np.testing.assert_array_equal(initial['prediction'], initial['base_prediction'])
        mask_sha = hashlib.sha256(initial['availability'].tobytes()).hexdigest()
        validate_reference(reference, parent, asdict(bundle['cfg']), mask_sha, args.dataset, args.fold, rate)
        if baseline != reference['baseline']:
            raise ValueError('Initial frozen parent metrics differ from Signature baseline')
        count = sum(p.numel() for p in model.block.parameters())
        if count != PARAMETERS:
            raise ValueError('Require unchanged original Nested with159235parameters')
        audit_root = Path(__file__).resolve().parents[2]
        sources = [Path(__file__).resolve(), Path(__file__).with_name('protocol.py'),
            Path(model_module.__file__).resolve(), Path(shared.__file__).resolve()]
        nested_names = ('meaningful_input', 'meaningful_input_new40', 'meaningful_blocks_common',
                        'meaningful_new40_structure', 'meaningful_new40_registry')
        nested_files = {str(Path(importlib.import_module('gcnet_missing_m3.' + name).__file__).resolve()):
                        h.sha(Path(importlib.import_module('gcnet_missing_m3.' + name).__file__).resolve())
                        for name in nested_names}
        if any(not Path(name).is_relative_to(numeric.resolve()) for name in nested_files):
            raise ValueError('Original Nested modules must resolve from pinned numerical source')
        identity = dict(stage='frozen', mode='nested', method=METHOD, dataset=args.dataset,
            seed=args.seed, fold=args.fold, evaluation_rate=rate, training_rates=list(RATES),
            cyclic_training=True, source_commit=args.source_commit,
            numerical_source_commit=numerical_source(args.dataset), numerical_source_root=str(numeric),
            smoke=args.smoke, stage2_epochs=epochs, parent=parent, parent_state_sha256=parent_before,
            initial_block_sha256=h.backbone_digest(model.block), effective_stage2_config=asdict(bundle['cfg']),
            learning_rate_formula='3e-4 constant', source_hashes={str(p.relative_to(audit_root)): h.sha(p) for p in sources},
            numerical_source_hashes=numerical_hashes, nested_source_hashes=nested_files,
            environment=bundle['environment'], data_files=bundle['data_files'], split=bundle['split'],
            seed_protocol=bundle['seed_protocol'], mask_protocol='gcnet-random-mask-v1',
            test_mask_sha256=mask_sha, rng_repair=repair,
            reference_signature=dict(output=str(reference_path), summary_sha256=h.sha(reference_path/'SUMMARY.json'),
                history_sha256=h.sha(reference_path/'history.json')),
            task='original MSE including gold0; no auxiliary loss' if args.dataset == 'CMUMOSEI' else 'original CE; no auxiliary loss',
            selection=f'own-rate TEST {selection_metric(args.dataset)}; strict improvement/earliest tie',
            architecture='original Nested dim64/depth3/MLP64/heads8; raw LocalSkip unchanged',
            recovery='block/Adam/RNG/history plus immutable own-rate original Flat reference')
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
        expected_history = original_history
        if args.smoke:
            from experiments.osram_signature_cyclic68_20261025.train import train_epoch as reference_train
            from osram_four_block_wiring import SignatureBlock
            state = h.rng_state()
            try:
                with h.isolated_rng(args.seed):
                    control = SignatureBlock().to('cuda:0')
                control_optimizer = torch.optim.Adam(control.parameters(), lr=3e-4, weight_decay=1e-5)
                reset_training_stochasticity(args.seed, args.fold)
                expected_history = [reference_train(bundle, reader, control, control_optimizer, e, rate, True)
                    for e in range(limit)]
                del control_optimizer, control
            finally:
                h.restore_rng(state)
        for row in history:
            validate_epoch_trace(args.dataset, row, expected_history[row['epoch']-1])
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
            if args.dataset != 'CMUMOSEI':
                row['task_CE'] = row.pop('task_MSE')
            if row['live_scan_calls'] != [1]*row['optimizer_steps']:
                raise RuntimeError('Require exactly one original live CHM scan')
            validate_epoch_trace(args.dataset, row, expected_history[epoch])
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
            print(json.dumps(dict(dataset=args.dataset, fold=args.fold, rate=rate, epoch=epoch+1,
                                 current=score, best=best['metrics'])), flush=True)
        shared._verify_artifacts(args.output, best)
        saved = torch.load(args.output/best['checkpoint'], map_location='cpu', weights_only=False)
        if saved['identity'] != identity or saved['parent_id'] != parent['parent_id']:
            raise ValueError('Selected identity/parent mismatch')
        model.block.load_state_dict(saved['block'], strict=True)
        score, replay = evaluate(args.dataset, model, rows)
        with np.load(args.output/best['predictions'], allow_pickle=False) as stored:
            if set(replay) != set(stored.files):
                raise ValueError('Selected array fields differ')
            for key in replay:
                np.testing.assert_array_equal(replay[key], stored[key])
        if score != best['metrics']:
            raise ValueError('Selected metrics do not replay')
        if not any(row['nonzero_gradient_batches'] and row['updated_parameters'] for row in history):
            raise RuntimeError('Nested has no actual task gradient/update')
        if not args.smoke and {r for row in history for r in row['training_rates']} != {float(r) for r in RATES}:
            raise RuntimeError('Missing a cyclic training rate')
        shared._check_frozen(model)
        if h.backbone_digest(model.parent) != parent_before:
            raise RuntimeError('Frozen parent changed')
        if any(h.sha(audit_root/name) != digest for name, digest in identity['source_hashes'].items()):
            raise RuntimeError('Overlay source changed')
        if any(h.sha(numeric/name) != digest for name, digest in numerical_hashes.items()):
            raise RuntimeError('Original numerical source changed')
        if any(h.sha(Path(name)) != digest for name, digest in nested_files.items()):
            raise RuntimeError('Original Nested implementation changed')
        if (h.sha(reference_path/'SUMMARY.json') != identity['reference_signature']['summary_sha256']
                or h.sha(reference_path/'history.json') != identity['reference_signature']['history_sha256']):
            raise RuntimeError('Completed Signature reference changed')
        summary.update(status='smoke_passed' if args.smoke else 'complete',
            **costs(previous, time.monotonic()-tick, torch.cuda.max_memory_allocated()/2**20,
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20),
            checks=dict(parent_unchanged=True, parent_grad_none=True, parent_eval=True, only_block_optimizer=True,
                checkpoint_replay_exact=True, selected_all_arrays_exact=True, cyclic_train_rates_verified=True,
                only_requested_test_rate=True, original_parent_and_test_mask_exact=True,
                original_training_mask_and_optimizer_trace_exact=True, original_initial_prediction_exact=True,
                finite_connected_block_gradients=True, nonzero_branch_gradient_and_update=True,
                no_auxiliary_loss=True, exact_learning_rate_expression=True, single_live_scan_verified=True))
        h.write_json(args.output/'RESULT.json', summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error),
            seconds=(previous or {}).get('seconds', 0.)+time.monotonic()-tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
