"""Original Nested, trained alone on the protected MOSI66 own-rate Flat."""
import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from experiments.osram_recent_noarea_serial_20261025 import stage2 as shared

RATES, EPOCHS = shared.RATES, shared.EPOCHS
learning_rate = shared.learning_rate
PARAMETERS = 159235
METHOD = 'nested_gnn_rooted_evidence'


def validate_scope(rate, seed=66):
    return shared.validate_scope('serial', rate, seed=seed)


def starting_summary(previous, rate, source):
    summary = dict(previous or {})
    summary.update(status='preflight', dataset='CMUMOSI', seed=66, fold=1, rate=rate,
        stage='frozen', mode='nested', pid=os.getpid(), source_commit=source,
        label='INTERNAL TEST-ORACLE SCREEN')
    return summary


def costs(previous, elapsed, peak, rss):
    previous = previous or {}
    return dict(seconds=previous.get('seconds',0.)+elapsed,
        peak_memory_mib=max(previous.get('peak_memory_mib',0.),peak),
        peak_rss_gib=max(previous.get('peak_rss_gib',0.),rss))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    rate = validate_scope(args.rate)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from experiments.osram_three_block_joint_mosi_20261025.scheduler import gpu_inventory, guard_gpu
    host = gpu_inventory().get(args.gpu_uuid, {}).get('host_index')
    if host not in (0, 1, 2):
        raise ValueError('Only physical healthy GPU0/1/2 are authorized')
    guard_gpu(args.gpu_uuid, host)
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import _atomic_arrays
    from experiments.osram_signature_cyclic_multi_20261025 import prepare
    from . import model as model_module
    torch.set_num_threads(1)
    repair = repair_uuid_rng(torch)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'SUMMARY.json'
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is not None and (not args.resume or previous['status'] in ('complete', 'smoke_passed')):
        raise FileExistsError('Explicit resume of unfinished own-rate run required')
    if previous is None and args.resume:
        raise FileNotFoundError('No unfinished summary to resume')
    tick = time.monotonic()
    summary = starting_summary(previous, rate, args.source_commit)
    try:
        reference_dir = shared.REFERENCE_ROOT/f'CMUMOSI_66_{rate}'
        reference = json.loads((reference_dir/'SUMMARY.json').read_text())
        original_history = json.loads((reference_dir/'history.json').read_text())
        shared.validate_history(original_history, EPOCHS)
        numerical_hashes = reference['identity']['source_hashes']
        if any(h.sha(prepare.ROOT/name) != digest for name, digest in numerical_hashes.items()):
            raise ValueError('Pinned numerical source differs from completed comparison')
        shared._verify_artifacts(reference_dir, reference['best'])
        bundle, reader, parent = prepare.setup('CMUMOSI', 66, args.gpu_uuid, args.output, rate)
        bundle['cfg'] = replace(bundle['cfg'], train_rate_mode='cyclic', fixed_missing_rate=None,
            train_missing_rates=tuple(float(r) for r in RATES), epochs=EPOCHS, learning_rate=3e-4)
        rows = prepare.test_cache(bundle, reader, rate)
        with h.isolated_rng(66):
            model = model_module.FrozenNestedModel(reader.parent).to('cuda:0')
        parent_before = h.backbone_digest(model.parent)
        baseline, initial = shared.evaluate(model, rows)
        np.testing.assert_array_equal(initial['prediction'], initial['base_prediction'])
        mask_sha = hashlib.sha256(initial['availability'].tobytes()).hexdigest()
        shared.validate_reference(reference, parent, asdict(bundle['cfg']), mask_sha, rate)
        count = sum(p.numel() for p in model.block.parameters())
        if count != PARAMETERS:
            raise ValueError('Must use original unexpanded Nested with159235parameters')
        audit_root = Path(__file__).resolve().parents[2]
        sources = [Path(__file__).resolve(), Path(model_module.__file__).resolve(), Path(shared.__file__).resolve()]
        nested_sources = ('gcnet_missing_m3/meaningful_input.py',
            'gcnet_missing_m3/meaningful_input_new40.py', 'gcnet_missing_m3/meaningful_blocks_common.py',
            'gcnet_missing_m3/meaningful_new40_structure.py', 'gcnet_missing_m3/meaningful_new40_registry.py')
        identity = dict(stage='frozen', mode='nested', method=METHOD, dataset='CMUMOSI', seed=66,
            fold=1, evaluation_rate=rate, training_rates=list(RATES), cyclic_training=True,
            source_commit=args.source_commit, numerical_source_commit=shared.NUMERICAL_SOURCE,
            numerical_source_root=str(prepare.ROOT), smoke=args.smoke, stage2_epochs=EPOCHS,
            parent=parent, parent_state_sha256=parent_before, initial_block_sha256=h.backbone_digest(model.block),
            effective_stage2_config=asdict(bundle['cfg']), learning_rate_formula='3e-4 if epoch0 < 100 else 3e-4 / 10',
            source_hashes={str(p.relative_to(audit_root)):h.sha(p) for p in sources},
            numerical_source_hashes=numerical_hashes, environment=bundle['environment'],
            nested_source_hashes={name:h.sha(prepare.ROOT/name) for name in nested_sources},
            data_files=bundle['data_files'], split=bundle['split'], seed_protocol=bundle['seed_protocol'],
            mask_protocol='gcnet-random-mask-v1', test_mask_sha256=mask_sha, rng_repair=repair,
            reference_signature=dict(output=str(reference_dir), summary_sha256=h.sha(reference_dir/'SUMMARY.json'),
                history_sha256=h.sha(reference_dir/'history.json')),
            task='original MSE including gold0; no auxiliary loss',
            selection='own-rate TEST W-F1; strict improvement/earliest tie; ACC same predictions',
            architecture='original Nested dim64/depth3/MLP64/heads8; raw LocalSkip unchanged',
            recovery='block/Adam/RNG/history plus immutable own-rate original Flat reference')
        identity = json.loads(json.dumps(identity))
        optimizer = shared.build_optimizer(model)
        start, history, best = 0, [], None
        limit = 2 if args.smoke else EPOCHS
        if args.resume:
            saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            shared._verify_artifacts(args.output, saved['best'])
            start, history, best = shared.restore_checkpoint(model, optimizer, saved, identity, limit)
            if h.sha(args.output/'baseline.npz') != previous['baseline_artifact']['sha256']:
                raise ValueError('Baseline changed during resume')
        else:
            _atomic_arrays(args.output/'baseline.npz', initial)
            reset_training_stochasticity(66, bundle['cfg'].fold)
        expected_history = original_history
        if args.smoke:
            # Compare one-batch smoke traces to the same audited reference trainer.
            from experiments.osram_signature_cyclic68_20261025.train import train_epoch as reference_train
            from osram_four_block_wiring import SignatureBlock
            state = h.rng_state()
            try:
                with h.isolated_rng(66): control = SignatureBlock().to('cuda:0')
                control_optimizer = torch.optim.Adam(control.parameters(), lr=3e-4, weight_decay=1e-5)
                reset_training_stochasticity(66, bundle['cfg'].fold)
                expected_history = [reference_train(bundle, reader, control, control_optimizer, e, rate, True)
                    for e in range(limit)]
                del control_optimizer, control
            finally:
                h.restore_rng(state)
        for row in history:
            shared.validate_epoch_trace(row, expected_history[row['epoch']-1])
        summary.update(status='running', identity=identity, parent_id=parent['parent_id'], baseline=baseline,
            baseline_artifact=dict(path='baseline.npz', sha256=h.sha(args.output/'baseline.npz')),
            environment=bundle['environment'], gpu=bundle['gpu'], total_parameters=count,
            full_parameters=sum(p.numel() for p in model.parameters()), epoch=start, best=best)
        h.write_json(args.output/'config.json', identity)
        h.write_json(path, summary)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, limit):
            optimizer.param_groups[0]['lr'] = learning_rate(epoch)
            row = shared.train_epoch(bundle, model, optimizer, epoch, rate, args.smoke)
            if row['live_scan_calls'] != [1]*row['optimizer_steps']:
                raise RuntimeError('Require exactly one original live CHM scan')
            shared.validate_epoch_trace(row, expected_history[epoch])
            score, arrays = shared.evaluate(model, rows)
            if shared.improves(score, None if best is None else best['metrics']):
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
                torch.cuda.max_memory_allocated()/2**20,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
            h.write_json(args.output/'history.json', history)
            h.write_json(path, summary)
            print(json.dumps(dict(rate=rate, epoch=epoch+1, current=score, best=best['metrics'])), flush=True)
        shared._verify_artifacts(args.output, best)
        saved = torch.load(args.output/best['checkpoint'], map_location='cpu', weights_only=False)
        if saved['identity'] != identity or saved['parent_id'] != parent['parent_id']:
            raise ValueError('Selected identity/parent mismatch')
        model.block.load_state_dict(saved['block'], strict=True)
        score, replay = shared.evaluate(model, rows)
        with np.load(args.output/best['predictions'], allow_pickle=False) as stored:
            if set(replay) != set(stored.files): raise ValueError('Selected array fields differ')
            for key in replay: np.testing.assert_array_equal(replay[key], stored[key])
        if score != best['metrics']: raise ValueError('Selected metrics do not replay')
        if not any(row['nonzero_gradient_batches'] and row['updated_parameters'] for row in history):
            raise RuntimeError('Nested has no actual task gradient/update')
        if not args.smoke and {r for row in history for r in row['training_rates']} != {float(r) for r in RATES}:
            raise RuntimeError('Missing a cyclic training rate')
        shared._check_frozen(model)
        if h.backbone_digest(model.parent) != parent_before: raise RuntimeError('Frozen parent changed')
        if any(h.sha(audit_root/name) != digest for name,digest in identity['source_hashes'].items()):
            raise RuntimeError('Overlay source changed')
        if any(h.sha(prepare.ROOT/name) != digest for name,digest in numerical_hashes.items()):
            raise RuntimeError('Original numerical source changed')
        if any(h.sha(prepare.ROOT/name) != digest for name,digest in identity['nested_source_hashes'].items()):
            raise RuntimeError('Original Nested implementation changed')
        if (h.sha(reference_dir/'SUMMARY.json') != identity['reference_signature']['summary_sha256']
                or h.sha(reference_dir/'history.json') != identity['reference_signature']['history_sha256']):
            raise RuntimeError('Completed training reference changed')
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
            seconds=(previous or {}).get('seconds',0.)+time.monotonic()-tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
