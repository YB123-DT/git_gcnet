"""Matched seed68 IEMOCAP-6 training control without antisymmetric level two."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from experiments.osram_signature_iemocap_cyclic_20261025.run import (
    budget, evaluate, improves, learning_rate, measured_metrics,
)

NUMERICAL_SOURCE = '21e46f2ecd092345818659694db8ff372672c35e'
REFERENCE_ROOT = Path('/data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/execution_v1/train')
DEFINITION = 'Keep S1 and sym(S2)=(S2+S2.T)/2; remove only antisymmetric S2; no parameter or budget change'


def validate_scope(seed, fold):
    if type(seed) is not int or seed != 68 or type(fold) is not int or fold not in range(1, 6):
        raise ValueError('Only IEMOCAP-6 seed68 and folds1..5 are authorized')


def validate_history(history, epochs):
    if [row['epoch'] for row in history] != list(range(1, epochs + 1)):
        raise ValueError('Require complete contiguous training history')


def validate_reference(reference, parent, config, initial_sha, mask_sha, fold, rate):
    identity = reference['identity']
    expected = dict(mode='signature', dataset='IEMOCAPSix', seed=68, fold=fold,
        evaluation_rate=rate, source_commit=NUMERICAL_SOURCE, smoke=False,
        stage2_epochs=budget(), cyclic_training=True, mask_protocol='gcnet-random-mask-v1',
        initial_block_sha256=initial_sha, effective_stage2_config=config,
        test_mask_sha256=mask_sha, parent=parent)
    # The completed reference is JSON, while TrainConfig/asdict retains tuples.
    # Compare the exact persisted representation, including all nested fields.
    identity = json.loads(json.dumps(identity))
    expected = json.loads(json.dumps(expected))
    if (reference.get('status') != 'complete' or reference.get('epoch') != budget()
            or reference.get('total_parameters') != 281856
            or reference.get('parent_id') != parent['parent_id']
            or any(identity.get(key) != value for key, value in expected.items())):
        raise ValueError('Completed original Signature reference differs in parent, initialization, mask or protocol')


def validate_epoch_trace(row, original):
    for key in ('epoch', 'optimizer_steps', 'training_rates', 'mask_trace_sha256', 'N'):
        if row.get(key) != original.get(key):
            raise ValueError('Training sampling/update trace differs from original Signature: ' + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--rate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    validate_scope(args.seed, args.fold)
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
    from experiments.osram_signature_iemocap_cyclic_20261025 import prepare
    prepare.configure_environment(args.output)
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import _atomic_arrays
    from experiments.osram_signature_cyclic68_20261025.train import train_epoch
    from experiments.osram_signature_iemocap_cyclic_20261025.summarize import selected_row
    from osram_four_block_wiring import SignatureBlock
    from .block import NoAreaSignatureBlock
    rate = prepare.validate_rate(args.rate)
    torch.set_num_threads(1)
    repair = repair_uuid_rng(torch)
    if (args.output / 'SUMMARY.json').exists():
        previous = json.loads((args.output / 'SUMMARY.json').read_text())
        if not args.resume or previous['status'] in ('complete', 'smoke_passed'):
            raise FileExistsError('Explicit resume of unfinished run required')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tick, path = time.monotonic(), args.output / 'SUMMARY.json'
    summary = dict(status='preflight', source_commit=args.source_commit, mode='signature_noarea',
        dataset='IEMOCAPSix', seed=68, fold=args.fold, rate=rate, pid=os.getpid(),
        label='INTERNAL TEST-ORACLE SCREEN')
    try:
        reference_dir = REFERENCE_ROOT / f'IEMOCAPSix_68_fold{args.fold}_{rate}'
        original = json.loads((reference_dir / 'SUMMARY.json').read_text())
        original_history = json.loads((reference_dir / 'history.json').read_text())
        validate_history(original_history, budget())
        selected_row(dict(output=str(reference_dir), seed=68, fold=args.fold, rate=rate), NUMERICAL_SOURCE)
        numerical_hashes = original['identity']['source_hashes']
        if any(h.sha(prepare.ROOT / name) != digest for name, digest in numerical_hashes.items()):
            raise ValueError('Original immutable numerical source hashes differ')
        bundle, reader, parent = prepare.setup(68, args.fold, args.gpu_uuid, args.output, rate)
        bundle['cfg'] = replace(bundle['cfg'], train_rate_mode='cyclic', fixed_missing_rate=None,
            train_missing_rates=tuple(float(r) for r in prepare.RATES), epochs=budget(), learning_rate=3e-4)
        rows = prepare.test_cache(bundle, reader, rate)
        with h.isolated_rng(68):
            block = NoAreaSignatureBlock().to('cuda:0')
        before = h.backbone_digest(reader.parent)
        baseline, initial = evaluate(reader, block, rows)
        np.testing.assert_array_equal(initial['prediction'], initial['base_prediction'])
        mask_sha = hashlib.sha256(initial['availability'].tobytes()).hexdigest()
        initial_sha = h.backbone_digest(block)
        validate_reference(original, parent, asdict(bundle['cfg']), initial_sha, mask_sha, args.fold, rate)
        if sum(p.numel() for p in block.parameters()) != 281856:
            raise ValueError('No-area control must preserve original parameter count')
        audit_root = Path(__file__).resolve().parents[2]
        files = sorted(Path(__file__).parent.glob('*.py'))
        identity = dict(mode='signature_noarea', ablation='degree2_no_area', definition=DEFINITION,
            dataset='IEMOCAPSix', seed=68,
            fold=args.fold, source_commit=args.source_commit, numerical_source_commit=NUMERICAL_SOURCE,
            smoke=args.smoke, parent=parent, training_rates=list(prepare.RATES), evaluation_rate=rate,
            cyclic_training=True, initial_block_sha256=initial_sha, parent_config=parent['config'],
            effective_stage2_config=asdict(bundle['cfg']),
            source_hashes={str(p.relative_to(audit_root)): h.sha(p) for p in files},
            numerical_source_hashes=numerical_hashes,
            reference_signature=dict(output=str(reference_dir),
                summary=str(reference_dir / 'SUMMARY.json'), summary_sha256=h.sha(reference_dir / 'SUMMARY.json'),
                history=str(reference_dir / 'history.json'), history_sha256=h.sha(reference_dir / 'history.json'),
                checkpoint=str(reference_dir / original['best']['checkpoint']),
                checkpoint_sha256=original['best']['checkpoint_sha256'], best=original['best']),
            rng_repair=repair, mask_protocol='gcnet-random-mask-v1', task='original six-class CE',
            test_mask_sha256=mask_sha, stage2_epochs=budget(),
            selection='one current-rate TEST ACC; earliest strict tie; same-epoch UA/WF1/CE',
            algorithm='Original Signature with only antisymmetric second-level descriptor removed')
        optimizer = torch.optim.Adam(block.parameters(), lr=3e-4, weight_decay=1e-5)
        start, history, best = 0, [], None
        if args.resume:
            saved = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            if saved['identity'] != identity:
                raise ValueError('Immutable resume identity differs')
            start, history, best = saved['epoch'], saved['history'], saved['best']
            validate_history(history, start)
            if not 0 < start <= (2 if args.smoke else budget()):
                raise ValueError('Saved epoch outside authorized budget')
            block.load_state_dict(saved['block'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            h.restore_rng(saved['rng'])
            for field, digest in [('checkpoint', 'checkpoint_sha256'), ('predictions', 'prediction_sha256')]:
                if h.sha(args.output / best[field]) != best[digest]:
                    raise ValueError('Saved best artifact integrity differs')
        else:
            reset_training_stochasticity(68, args.fold)
        expected_history = original_history
        if args.smoke:
            # Full historical trace hashes cannot equal truncated one-batch smoke traces.
            # Replay the original two-update control using the identical RNG stream.
            current_rng = h.rng_state()
            with h.isolated_rng(68):
                control = SignatureBlock().to('cuda:0')
            control_optimizer = torch.optim.Adam(control.parameters(), lr=3e-4, weight_decay=1e-5)
            reset_training_stochasticity(68, args.fold)
            expected_history = [train_epoch(bundle, reader, control, control_optimizer, e, rate, True)
                                for e in range(2)]
            del control_optimizer, control
            h.restore_rng(current_rng)
        for old_row in history:
            validate_epoch_trace(old_row, expected_history[old_row['epoch'] - 1])
        summary.update(status='running', identity=identity, parent_id=parent['parent_id'], baseline=baseline,
            initial_real_batch_exact=True, environment=bundle['environment'], gpu=bundle['gpu'],
            total_parameters=sum(p.numel() for p in block.parameters()))
        h.write_json(args.output / 'config.json', identity)
        h.write_json(path, summary)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, 2 if args.smoke else budget()):
            optimizer.param_groups[0]['lr'] = learning_rate(epoch)
            row = train_epoch(bundle, reader, block, optimizer, epoch, rate, args.smoke)
            validate_epoch_trace(row, expected_history[epoch])
            row['task_CE'] = row.pop('task_MSE')
            score, arrays = evaluate(reader, block, rows)
            if improves(score, None if best is None else best['metrics']):
                ckpt, pred = f'best_epoch_{epoch + 1}.pt', f'predictions_epoch_{epoch + 1}.npz'
                _atomic_arrays(args.output / pred, arrays)
                h.atomic_torch_save(args.output / ckpt, dict(block=block.state_dict(), identity=identity,
                    epoch=epoch + 1, parent_id=parent['parent_id']))
                best = dict(epoch=epoch + 1, metrics=score, checkpoint=ckpt, predictions=pred,
                    checkpoint_sha256=h.sha(args.output / ckpt), prediction_sha256=h.sha(args.output / pred))
            row['test'] = score
            history.append(row)
            h.atomic_torch_save(args.output / 'last.pt', dict(block=block.state_dict(), optimizer=optimizer.state_dict(),
                identity=identity, epoch=epoch + 1, history=history, best=best, rng=h.rng_state()))
            summary.update(epoch=epoch + 1, best=best, seconds=time.monotonic() - tick,
                peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
            h.write_json(args.output / 'history.json', history)
            h.write_json(path, summary)
            print(json.dumps(dict(dataset='IEMOCAPSix', seed=68, fold=args.fold, rate=rate,
                epoch=epoch + 1, current=score, best=best['metrics'], seconds=summary['seconds'])), flush=True)
        saved = torch.load(args.output / best['checkpoint'], map_location='cpu', weights_only=False)
        block.load_state_dict(saved['block'], strict=True)
        score, replay = evaluate(reader, block, rows)
        with np.load(args.output / best['predictions']) as stored:
            np.testing.assert_array_equal(replay['prediction'], stored['prediction'])
        if score != best['metrics']:
            raise ValueError('Selected checkpoint metrics differ after restoration')
        if not args.smoke and {r for row in history for r in row['training_rates']} != {float(r) for r in prepare.RATES}:
            raise ValueError('Cyclic stage2 did not cover all eight training rates')
        if not any(row['gradients']['nonzero'] for row in history):
            raise RuntimeError('No-area Signature never received nonzero gradient')
        if before != h.backbone_digest(reader.parent) or any(p.grad is not None for p in reader.parent.parameters()):
            raise RuntimeError('Frozen parent changed or received gradient')
        if any(h.sha(audit_root / name) != digest for name, digest in identity['source_hashes'].items()):
            raise RuntimeError('Audit source changed during training')
        if any(h.sha(prepare.ROOT / name) != digest for name, digest in numerical_hashes.items()):
            raise RuntimeError('Immutable original numerical source changed')
        if (h.sha(reference_dir / 'SUMMARY.json') != identity['reference_signature']['summary_sha256']
                or h.sha(reference_dir / 'history.json') != identity['reference_signature']['history_sha256']):
            raise RuntimeError('Original completed reference changed during training')
        summary.update(status='smoke_passed' if args.smoke else 'complete', seconds=time.monotonic() - tick,
            checks=dict(parent_unchanged=True, parent_grad_none=True, checkpoint_replay_exact=True,
                cyclic_train_rates_verified=True, only_requested_test_rate=True,
                original_initialization_exact=True, original_parent_and_test_mask_exact=True,
                original_training_mask_and_optimizer_trace_exact=True),
            peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        h.write_json(args.output / 'RESULT.json', summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
