"""MOSEI66 frozen Nested: one approved architecture factor changes per run."""
import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import time

from .protocol import (CONFIGS, EPOCHS, LR, METHOD, NUMERIC, NUMERICAL_SHA,
    RATES, reference_dir, sha, validate_epoch_trace, validate_reference, validate_scope)
from experiments.osram_frozen_nested_cross_20261025.run import (
    configure_environment, costs, evaluate, starting_summary)


def set_learning_rate(optimizer, lr):
    if not math.isfinite(lr) or lr != LR:
        raise ValueError('Capacity runs require the fixed original Nested learning rate')
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer_learning_rates(optimizer, lr)


def optimizer_learning_rates(optimizer, lr):
    actual = [group['lr'] for group in optimizer.param_groups]
    if not actual or any(value != lr for value in actual):
        raise ValueError('Actual optimizer learning rate differs from the requested configuration')
    return actual


def validate_initial_block(config, digest, count, repeated_digest):
    if count != CONFIGS[config]['parameters'] or digest != repeated_digest:
        raise ValueError('Configured Nested initial state/parameter count differs')

def validate_smoke_prefix(row, reference, lr):
    """A one-batch smoke cannot validate the completed full-epoch mask hash."""
    if (row['epoch'] != reference['epoch'] or row['optimizer_steps'] != 1
            or row['training_rates'] != reference['training_rates'][:1]
            or row['live_scan_calls'] != [1] or not 0 < row['N'] <= reference['N']
            or row['learning_rate'] != lr):
        raise ValueError('Single-batch smoke sampling/update prefix differs')
    return dict(rate_prefix_verified=True, one_optimizer_step_verified=True,
                full_epoch_mask_hash_verified=False)


def completion_checks(smoke):
    """Only applicable proven checks enter the all-true completion contract.

    Smoke deliberately omits full-epoch trace/coverage claims; its individual
    history rows retain the explicit full_epoch_mask_hash_verified=False.
    """
    checks = dict(parent_unchanged=True, parent_grad_none=True, parent_eval=True,
        only_block_optimizer=True, checkpoint_replay_exact=True, selected_all_arrays_exact=True,
        only_requested_test_rate=True, original_parent_and_test_mask_exact=True,
        original_initial_prediction_exact=True, configured_initial_state_reproducible=True, single_architecture_factor_verified=True,
        finite_connected_block_gradients=True, nonzero_branch_gradient_and_update=True,
        no_auxiliary_loss=True, exact_requested_learning_rate_all_groups=True,
        single_live_scan_verified=True)
    if smoke:
        checks['smoke_rate_step_prefix_verified'] = True
    else:
        checks.update(cyclic_train_rates_verified=True, original_training_trace_exact=True)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', choices=tuple(CONFIGS), required=True)
    parser.add_argument('--rate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    rate, lr = validate_scope(args.config, args.rate), LR
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    # These globals must be bound before importing trainer/config or any model.
    prepare = importlib.import_module('experiments.osram_signature_cyclic_multi_20261025.prepare')
    configure_environment('CMUMOSEI', prepare, args.output)
    if prepare.ROOT.resolve() != NUMERIC.resolve() or Path.cwd().resolve() != NUMERIC.resolve():
        raise ValueError('Require pinned e0ca94c numerical working directory with immutable overlay')
    from experiments.osram_three_block_joint_mosi_20261025.scheduler import gpu_inventory, guard_gpu
    host = gpu_inventory().get(args.gpu_uuid, {}).get('host_index')
    if host not in (0, 1, 2):
        raise ValueError('Only healthy physical GPU0/1/2 authorized')
    guard_gpu(args.gpu_uuid, host)
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import _atomic_arrays
    from experiments.osram_recent_noarea_serial_20261025 import stage2 as shared
    from . import model as model_module
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
    summary = starting_summary(previous, 'CMUMOSEI', 66, 1, rate, args.source_commit)
    summary.update(config=args.config, learning_rate=lr, architecture=dict(CONFIGS[args.config]))
    try:
        reference_path = reference_dir(rate)
        reference = json.loads((reference_path/'SUMMARY.json').read_text())
        reference_history = json.loads((reference_path/'history.json').read_text())
        shared.validate_history(reference_history, EPOCHS)
        # This is the completed Nested reference, never deleted Signature weights.
        shared._verify_artifacts(reference_path, reference['best'])
        numeric_hashes = reference['identity']['numerical_source_hashes']
        if any(sha(NUMERIC/name) != digest for name, digest in numeric_hashes.items()):
            raise ValueError('Pinned numerical source differs from the completed Nested reference')
        bundle, reader, parent = prepare.setup('CMUMOSEI', 66, args.gpu_uuid, args.output, rate)
        bundle['cfg'] = replace(bundle['cfg'], train_rate_mode='cyclic', fixed_missing_rate=None,
            train_missing_rates=tuple(float(r) for r in RATES), epochs=EPOCHS, learning_rate=lr)
        rows = prepare.test_cache(bundle, reader, rate)
        with h.isolated_rng(66):
            model = model_module.VariantFrozenNestedModel(reader.parent, args.config).to('cuda:0')
        parent_before = h.backbone_digest(model.parent)
        baseline, initial = evaluate('CMUMOSEI', model, rows)
        np.testing.assert_array_equal(initial['prediction'], initial['base_prediction'])
        mask_sha = hashlib.sha256(initial['availability'].tobytes()).hexdigest()
        validate_reference(reference, parent, asdict(bundle['cfg']), mask_sha, rate)
        if baseline != reference['baseline']:
            raise ValueError('Initial parent metrics differ from the completed Nested baseline')
        count = sum(p.numel() for p in model.block.parameters())
        initial_hash = h.backbone_digest(model.block)
        with h.isolated_rng(66):
            repeated = model_module.VariantFrozenNestedModel(reader.parent, args.config)
            repeated_hash = h.backbone_digest(repeated.block)
        del repeated
        validate_initial_block(args.config, initial_hash, count, repeated_hash)
        if parent_before != reference['identity']['parent_state_sha256']:
            raise ValueError('Original frozen parent state differs from completed Nested')
        nested_names = ('meaningful_input', 'meaningful_input_new40', 'meaningful_blocks_common',
                        'meaningful_new40_structure', 'meaningful_new40_registry', 'nested_sweep')
        nested_files = {str(Path(importlib.import_module('gcnet_missing_m3.'+name).__file__).resolve()):
            sha(Path(importlib.import_module('gcnet_missing_m3.'+name).__file__).resolve()) for name in nested_names}
        if any(not Path(name).is_relative_to(NUMERIC.resolve()) for name in nested_files):
            raise ValueError('Nested numerical modules must resolve from pinned e0ca94c')
        source_files = [Path(__file__).resolve(), Path(__file__).with_name('protocol.py'),
            Path(importlib.import_module('experiments.osram_frozen_nested_gcnet_20261025.model').__file__).resolve(),
            Path(model_module.__file__).resolve(), Path(shared.__file__).resolve(),
            Path(importlib.import_module('experiments.osram_frozen_nested_cross_20261025.run').__file__).resolve()]
        identity = dict(stage='frozen', mode='nested', method=CONFIGS[args.config]['method'], dataset='CMUMOSEI', seed=66,
            fold=1, evaluation_rate=rate, config=args.config, learning_rate=lr,
            training_rates=list(RATES), cyclic_training=True, source_commit=args.source_commit,
            numerical_source_commit=NUMERICAL_SHA, numerical_source_root=str(NUMERIC),
            smoke=args.smoke, stage2_epochs=EPOCHS, parent=parent, parent_state_sha256=parent_before,
            initial_block_sha256=initial_hash, effective_stage2_config=asdict(bundle['cfg']),
            learning_rate_formula=f'{lr} constant for all Adam parameter groups',
            source_hashes={str(p): sha(p) for p in source_files},
            numerical_source_hashes=numeric_hashes, nested_source_hashes=nested_files,
            environment=bundle['environment'], data_files=bundle['data_files'], split=bundle['split'],
            seed_protocol=bundle['seed_protocol'], mask_protocol='gcnet-random-mask-v1',
            test_mask_sha256=mask_sha, rng_repair=repair,
            reference_nested=dict(output=str(reference_path), summary_sha256=sha(reference_path/'SUMMARY.json'),
                history_sha256=sha(reference_path/'history.json')),
            task='original MSE including gold0; no auxiliary loss',
            selection='own-rate TEST WF1; strict improvement/earliest tie',
            architecture={key: CONFIGS[args.config][key] for key in ('dim','depth','mlp_hidden','groups')},
            placement='raw original LocalSkip unchanged; correction only at original Flat inputs',
            recovery='block/Adam/RNG/history plus immutable original Flat parent')
        identity = json.loads(json.dumps(identity))
        optimizer = shared.build_optimizer(model)
        set_learning_rate(optimizer, lr)
        start, history, best = 0, [], None
        limit = 2 if args.smoke else EPOCHS
        if args.resume:
            saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            shared._verify_artifacts(args.output, saved['best'])
            start, history, best = shared.restore_checkpoint(model, optimizer, saved, identity, limit)
            optimizer_learning_rates(optimizer, lr)
            if sha(args.output/'baseline.npz') != previous['baseline_artifact']['sha256']:
                raise ValueError('Baseline artifact changed during resume')
        else:
            _atomic_arrays(args.output/'baseline.npz', initial)
            reset_training_stochasticity(66, 1)
        for row in history:
            if args.smoke:
                validate_smoke_prefix(row, reference_history[row['epoch']-1], lr)
            else:
                validate_epoch_trace(row, reference_history[row['epoch']-1], lr)
        summary.update(status='running', identity=identity, parent_id=parent['parent_id'], baseline=baseline,
            baseline_artifact=dict(path='baseline.npz', sha256=sha(args.output/'baseline.npz')),
            environment=bundle['environment'], gpu=bundle['gpu'], total_parameters=count,
            full_parameters=sum(p.numel() for p in model.parameters()), epoch=start, best=best,
            optimizer_learning_rates=optimizer_learning_rates(optimizer, lr))
        h.write_json(args.output/'config.json', identity)
        h.write_json(path, summary)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, limit):
            set_learning_rate(optimizer, lr)
            row = shared.train_epoch(bundle, model, optimizer, epoch, rate, args.smoke)
            if row['live_scan_calls'] != [1]*row['optimizer_steps']:
                raise RuntimeError('Require exactly one original live CHM scan per training step')
            row['optimizer_learning_rates'] = optimizer_learning_rates(optimizer, lr)
            if args.smoke:
                row['trace_validation'] = validate_smoke_prefix(row, reference_history[epoch], lr)
            else:
                validate_epoch_trace(row, reference_history[epoch], lr)
            score, arrays = evaluate('CMUMOSEI', model, rows)
            if not all(math.isfinite(value) for value in score.values()):
                raise RuntimeError('Nonfinite evaluation metrics')
            if best is None or score['WF1'] > best['metrics']['WF1']:
                checkpoint, prediction = f'best_epoch_{epoch+1}.pt', f'predictions_epoch_{epoch+1}.npz'
                _atomic_arrays(args.output/prediction, arrays)
                h.atomic_torch_save(args.output/checkpoint, dict(version=1, block=model.block.state_dict(),
                    identity=identity, epoch=epoch+1, parent_id=parent['parent_id']))
                best = dict(epoch=epoch+1, metrics=score, checkpoint=checkpoint, predictions=prediction,
                    checkpoint_sha256=sha(args.output/checkpoint), prediction_sha256=sha(args.output/prediction))
            row['test'] = score
            history.append(row)
            h.atomic_torch_save(args.output/'last.pt', shared.checkpoint_payload(model, optimizer, identity,
                epoch+1, history, best))
            summary.update(epoch=epoch+1, best=best, **costs(previous, time.monotonic()-tick,
                torch.cuda.max_memory_allocated()/2**20, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
            h.write_json(args.output/'history.json', history)
            h.write_json(path, summary)
            print(json.dumps(dict(config=args.config, dataset='CMUMOSEI', seed=66, rate=rate,
                epoch=epoch+1, learning_rate=lr, current=score, best=best['metrics'])), flush=True)
        shared._verify_artifacts(args.output, best)
        saved = torch.load(args.output/best['checkpoint'], map_location='cpu', weights_only=False)
        if saved['identity'] != identity or saved['parent_id'] != parent['parent_id']:
            raise ValueError('Selected checkpoint identity/parent mismatch')
        model.block.load_state_dict(saved['block'], strict=True)
        score, replay = evaluate('CMUMOSEI', model, rows)
        with np.load(args.output/best['predictions'], allow_pickle=False) as stored:
            if set(replay) != set(stored.files):
                raise ValueError('Selected array fields differ')
            for key in replay:
                np.testing.assert_array_equal(replay[key], stored[key])
        if score != best['metrics']:
            raise ValueError('Selected checkpoint metrics do not replay')
        if not any(row['nonzero_gradient_batches'] and row['updated_parameters'] for row in history):
            raise RuntimeError('No actual Nested task gradient/parameter update')
        if not args.smoke and {r for row in history for r in row['training_rates']} != {float(r) for r in RATES}:
            raise RuntimeError('A cyclic training rate was omitted')
        shared._check_frozen(model)
        if h.backbone_digest(model.parent) != parent_before:
            raise RuntimeError('Frozen parent changed')
        for mapping, prefix in ((identity['source_hashes'], Path('/')),
                                (numeric_hashes, NUMERIC), (nested_files, Path('/'))):
            if any(sha(prefix/name) != digest for name, digest in mapping.items()):
                raise RuntimeError('Immutable source changed during training')
        proof = identity['reference_nested']
        if (sha(reference_path/'SUMMARY.json') != proof['summary_sha256']
                or sha(reference_path/'history.json') != proof['history_sha256']):
            raise RuntimeError('Completed default Nested reference changed')
        summary.update(status='smoke_passed' if args.smoke else 'complete',
            **costs(previous, time.monotonic()-tick, torch.cuda.max_memory_allocated()/2**20,
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20),
            checks=completion_checks(args.smoke))
        h.write_json(args.output/'RESULT.json', summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error),
            seconds=(previous or {}).get('seconds', 0.)+time.monotonic()-tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()

