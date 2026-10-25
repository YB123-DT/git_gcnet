"""Fresh jointly trained MOSI66 parallel/serial NoArea; INTERNAL TEST-ORACLE SCREEN."""
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
NUMERICAL_ROOT = Path('/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c')
NUMERICAL_COMMIT = 'e0ca94c4c33de0efda2c2f4786fd289067e1ab2c'
ACCEPTED = Path('/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2')
MODES = ('parallel', 'serial')
RATES = tuple(str(i / 10) for i in range(8))


def epoch_budget(smoke):
    return 3 if smoke else 150


def epoch_lr(epoch):
    return 3e-4 if epoch < 100 else 3e-4 / 10


def validate_protocol(cfg):
    expected = dict(dataset='CMUMOSI', seed=66, fold=1, epochs=150, learning_rate=3e-4,
        weight_decay=1e-5, batch_size=32, optimizer='adam',
        missing_mask_protocol='gcnet-random-mask-v1', checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1', train_rate_mode='cyclic',
        mosi_task_mode='regression', task_regression_loss='mse',
        training_objective='emotion-only', evaluation_protocol='official')
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise ValueError('Original joint protocol differs: ' + key)
    if tuple(cfg.train_missing_rates) != tuple(float(r) for r in RATES):
        raise ValueError('Original cyclic train_missing_rates differ')


def is_improvement(metrics, previous):
    return previous is None or metrics['router']['WF1'] > previous['metrics']['router']['WF1']


def validate_block_connection(row):
    # The original zero Flat outlet blocks serial task gradients on update1.
    # A connected zero derivative is valid; finite gradients are checked by train_epoch.
    if not row['gradients'].get('block', {}).get('connected'):
        raise RuntimeError('New NoArea block is disconnected from the task')


def validate_block_progress(history):
    if not any(row['gradients'].get('block', {}).get('nonzero') for row in history):
        raise RuntimeError('New NoArea block never received nonzero task gradients')


def validate_masks(masks, expected):
    hashes = {rate: hashlib.sha256(b''.join(row['availability'].numpy().tobytes()
              for row in rows)).hexdigest() for rate, rows in masks['test'].items()}
    if set(hashes) != set(RATES) or hashes != expected:
        raise ValueError('Accepted exact TEST masks changed')
    return hashes


def source_digest(groups):
    result = {}
    for name, (root, files) in groups.items():
        result[name] = {}
        for relative in sorted(set(map(str, files))):
            path = (root / relative).resolve()
            path.relative_to(root.resolve())
            result[name][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_source_digest(expected, numerical_root=NUMERICAL_ROOT, overlay_root=ROOT):
    actual = source_digest({'numerical': (numerical_root, expected['numerical']),
                            'overlay': (overlay_root, expected['overlay'])})
    if actual != expected:
        raise ValueError('Source snapshot changed during training')


def restore_training(model, optimizer, generator, state, identity):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    required = {'model', 'optimizer', 'identity', 'epoch', 'history', 'best_by_rate', 'rng', 'noise_rng'}
    if not required.issubset(state):
        raise ValueError('Incomplete full training checkpoint')
    if state['identity'] != identity:
        raise ValueError('Resume identity mismatch')
    if state['epoch'] != len(state['history']):
        raise ValueError('Full checkpoint epoch/history mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    h.restore_rng(state['rng'])
    generator.set_state(state['noise_rng'])
    return state['epoch'], state['history'], state['best_by_rate']


@contextmanager
def memory_scan_probe(model):
    """Count the real memory implementation, with a per-forward one-scan assertion."""
    memory = model.parent.osram
    original = memory._scan
    sentinel = object()
    previous = memory.__dict__.get('_scan', sentinel)
    counts, starts = dict(forwards=0, scans=0), []

    def scan(*args, **kwargs):
        counts['scans'] += 1
        return original(*args, **kwargs)

    def before(module, args):
        starts.append(counts['scans'])

    def after(module, args, output):
        if counts['scans'] - starts.pop() != 1:
            raise RuntimeError('Every joint forward must execute exactly one actual Memory._scan')
        counts['forwards'] += 1

    memory._scan = scan
    handles = [model.register_forward_pre_hook(before), model.register_forward_hook(after)]
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()
        if previous is sentinel:
            del memory._scan
        else:
            memory._scan = previous


def guard_gpu(uuid):
    if not uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('An explicit visible healthy GPU UUID is required')
    listing = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
        '--format=csv,noheader,nounits'], text=True)
    inventory = {value.strip(): int(index) for index, value in
                 (line.split(',') for line in listing.splitlines())}
    if inventory.get(uuid) not in (0, 1, 2):
        raise ValueError('Only healthy physical GPU0/1/2 is permitted')
    return dict(host_index=inventory[uuid], uuid=uuid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=MODES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    physical_gpu = guard_gpu(args.gpu_uuid)  # No torch/CUDA import before physical identity check.
    import fcntl
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_mosi_cmerc_20261023 import worker as control
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_signature_stage2_extension_20261025 import prepare as bridge
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_three_block_joint_mosi_20261025 import run as original
    from experiments.osram_signature_noarea_iemocap68_20261025 import block as noarea_source
    from .model import RecentNoAreaModel

    torch.set_num_threads(1)
    repair = repair_uuid_rng(torch)
    # Namespace overlays contain only this experiment. Numerical helpers must resolve
    # to the immutable original snapshot, never infer that snapshot from overlay ROOT.
    for module in (control, selected, bridge, original, h):
        Path(module.__file__).resolve().relative_to(NUMERICAL_ROOT.resolve())
    if (args.output / 'SUMMARY.json').exists():
        prior_status = json.loads((args.output / 'SUMMARY.json').read_text()).get('status')
        if not args.resume or prior_status in ('complete', 'smoke_passed'):
            raise FileExistsError('Explicit resume of an unfinished run required')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tick, summary_path = time.monotonic(), args.output / 'SUMMARY.json'
    summary = dict(status='preflight', stage='joint', mode=args.mode, seed=66,
        source_commit=args.source_commit, numerical_commit=NUMERICAL_COMMIT, pid=os.getpid(),
        label='INTERNAL TEST-ORACLE SCREEN', gpu=physical_gpu)
    try:
        prior = json.loads((control.REFERENCE / 'SUMMARY.json').read_text())
        accepted = json.loads((ACCEPTED / 'SUMMARY.json').read_text())
        if accepted['status'] != 'complete' or accepted['epoch'] != 150:
            raise ValueError('Complete accepted original Flat150 required')
        reference_history = json.loads((ACCEPTED / 'history.json').read_text())
        if len(reference_history) != 150:
            raise ValueError('Complete accepted Flat training history required')
        bundle = control.setup(args.gpu_uuid, args.output, prior)
        parent, cfg = bundle['model'], bundle['cfg']
        validate_protocol(cfg)
        reference_cfg, added_defaults = bridge._config(prior['identity']['config'], 'CMUMOSI', 66)
        if asdict(cfg) != asdict(reference_cfg):
            raise ValueError('Original configuration/default bridge differs')
        accepted_cfg, _ = bridge._config(accepted['identity']['config'], 'CMUMOSI', 66)
        if asdict(cfg) != asdict(accepted_cfg):
            raise ValueError('Accepted baseline configuration differs')
        for key in ('split', 'seed_protocol'):
            if bundle[key] != accepted['identity'][key]:
                raise ValueError('Accepted baseline identity differs: ' + key)
        # A reconstructed manifest's origin names its immutable source directory;
        # compare protected bytes, rather than requiring identical snapshot paths.
        if bundle['data_manifest']['data_sha256'] != accepted['identity']['data_manifest']['data_sha256']:
            raise ValueError('Accepted baseline data manifest bytes differ')
        for key in ('torch', 'numpy', 'sklearn'):
            if bundle['environment'][key] != accepted['environment'][key]:
                raise ValueError('Accepted baseline environment differs: ' + key)
        core = {p: digest for p, digest in accepted['identity']['source_hashes'].items()
                if p.startswith(('gcnet_missing_m3/', 'gcnet_modality_jepa/'))}
        if not core:
            raise ValueError('Accepted numerical core identity is missing')
        core_differences = bridge._core_differences(core, cfg)
        initial_parent = h.backbone_digest(parent)
        if initial_parent != accepted['initial_backbone_sha256']:
            raise ValueError('Fresh original initialization differs')
        if sum(p.numel() for p in parent.parameters()) != 13509793:
            raise ValueError('Original Flat parameter count changed')
        with h.isolated_rng(66):
            model = RecentNoAreaModel(parent, mode=args.mode, stage='joint').to('cuda:0')
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError('Joint training requires every parent and new parameter trainable')
        initial_block = h.backbone_digest(model.block)
        optimizer = torch.optim.Adam(model.parameters(), lr=epoch_lr(0), weight_decay=1e-5)
        generator = torch.Generator(device='cuda:0').manual_seed(6601)
        masks = torch.load(ACCEPTED / 'fixed_masks.pt', map_location='cpu', weights_only=False)
        hashes = validate_masks(masks, accepted['identity']['mask_hashes']['test'])
        numerical_files = list(prior['identity']['source_hashes']) + list(core)
        numerical_files += [str(Path(m.__file__).resolve().relative_to(NUMERICAL_ROOT))
                            for m in (control, selected, bridge, original, h)]
        numerical_files += ['osram_four_block_wiring/blocks.py', 'osram_four_block_wiring/frozen_flat.py',
                            'osram_four_block_wiring/__init__.py']
        overlay_files = [str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('*.py')]
        overlay_files.append(str(Path(noarea_source.__file__).resolve().relative_to(ROOT)))
        sources = source_digest({'numerical': (NUMERICAL_ROOT, numerical_files),
                                 'overlay': (ROOT, overlay_files)})
        identity = dict(stage='joint', mode=args.mode, source_commit=args.source_commit,
            numerical_commit=NUMERICAL_COMMIT, numerical_root=str(NUMERICAL_ROOT), seed=66,
            smoke=args.smoke, source_hashes=sources, config=asdict(cfg), environment=bundle['environment'],
            data_manifest=bundle['data_manifest'], split=bundle['split'], seed_protocol=bundle['seed_protocol'],
            mask_hashes=hashes, rng_repair=repair, old_weights_loaded=False, initialization=initial_parent,
            initial_block_sha256=initial_block, accepted_summary_sha256=h.sha(ACCEPTED / 'SUMMARY.json'),
            accepted_history_sha256=h.sha(ACCEPTED / 'history.json'),
            reference_summary_sha256=h.sha(control.REFERENCE / 'SUMMARY.json'),
            added_default_config_fields=added_defaults, core_source_differences=core_differences,
            source_bridge='Reviewed disabled-extension defaults; exact fresh initialization and real-batch replay',
            selection='test-oracle-per-rate; weighted_f1; earliest strict tie; ACC from same predictions',
            optimizer='Adam; all parameters; WD1e-5', aux_loss_weight=0,
            rng_protocol='RNG-isolated module init66; reset_training_stochasticity66/fold1; noise6601')
        total = sum(p.numel() for p in model.parameters())
        summary.update(identity=identity, total_parameters=total, parent_parameters=13509793,
            added_parameters=total - 13509793, initial_backbone_sha256=initial_parent,
            environment=bundle['environment'], initial_parameters_exact=True, status='running',
            reference_path=str(ACCEPTED), reference_aggregate=accepted['aggregate'])
        with memory_scan_probe(model) as initial_scans:
            _, initial = original.evaluate(bundle, model, masks['test']['0.3'], '0.3', True)
        _, bare = original.evaluate(bundle, parent, masks['test']['0.3'], '0.3', True)
        for key in ('prediction', 'base_prediction'):
            np.testing.assert_array_equal(initial[key], bare['prediction'])
        summary.update(initial_real_batch_exact=True, initial_actual_memory_scan_probe=initial_scans)
        reset_training_stochasticity(66, cfg.fold)
        start, history, best = 0, [], {}
        if args.resume:
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            start, history, best = restore_training(model, optimizer, generator, state, identity)
            selected.validate_selected(args.output, best)
            if not args.smoke:
                for index, row in enumerate(history):
                    control.check_training_parity(row, reference_history[index])
        h.write_json(args.output / 'config.json', identity)
        h.write_json(summary_path, summary)
        if not args.resume:
            h.atomic_torch_save(args.output / 'fixed_masks.pt', masks)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, epoch_budget(args.smoke)):
            for group in optimizer.param_groups:
                group['lr'] = epoch_lr(epoch)
            with (memory_scan_probe(model) if args.smoke else nullcontext(None)) as train_scans:
                row = original.train_epoch(bundle, model, optimizer, generator, epoch, args.smoke)
            if args.smoke:
                row['actual_training_memory_scan_probe'] = train_scans
            if not args.smoke:
                control.check_training_parity(row, reference_history[epoch])
            validate_block_connection(row)
            row['block_task_gradient_nonzero'] = bool(row['gradients']['block']['nonzero'])
            with (memory_scan_probe(model) if args.smoke else nullcontext(None)) as test_scans:
                evaluated = {r: original.evaluate(bundle, model, masks['test'][r], r, args.smoke) for r in RATES}
            if args.smoke:
                row['actual_test_memory_scan_probe'] = test_scans
            row['test'] = {r: value[0] for r, value in evaluated.items()}
            history.append(row)
            improved = []
            for rate, (metrics, arrays) in evaluated.items():
                if not is_improvement(metrics, best.get(rate)):
                    continue
                stem = 'miss_' + rate.replace('.', 'p') + f'_epoch_{epoch + 1}'
                predpath = args.output / ('predictions_' + stem + '.npz')
                ckptpath = args.output / ('best_' + stem + '.pt')
                selected._atomic_arrays(predpath, arrays)
                h.atomic_torch_save(ckptpath, dict(model=model.state_dict(), identity=identity,
                    config=asdict(cfg), epoch=epoch + 1, rate=rate))
                best[rate] = dict(epoch=epoch + 1, metrics=metrics, checkpoint=ckptpath.name,
                    predictions=predpath.name, checkpoint_sha256=h.sha(ckptpath), prediction_sha256=h.sha(predpath))
                improved.append(rate)
            h.atomic_torch_save(args.output / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                identity=identity, epoch=epoch + 1, history=history, best_by_rate=best,
                rng=h.rng_state(), noise_rng=generator.get_state()))
            h.write_json(args.output / 'history.json', history)
            h.write_json(args.output / 'best_by_rate.json', best)
            summary.update(epoch=epoch + 1, best_by_rate=best, aggregate=original.aggregate(best),
                seconds=time.monotonic() - tick, peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
            h.write_json(summary_path, summary)
            print(json.dumps(dict(stage='joint', mode=args.mode, epoch=epoch + 1,
                task_MSE=row['task_MSE'], mean=summary['aggregate']['router']['mean'],
                seconds=summary['seconds'], improved_rates=improved)), flush=True)
        selected.validate_selected(args.output, best)
        state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        end, restored_history, restored_best = restore_training(model, optimizer, generator, state, identity)
        if end != epoch_budget(args.smoke) or restored_history != history or restored_best != best:
            raise ValueError('Final full checkpoint ledger restore differs')
        if h.backbone_digest(model.parent) == initial_parent or h.backbone_digest(model.block) == initial_block:
            raise RuntimeError('Joint parent or new block failed to update')
        validate_block_progress(history)
        summary['full_checkpoint_restore_verified'] = True
        replay_errors, replay_scan_counts = {}, {}
        for rate in RATES:
            restored = torch.load(args.output / best[rate]['checkpoint'], map_location='cpu', weights_only=False)
            if restored['identity'] != identity or restored['epoch'] != best[rate]['epoch'] or restored['rate'] != rate:
                raise ValueError('Selected checkpoint identity/epoch/rate differs')
            model.load_state_dict(restored['model'], strict=True)
            with (memory_scan_probe(model) if args.smoke else nullcontext(None)) as replay_scans:
                metrics, replay = original.evaluate(bundle, model, masks['test'][rate], rate, args.smoke)
            if args.smoke:
                replay_scan_counts[rate] = replay_scans
            with np.load(args.output / best[rate]['predictions'], allow_pickle=False) as saved:
                for key in ('labels', 'availability', 'conversation_ids', 'utterance_indices'):
                    np.testing.assert_array_equal(replay[key], saved[key])
                for key in ('prediction', 'base_prediction'):
                    np.testing.assert_allclose(replay[key], saved[key], rtol=1e-5, atol=1e-6)
                replay_errors[rate] = float(np.max(np.abs(replay['prediction'] - saved['prediction'])))
            if metrics != best[rate]['metrics']:
                raise ValueError('Replayed selected metrics differ')
        verify_source_digest(sources)
        summary.update(status='smoke_passed' if args.smoke else 'complete', seconds=time.monotonic() - tick,
            epoch=end, best_by_rate=best, aggregate=original.aggregate(best),
            checkpoint_replay_max_abs_by_rate=replay_errors, selected_all_rates_replayed=True,
            actual_replay_memory_scan_probe_by_rate=replay_scan_counts,
            training_inputs_exact=not args.smoke,
            first_nonzero_block_task_gradient_epoch=next(row['epoch'] for row in history
                if row['gradients'].get('block', {}).get('nonzero')),
            peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        h.write_json(args.output / 'RESULT.json', summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        h.write_json(summary_path, summary)


if __name__ == '__main__':
    main()
