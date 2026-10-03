"""One locked, source-verified meaningful-block run; no protocol tuning."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import sys

from .manifest import digest_json, now, read, sha, validate_round, verify_snapshot, write
from .preflight import admission, query_gpus, validate_gpu, validate_readiness
from .summarize import RATES, scores

LABEL = 'INTERNAL ADAPTIVE PER-RATE TEST-ORACLE SEARCH; NOT A FORMAL PAPER RESULT'
EXPECTED = dict(dataset='CMUMOSI', epochs=100, batch_size=32, optimizer='adam',
                learning_rate=.001, weight_decay=.00001, training_objective='emotion-only',
                task_regression_loss='mse', mosi_task_mode='regression', completion_path='none',
                train_rate_mode='cyclic', train_missing_rates=[i / 10 for i in range(8)],
                checkpoint_selection='test-oracle-per-rate', evaluation_protocol='official',
                evaluate_test=True, osram_bidirectional=False, osram_write_step=.6,
                osram_readout_fusion='flat', osram_ablation='full', osram_emotion_ablation='full',
                osram_gap_read='residual', osram_output_dim=1600, osram_num_heads=8,
                osram_key_dim=64, osram_value_dim=64, latent_dim=256,
                osram_forward_slot_reuse=False, backbone_type='osram', device='cuda',
                lr_schedule='constant', readout_type='shared', disable_unused_aux_modules=True)
FORBIDDEN = ('paired_history_views', 'osram_relation_block', 'osram_relation_dual_readout',
             'osram_decision_correction', 'osram_gap_increment_filter', 'osram_post_grn',
             'osram_history_query_adapter', 'classification_completion', 'osram_local_skip_gate',
             'osram_memory_only_adapter', 'osram_history_input_gate', 'osram_local_evidence_gate',
             'osram_hierarchical_evidence_gate', 'osram_hierarchical_feature_only',
             'completion_write_to_memory', 'local_context_residual', 'node_interaction_residual', 'text_core')


def candidate_config(baseline, candidate, *, seed):
    if seed not in (66, 67, 68) or baseline.get('seed') != seed:
        raise ValueError('Use an exact same-seed Flat reference')
    for key, expected in EXPECTED.items():
        if baseline.get(key) != expected: raise ValueError(f'Baseline protocol mismatch: {key}')
    for key in FORBIDDEN:
        if baseline.get(key, False): raise ValueError(f'Incompatible baseline intervention: {key}')
    for key in ('osram_readout_candidate', 'osram_meaningful_block'):
        if baseline.get(key, 'none') != 'none': raise ValueError(f'Baseline already enables {key}')
    for key in ('initial_backbone_checkpoint', 'b2_base_checkpoint', 'b2_pretrain_checkpoint',
                'teacher_checkpoint', 'text_subspace_checkpoint', 'joint_pretrain_checkpoint'):
        if baseline.get(key) is not None: raise ValueError('A from-scratch reference is required')
    return dict(baseline, osram_meaningful_block=candidate)


def completion_status(output):
    output = Path(output)
    if not output.exists(): return 'missing', 'no output directory'
    try:
        record = read(output / 'PROVENANCE.json')
        if record.get('status') == 'failed': return 'failed', record.get('error', 'run failed')
        if record.get('status') != 'complete' or record.get('exit_code') != 0 or not record.get('outputs_verified'):
            return 'incomplete', 'no completed process and artifact audit'
        metrics, history = read(output / 'metrics.json'), read(output / 'history.json')
        scores(metrics)
        if len(history) != 100 or [r.get('epoch') for r in history] != list(range(1, 101)):
            return 'incomplete', 'not exactly 100 complete epochs'
        if set(metrics.get('selected_epoch_by_rate', {})) != set(RATES):
            return 'incomplete', 'missing selected epochs'
        required = ['metrics.json', 'history.json', 'config.json', 'RAW_CONFIG.json',
                    'PARAMETERS.json', 'last_training.pt']
        required += [f'best_miss_{r.replace(".", "p")}.pt' for r in RATES]
        required += [f'predictions_miss_{r.replace(".", "p")}.npz' for r in RATES]
        for name in required:
            path = output / name
            if not path.is_file() or path.stat().st_size == 0 or record.get('artifact_sha256', {}).get(name) != sha(path):
                return 'incomplete', f'missing/corrupt artifact: {name}'
        for key in ('source_sha256', 'reference_sha256', 'code_commit', 'manifest_sha256',
                    'data_manifest_sha256', 'environment', 'gpu_uuid', 'process_identity'):
            if not record.get(key): return 'incomplete', f'missing provenance: {key}'
        if str(record.get('gpu_index')) == '4': return 'incomplete', 'forbidden GPU'
        return 'complete', None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return 'incomplete', str(error)


def validate_data_manifest(path):
    data = read(path)
    roots = data.get('feature_roots', [])
    dataset_root = Path(data.get('dataset_root', ''))
    if not dataset_root.is_absolute() or not dataset_root.is_dir():
        raise ValueError('Explicit absolute dataset_root is required')
    dataset_root = dataset_root.resolve()
    label = dataset_root / 'CMUMOSI' / 'CMUMOSI_features_raw_2way.pkl'
    splits = data.get('split_files', [])
    if (len(roots) != 3 or not data.get('files') or str(label) not in splits
            or any(not Path(p).is_absolute() or not Path(p).resolve().is_relative_to(dataset_root)
                   or p not in data['files'] for p in splits)):
        raise ValueError('Three feature roots and hashed canonical CMUMOSI labels/splits are required')
    expected = set()
    for root in roots:
        root = Path(root)
        if (not root.is_absolute() or not root.is_dir()
                or not root.resolve().is_relative_to(dataset_root)):
            raise ValueError('Missing absolute feature root inside declared dataset root')
        expected.update(str(p.resolve()) for p in root.rglob('*') if p.is_file())
    if not expected or not expected <= set(data['files']):
        raise ValueError('Data manifest does not cover every feature file')
    for name, expected_hash in data['files'].items():
        if sha(name) != expected_hash: raise ValueError(f'Data file hash mismatch: {name}')
    return tuple(roots)


def bind_data_environment(manifest_path):
    """Validate data before any config import, then bind the explicit label root."""
    roots = validate_data_manifest(manifest_path)
    data = read(manifest_path)
    dataset_root = Path(data['dataset_root']).resolve()
    label = dataset_root / 'CMUMOSI' / 'CMUMOSI_features_raw_2way.pkl'
    for name in ('config', 'gcnet.config'):
        loaded = sys.modules.get(name)
        if loaded is not None:
            configured = getattr(loaded, 'PATH_TO_LABEL', {}).get('CMUMOSI')
            if configured is None or Path(configured).resolve() != label:
                raise ValueError('Config was imported before binding the audited dataset root')
    os.environ['GCNET_DATASET_ROOT'] = str(dataset_root)
    os.environ['GCNET_CACHE_ROOT'] = str(Path(manifest_path).resolve().parent / ('cache_' + sha(manifest_path)[:16]))
    return roots


def parameter_counts(config):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    counts = {}
    for block in ('none', config['osram_meaningful_block']):
        with torch.random.fork_rng(devices=[]):
            model = _build_model(TrainConfig(**dict(config, osram_meaningful_block=block)), (512, 1024, 1024))
        counts[block] = {'total': sum(p.numel() for p in model.parameters()),
                         'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        del model
    counts['added'] = counts[config['osram_meaningful_block']]['total'] - counts['none']['total']
    return counts


def make_training_state(output, identity):
    from gcnet_missing_m3.training_resume import TrainingState
    # The trainer binds the actual sampler indices/seed and mask config hashes.
    return TrainingState(output, identity=identity)


def verify_launch_hashes(args):
    for name in ('manifest', 'baseline_audit', 'data_manifest', 'readiness'):
        if sha(getattr(args, name)) != getattr(args, name + '_sha256'):
            raise ValueError(f'Pinned launch input changed: {name}')


def train(args):
    from .queue import process_identity
    verify_launch_hashes(args)
    manifest = read(args.manifest)
    ids = validate_round(manifest)
    if args.candidate not in ids: raise ValueError('Candidate not in accepted twenty')
    card = next(c for c in manifest['cards'] if c['id'] == args.candidate)
    source = verify_snapshot(args.snapshot)
    actual_repo = Path(__file__).resolve().parents[2]
    if actual_repo != args.snapshot.resolve(): raise ValueError('Run from the immutable snapshot, not mutable worktree')
    readiness = read(args.readiness)
    profile = validate_readiness(readiness, candidate=args.candidate,
                                 design_sha256=card['design_sha256'], source_sha256=source['source_sha256'])
    reference = args.reference_root / f'seed_{args.seed}'
    audit = next(r for r in read(args.baseline_audit)['runs'] if r['seed'] == args.seed)
    for filename, key in [('config.json', 'config'), ('metrics.json', 'metrics')]:
        if sha(reference / filename) != audit['sha256'][key]: raise ValueError('Exact baseline audit mismatch')
    if read(reference / 'PROVENANCE.json').get('status') != 'complete': raise ValueError('Incomplete baseline')
    config = candidate_config(read(reference / 'config.json'), args.candidate, seed=args.seed)
    feature_roots = bind_data_environment(args.data_manifest)
    verify_launch_hashes(args)
    resources = query_gpus()
    mapping = {i: row['uuid'] for i, row in resources.items()}
    validate_gpu(args.gpu, args.gpu_uuid, mapping)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('Child visibility must equal one audited healthy UUID')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    admitted, reason = admission(resources[args.gpu], profile,
                                 disk_free_gib=shutil.disk_usage(args.output.parent).free / 1024 ** 3)
    if not admitted: raise RuntimeError(f'Admission changed before child start: {reason}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / (args.output.name + '.run.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if not (args.output / 'last_training.pt').is_file(): raise ValueError('No complete checkpoint for resume')
            previous = read(args.output / 'PROVENANCE.json')
            if previous.get('run_id') != args.run_id: raise ValueError('Cannot resume a different run identity')
        else:
            args.output.mkdir(exist_ok=False)
            previous = None
        process = dict(process_identity(os.getpid()), run_id=args.run_id)
        write(args.output / 'RUN_IDENTITY.json', process)
        identity = {'source': digest_json(source['source_sha256']), 'config': digest_json(config),
                    'design': card['design_sha256'], 'data': sha(args.data_manifest),
                    'manifest': sha(args.manifest)}
        if previous and previous.get('identity') != identity: raise ValueError('Resume source/config/data mismatch')
        attempts_path = args.output / 'ATTEMPTS.json'
        attempts = read(attempts_path) if attempts_path.exists() else []
        if previous: attempts.append(previous)
        write(attempts_path, attempts)
        record = dict(status='starting', label=LABEL, candidate=args.candidate, seed=args.seed,
                      run_id=args.run_id, identity=identity, config=config, server='biggpu',
                      hostname=socket.gethostname(), gpu_index=args.gpu, gpu_uuid=args.gpu_uuid,
                      process_identity=process, started_at=now(), resumed=args.resume,
                      code_commit=source['code_commit'], scoped_diff=source.get('scoped_diff', ''),
                      source_sha256=source['source_sha256'], manifest_sha256=sha(args.manifest),
                      readiness_sha256=sha(args.readiness), data_manifest_sha256=sha(args.data_manifest),
                      dataset_root=os.environ['GCNET_DATASET_ROOT'], cache_root=os.environ['GCNET_CACHE_ROOT'],
                      original_baseline_run_commit=read(args.baseline_audit).get('original_run_commit'),
                      baseline_commit_note=read(args.baseline_audit).get('original_run_commit_note'),
                      reference_sha256={n: sha(reference / n) for n in ('config.json', 'metrics.json')})
        write(args.output / 'PROVENANCE.json', record)
        try:
            import torch
            from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
            from experiments.osram_cfg84_history_query_random_20260928.run import verify_outputs
            torch.set_num_threads(2)
            if torch.cuda.device_count() != 1: raise ValueError('Expected exactly one visible GPU')
            effective = TrainConfig(**config)
            off = TrainConfig(**dict(config, osram_meaningful_block='none'))
            effective_json = json.loads(json.dumps(asdict(effective)))
            off_json = json.loads(json.dumps(asdict(off)))
            changes = {key for key in effective_json if effective_json[key] != off_json[key]}
            if changes != {'osram_meaningful_block'}: raise ValueError('Effective protocol drift')
            record.update(status='training', effective_config=effective_json,
                schema_defaults={k: v for k, v in off_json.items() if k not in config},
                environment={'python': platform.python_version(), 'executable': os.sys.executable,
                             'torch': torch.__version__, 'cuda': torch.version.cuda,
                             'cudnn': torch.backends.cudnn.version(), 'gpu_name': torch.cuda.get_device_name(0)})
            write(args.output / 'RAW_CONFIG.json', config)
            write(args.output / 'PARAMETERS.json', parameter_counts(config))
            write(args.output / 'PROVENANCE.json', record)
            state = make_training_state(args.output, identity)
            run_experiment(effective, *feature_roots, output_dir=str(args.output), training_state=state)
            verify_outputs(args.output, reference)
            metrics = read(args.output / 'metrics.json')
            scores(metrics)
            if read(args.output / 'config.json') != effective_json: raise ValueError('Saved config drift')
            for rate in RATES:
                best = torch.load(args.output / f'best_miss_{rate.replace(".", "p")}.pt', map_location='cpu', weights_only=False)
                if (best.get('epoch') != metrics['selected_epoch_by_rate'][rate]
                        or best.get('selection_protocol') != 'per-rate-test-oracle'):
                    raise ValueError('BEST metadata mismatch')
                del best
            verify_snapshot(args.snapshot)
            verify_launch_hashes(args)
            if len(read(args.output / 'history.json')) != 100: raise ValueError('Incomplete training history')
            record.update(status='complete', exit_code=0, outputs_verified=True,
                artifact_sha256={str(p.relative_to(args.output)): sha(p) for p in args.output.rglob('*')
                                 if p.is_file() and p.suffix in ('.pt', '.json', '.npz')
                                 and p.name not in ('PROVENANCE.json', 'ATTEMPTS.json', 'RUN_IDENTITY.json')})
        except BaseException as error:
            message = str(error)
            category = ('oom' if 'out of memory' in message.lower() else
                        'nonfinite' if 'nan' in message.lower() or 'nonfinite' in message.lower() else
                        'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'execution_error')
            record.update(status='failed', exit_code=1, error=repr(error), failure_category=category)
            raise
        finally:
            record['updated_at'] = now()
            write(args.output / 'PROVENANCE.json', record)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'manifest', 'snapshot', 'readiness', 'reference-root', 'baseline-audit', 'data-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('candidate', 'manifest-sha256', 'baseline-audit-sha256', 'data-manifest-sha256',
                 'readiness-sha256', 'gpu', 'gpu-uuid', 'run-id'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--seed', type=int, choices=(66, 67, 68), required=True)
    p.add_argument('--resume', action='store_true')
    return p


def main():
    args = parser().parse_args()
    try:
        train(args)
    except BaseException as error:
        # Preflight can fail before an output directory exists. Preserve a
        # separate marker so the coordinator will not repeatedly retry it.
        write(args.output.parent / (args.output.name + '.launch-failure.json'),
              {'run_id': args.run_id,
               'failure_category': 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'preflight_or_execution',
               'error': repr(error), 'at': now()})
        raise


if __name__ == '__main__': main()
